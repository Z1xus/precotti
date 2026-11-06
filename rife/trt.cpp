#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "network.h"

namespace {

struct Picture {
    uint8_t* data[3];
    ptrdiff_t stride[3];
    int id;
};

const int MAX_SIDE = 4096;

enum { MAKING, READY, BROKEN };

struct Frame {
    void* data = nullptr;
    cudaEvent_t done = nullptr;
    int id = 0;
    int state = MAKING;

    ~Frame()
    {
        if (done)
            cuda_event_destroy(done);
        if (data)
            cuda_free(data);
    }
};

struct Slot {
    IExecutionContext* pair = nullptr;
    IExecutionContext* frame = nullptr;
    void* memory = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;

    ~Slot()
    {
        // the gpu may still use the buffers of a frame that failed
        if (stream)
            cuda_stream_wait(stream);
        delete pair;
        delete frame;
        if (memory)
            cuda_free(memory);
        if (done)
            cuda_event_destroy(done);
        if (stream)
            cuda_stream_destroy(stream);
        if (device)
            cuda_free(device);
        if (host)
            cuda_free_host(host);
    }
};

struct Rife {
    std::filesystem::path folder;
    Monitor monitor;
    IRuntime* runtime = nullptr;
    ICudaEngine* pair = nullptr;
    ICudaEngine* frame = nullptr;
    bool loaded = false;
    Model model;
    Plan plan;
    std::string engines;
    int padding = 32;
    int width = 0, height = 0, wide = 0, high = 0;
    size_t planes[3] = { }, bytes = 0, tensor = 0;
    int channels = 3;
    // 1 when the color planes are half the size of the picture
    int half = 1;
    float color[15] = { };
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<Slot*> idle;
    std::mutex lock;
    std::condition_variable ready;
    std::mutex cache;
    std::condition_variable settled;
    std::vector<std::shared_ptr<Frame>> held;
    std::mutex spare;
    std::vector<std::unique_ptr<Frame>> pool;

    ~Rife()
    {
        slots.clear();
        held.clear();
        pool.clear();
        delete pair;
        delete frame;
        delete runtime;
    }

    std::shared_ptr<Frame> acquire(int id, bool& fresh)
    {
        const std::lock_guard<std::mutex> guard(cache);
        const auto found = std::find_if(held.begin(), held.end(), [&](const auto& item) { return item->id == id; });
        fresh = found == held.end();
        if (!fresh) {
            std::shared_ptr<Frame> frame = *found;
            held.erase(found);
            held.push_back(frame);
            return frame;
        }
        // the oldest frame goes first, so a free one is used before a new one is made
        if (held.size() >= 4)
            held.erase(held.begin());
        std::unique_ptr<Frame> made;
        {
            const std::lock_guard<std::mutex> inner(spare);
            if (!pool.empty()) {
                made = std::move(pool.back());
                pool.pop_back();
            }
        }
        if (!made) {
            made = std::make_unique<Frame>();
            if (!cuda_ok(cuda_malloc(&made->data, (size_t)channels * wide * high * 2)) || !cuda_ok(cuda_event_create(&made->done, DISABLE_TIMING)))
                return nullptr;
        }
        made->id = id;
        made->state = MAKING;
        std::shared_ptr<Frame> frame(made.release(), [this](Frame* item) {
            const std::lock_guard<std::mutex> inner(spare);
            pool.emplace_back(item);
        });
        held.push_back(frame);
        return frame;
    }

    void publish(const std::shared_ptr<Frame>& frame, int state)
    {
        {
            const std::lock_guard<std::mutex> guard(cache);
            frame->state = state;
            if (state == BROKEN)
                held.erase(std::remove(held.begin(), held.end(), frame), held.end());
        }
        settled.notify_all();
    }

    void drop()
    {
        const std::lock_guard<std::mutex> guard(cache);
        held.clear();
        const std::lock_guard<std::mutex> inner(spare);
        pool.clear();
    }

    bool open(Slot& slot)
    {
        if (!(slot.pair = pair->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED)))
            return false;
        if (!slot.pair->setInputShape("frame0", Dims4(1, channels, high, wide))
            || !slot.pair->setInputShape("frame1", Dims4(1, channels, high, wide)))
            return false;
        size_t memory = slot.pair->updateDeviceMemorySizeForShapes();
        if (frame) {
            if (!(slot.frame = frame->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED))
                || !slot.frame->setInputShape("rgb", Dims4(1, 3, high, wide)))
                return false;
            memory = std::max<size_t>(memory, slot.frame->updateDeviceMemorySizeForShapes());
        }
        // tensorrt wants tensors on 256 bytes
        const auto align = [](size_t size) { return (size + 255) / 256 * 256; };
        const size_t device = align(bytes) + align(tensor) + 256;
        const size_t host = 3 * align(bytes) + 256;
        if (!cuda_ok(cuda_malloc(&slot.memory, memory)) || !cuda_ok(cuda_malloc((void**)&slot.device, device))
            || !cuda_ok(cuda_malloc_host((void**)&slot.host, host)))
            return false;
        slot.pair->setDeviceMemoryV2(slot.memory, (int64_t)memory);
        if (slot.frame)
            slot.frame->setDeviceMemoryV2(slot.memory, (int64_t)memory);
        return cuda_ok(cuda_stream_create(&slot.stream, NON_BLOCKING)) && cuda_ok(cuda_event_create(&slot.done, BLOCKING_SYNC | DISABLE_TIMING));
    }

    ICudaEngine* engine(const std::string& kind, const char* input, const Edit& edit, float base, float span)
    {
        const std::filesystem::path name = folder / (engines + kind + ".engine");
        monitor.base = base;
        monitor.span = span;
        const auto fits = [&](ICudaEngine& found) {
            const Dims low = found.getProfileShape(input, 0, OptProfileSelector::kMIN),
                       top = found.getProfileShape(input, 0, OptProfileSelector::kMAX);
            return low.d[2] <= high && low.d[3] <= wide && top.d[2] >= high && top.d[3] >= wide;
        };
        return load_engine(runtime, name, name.parent_path().parent_path() / "timing.cache", edit, fits, monitor);
    }

    template <class F>
    Edit editor(F body)
    {
        return [body](INetworkDefinition& net, IOptimizationProfile& profile) {
            try {
                body(net, profile);
                return true;
            } catch (const std::runtime_error& error) {
                fail(error.what());
                return false;
            }
        };
    }

    ITensor* input(INetworkDefinition& net, IOptimizationProfile& profile, const char* name, int c)
    {
        ITensor* tensor = net.addInput(name, DataType::kHALF, Dims4(1, c, -1, -1));
        if (!tensor)
            throw std::runtime_error("tensorrt could not change the model");
        // any size up to a square of this side, a larger video builds again
        const int side = std::max({ MAX_SIDE, high, wide });
        profile.setDimensions(name, OptProfileSelector::kMIN, Dims4(1, c, padding, padding));
        profile.setDimensions(name, OptProfileSelector::kOPT, Dims4(1, c, high, wide));
        profile.setDimensions(name, OptProfileSelector::kMAX, Dims4(1, c, side, side));
        tensor->setAllowedFormats(1U << (int)TensorFormat::kLINEAR);
        return tensor;
    }

    void output(INetworkDefinition& net, ITensor* tensor, const char* name)
    {
        tensor->setName(name);
        net.markOutput(*tensor);
        tensor->setAllowedFormats(1U << (int)TensorFormat::kLINEAR);
    }

    // a rife that ran before keeps its engines, and its slots when the size is the same
    bool prepare(int w, int h, bool bt601, bool full, bool yuv444)
    {
        forget();
        const float kr = bt601 ? 0.299f : 0.2126f, kb = bt601 ? 0.114f : 0.0722f, kg = 1 - kr - kb;
        const float luma = full ? 255 : 219, chroma = full ? 255 : 224;
        const float rv = 2 * (1 - kr), bu = 2 * (1 - kb);
        const float numbers[15]
            = { full ? 0.f : 16.f, 1 / luma, 1 / chroma, rv, bu * kb / kg, rv * kr / kg, bu, luma, chroma, kr, kg, kb, 1 / bu, 1 / rv, 0 };
        std::copy(numbers, numbers + 15, color);
        drop();
        if (!slots.empty() && w == width && h == height && half == !yuv444)
            return true;
        slots.clear();
        idle.clear();
        width = w;
        height = h;
        half = !yuv444;
        // the newer models take the picture in larger steps of size
        const std::string name = folder.u8string();
        padding = name.find("rife-v4.25-lite") != std::string::npos                                        ? 128
            : name.find("rife-v4.25") != std::string::npos || name.find("rife-v4.26") != std::string::npos ? 64
                                                                                                           : 32;
        wide = (w + padding - 1) / padding * padding;
        high = (h + padding - 1) / padding * padding;
        // ncnn copies a mat in blocks of 16 bytes
        const auto word = [](size_t count) { return (count + 15) / 16 * 16; };
        const size_t area = (size_t)(w >> half) * (h >> half);
        planes[1] = word((size_t)w * h);
        planes[2] = planes[1] + word(area);
        bytes = planes[2] + word(area);
        tensor = (size_t)3 * wide * high * 2;

        if (!loaded) {
            try {
                model = read_model(folder);
                engines = "graph3-" + std::to_string(model.hash) + "-";
                plan = plan_of(model);
            } catch (const std::exception& error) {
                fail(error.what());
                return false;
            }
            loaded = true;
        }
        channels = plan.cached[0] >= 0 ? 3 + plan.features : 3;
        if (!runtime && !(runtime = (IRuntime*)create_runtime(&logger, NV_TENSORRT_VERSION)))
            return false;
        register_glue();
        delete pair;
        delete frame;
        pair = frame = nullptr;
        std::unique_ptr<Lowering> encoder, joiner;
        const float split = plan.cached[0] >= 0 ? 0.1f : 0.f;
        if (plan.cached[0] >= 0) {
            frame = engine("frame", "rgb", editor([&](INetworkDefinition& net, IOptimizationProfile& profile) {
                encoder = std::make_unique<Lowering>(net, model, plan);
                output(net, encoder->frame(input(net, profile, "rgb", 3), high, wide), "feat");
            }),
                0, split);
            if (!frame)
                return false;
        }
        pair = engine("pair", "frame0", editor([&](INetworkDefinition& net, IOptimizationProfile& profile) {
            joiner = std::make_unique<Lowering>(net, model, plan);
            ITensor* first = input(net, profile, "frame0", channels);
            ITensor* second = input(net, profile, "frame1", channels);
            ITensor* moment = net.addInput("time", DataType::kFLOAT, Dims4(1, 1, 1, 1));
            output(net, joiner->pair(first, second, moment, high, wide), "out0");
        }),
            split, 1 - split);
        if (!pair)
            return false;
        std::error_code error;
        for (const auto& file : std::filesystem::directory_iterator(folder, error)) {
            if (file.path().extension() == ".engine" && file.path().filename().string().rfind(engines, 0) != 0)
                std::filesystem::remove(file.path(), error);
        }
        // more slots gain little
        for (int count = 0; count < 2; count++) {
            slots.push_back(std::make_unique<Slot>());
            if (open(*slots.back()))
                idle.push_back(slots.back().get());
            else {
                slots.pop_back();
                if (idle.empty()) {
                    fail("tensorrt could not set up the engine");
                    return false;
                }
            }
        }
        forget();
        return true;
    }

    Slot* take()
    {
        std::unique_lock<std::mutex> guard(lock);
        ready.wait(guard, [&] { return !idle.empty(); });
        Slot* slot = idle.back();
        idle.pop_back();
        return slot;
    }

    void give(Slot* slot)
    {
        {
            const std::lock_guard<std::mutex> guard(lock);
            idle.push_back(slot);
        }
        ready.notify_one();
    }

    void pack(const Picture& picture, uint8_t* block, bool in)
    {
        for (int plane = 0; plane < 3; plane++) {
            const int shift = plane > 0 ? half : 0;
            const size_t line = (size_t)width >> shift;
            for (int y = 0; y < height >> shift; y++) {
                uint8_t* outside = picture.data[plane] + y * picture.stride[plane];
                uint8_t* inside = block + planes[plane] + y * line;
                if (in)
                    memcpy(inside, outside, line);
                else
                    memcpy(outside, inside, line);
            }
        }
    }

    Yuv yuv() const
    {
        return { width, height, (int)planes[1], (int)planes[2], half, wide, high, wide * high, color[0], color[1], color[2], color[3], color[4],
            color[5], color[6], color[7], color[8], color[9], color[10], color[11], color[12], color[13] };
    }

    bool make(Slot& slot, Frame& frame, const Picture& picture, uint8_t* host)
    {
        pack(picture, host, true);
        Yuv args = yuv();
        const uint8_t* source = slot.device;
        void* in[] = { &source, &frame.data, &args };
        const Dim3 block = { 16, 16, 1 }, grid = { (unsigned)(wide + 15) / 16, (unsigned)(high + 15) / 16, 1 };
        if (cuda_memcpy(slot.device, host, bytes, TO_DEVICE, slot.stream) || cuda_launch(kernels.yuv_in, grid, block, in, 0, slot.stream))
            return false;
        if (slot.frame) {
            if (!slot.frame->setTensorAddress("rgb", frame.data)
                || !slot.frame->setTensorAddress("feat", (char*)frame.data + (size_t)3 * wide * high * 2) || !slot.frame->enqueueV3(slot.stream))
                return false;
        }
        return !cuda_event_record(frame.done, slot.stream);
    }

    bool run(Slot& slot, const Picture& first, const Picture& second, float time, const Picture& output)
    {
        const auto align = [](size_t size) { return (size + 255) / 256 * 256; };
        const Picture* pictures[2] = { &first, &second };
        std::shared_ptr<Frame> frames[2];
        bool fresh[2] = { };
        struct Clean {
            Slot& slot;
            bool done = false;
            ~Clean()
            {
                if (!done)
                    cuda_stream_wait(slot.stream);
            }
        } clean { slot };
        struct Open {
            Rife& rife;
            std::shared_ptr<Frame> frame;
            ~Open()
            {
                if (frame)
                    rife.publish(frame, BROKEN);
            }
        } open[2] { { *this }, { *this } };
        for (int index = 0; index < 2; index++) {
            if (!(frames[index] = acquire(pictures[index]->id, fresh[index])))
                return false;
            if (fresh[index])
                open[index].frame = frames[index];
        }
        bool made = true;
        for (int index = 0; index < 2; index++) {
            if (fresh[index]) {
                made = made && make(slot, *frames[index], *pictures[index], slot.host + index * align(bytes));
                publish(frames[index], made ? READY : BROKEN);
                open[index].frame = nullptr;
            }
        }
        if (!made)
            return false;
        for (int index = 0; index < 2; index++) {
            if (fresh[index])
                continue;
            std::unique_lock<std::mutex> guard(cache);
            settled.wait(guard, [&] { return frames[index]->state != MAKING; });
            if (frames[index]->state == BROKEN || cuda_stream_wait_event(slot.stream, frames[index]->done, 0))
                return false;
        }
        const size_t result = align(bytes), moment = result + align(tensor);
        float* stored = (float*)(slot.host + 3 * align(bytes));
        *stored = time;
        if (cuda_memcpy(slot.device + moment, stored, sizeof(float), TO_DEVICE, slot.stream)
            || !slot.pair->setTensorAddress("frame0", frames[0]->data) || !slot.pair->setTensorAddress("frame1", frames[1]->data)
            || !slot.pair->setTensorAddress("time", slot.device + moment) || !slot.pair->setTensorAddress("out0", slot.device + result)
            || !slot.pair->enqueueV3(slot.stream))
            return false;
        Yuv args = yuv();
        const void* rgb = slot.device + result;
        void* sink = slot.device;
        void* out[] = { &rgb, &sink, &args };
        const unsigned count = (unsigned)(width * height + (width >> half) * (height >> half));
        uint8_t* host = slot.host + 2 * align(bytes);
        if (cuda_launch(kernels.yuv_out, { (count + 255) / 256, 1, 1 }, { 256, 1, 1 }, out, 0, slot.stream)
            || cuda_memcpy(host, slot.device, bytes, TO_HOST, slot.stream) || cuda_event_record(slot.done, slot.stream) || cuda_event_wait(slot.done))
            return false;
        clean.done = true;
        pack(output, host, false);
        return true;
    }

    int process(const Picture& first, const Picture& second, float time, const Picture& output)
    {
        struct Lease {
            Rife& rife;
            Slot* slot;
            ~Lease()
            {
                rife.give(slot);
            }
        } lease { *this, take() };
        return run(*lease.slot, first, second, time, output) ? 0 : 1;
    }
};

}

// an exception must not leave the library, the app gets the error
template <class F>
auto guard(F body, decltype(body()) error) -> decltype(body())
{
    try {
        return body();
    } catch (const std::exception& e) {
        fail(e.what());
    } catch (...) {
        fail("unknown error");
    }
    return error;
}

EXPORT void* rife_create(const char* models)
{
    return guard(
        [&]() -> void* {
            forget();
            if (!load_libraries())
                return nullptr;
            Rife* rife = new Rife;
            rife->folder = std::filesystem::u8path(models);
            return rife;
        },
        nullptr);
}

EXPORT int rife_prepare(void* rife, int width, int height, int bt601, int full, int yuv444, void (*report)(void*, float), void* user)
{
    Rife* self = (Rife*)rife;
    return guard(
        [&] {
            self->monitor.report = report;
            self->monitor.user = user;
            const bool ready = self->prepare(width, height, bt601, full, yuv444);
            self->monitor.report = nullptr;
            return ready ? 0 : 1;
        },
        1);
}

// thread safe
EXPORT int rife_process(void* rife, const Picture* first, const Picture* second, float time, const Picture* output)
{
    return guard([&] { return ((Rife*)rife)->process(*first, *second, time, *output); }, 1);
}

EXPORT void rife_destroy(void* rife)
{
    delete (Rife*)rife;
}

EXPORT const char* rife_error()
{
    return failure_text();
}
