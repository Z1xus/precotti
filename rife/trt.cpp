#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>

#include "tensorrt.h"

namespace {

struct Picture {
    uint8_t* data[3];
    ptrdiff_t stride[3];
    int id;
};

const char* const INPUTS[6] = { "first_y", "first_u", "first_v", "second_y", "second_u", "second_v" };
const char* const OUTPUTS[3] = { "y", "u", "v" };
enum { TIME, LUMA, LUMA_SCALE, CHROMA_SCALE, R_V, G_U, G_V, B_U, K_R, K_G, K_B, U_B, V_R, LUMA_RANGE, CHROMA_RANGE, VALUES };

struct Wrap {
    INetworkDefinition* network = nullptr;
    std::deque<float> floats;
    std::deque<int64_t> ints;

    ITensor* constant(float value)
    {
        floats.push_back(value);
        return network->addConstant(Dims4(1, 1, 1, 1), Weights { DataType::kFLOAT, &floats.back(), 1 })->getOutput(0);
    }

    ITensor* shape(const std::vector<int64_t>& values)
    {
        const size_t at = ints.size();
        ints.insert(ints.end(), values.begin(), values.end());
        Dims dims;
        dims.nbDims = 1;
        dims.d[0] = (int64_t)values.size();
        return network->addConstant(dims, Weights { DataType::kINT64, &ints[at], (int64_t)values.size() })->getOutput(0);
    }

    ITensor* math(ITensor* a, ITensor* b, ElementWiseOperation op)
    {
        return network->addElementWise(*a, *b, op)->getOutput(0);
    }

    ITensor* sum(ITensor* a, ITensor* b) { return math(a, b, ElementWiseOperation::kSUM); }
    ITensor* product(ITensor* a, ITensor* b) { return math(a, b, ElementWiseOperation::kPROD); }

    ITensor* clip(ITensor* tensor, float low, float high)
    {
        IActivationLayer* layer = network->addActivation(*tensor, ActivationType::kCLIP);
        layer->setAlpha(low);
        layer->setBeta(high);
        return layer->getOutput(0);
    }

    ITensor* cut(ITensor* tensor, Dims4 start, ITensor* size, SampleMode mode = SampleMode::kSTRICT_BOUNDS)
    {
        ISliceLayer* layer = network->addSlice(*tensor, start, Dims4(1, 1, 1, 1), Dims4(1, 1, 1, 1));
        layer->setInput(2, *size);
        layer->setMode(mode);
        return layer->getOutput(0);
    }

    ITensor* value(ITensor* values, int index)
    {
        return network->addSlice(*values, Dims4(0, index, 0, 0), Dims4(1, 1, 1, 1), Dims4(1, 1, 1, 1))->getOutput(0);
    }

    ITensor* bytes(ITensor* tensor, const char* name)
    {
        ITensor* rounded = clip(sum(tensor, constant(0.5f)), 0, 255);
        ITensor* output = network->addCast(*rounded, DataType::kUINT8)->getOutput(0);
        output->setName(name);
        network->markOutput(*output);
        return output;
    }

    // resizes to the size of another tensor, the color planes of 4:2:0 and 4:4:4 take the same engine
    ITensor* resize(ITensor* tensor, ITensor* like)
    {
        IResizeLayer* layer = network->addResize(*tensor);
        layer->setInput(1, *network->addShape(*like)->getOutput(0));
        layer->setResizeMode(InterpolationMode::kLINEAR);
        layer->setCoordinateTransformation(ResizeCoordinateTransformation::kHALF_PIXEL);
        return layer->getOutput(0);
    }

    // half is 1 when the color planes are half the size of the picture, as in 4:2:0
    bool operator()(INetworkDefinition& net, IOptimizationProfile& profile, int width, int height, int half, int side)
    {
        network = &net;
        // removeTensor on an input fails in tensorrt 10.9, so the input is reused
        ITensor* values = net.getInput(0);
        std::vector<std::pair<ILayer*, int>> readers;
        for (int index = 0; index < net.getNbLayers(); index++) {
            ILayer* layer = net.getLayer(index);
            for (int input = 0; input < layer->getNbInputs(); input++) {
                if (layer->getInput(input) == values)
                    readers.emplace_back(layer, input);
            }
        }
        values->setName("settings");
        values->setDimensions(Dims4(1, VALUES, 1, 1));
        ITensor* result = net.getOutput(0);
        auto at = [&](int index) { return value(values, index); };

        ITensor* rgb[2];
        ITensor* color = nullptr;
        for (int frame = 0; frame < 2; frame++) {
            ITensor* planes[3];
            for (int plane = 0; plane < 3; plane++) {
                const int shift = plane > 0 ? half : 0;
                ITensor* input = net.addInput(INPUTS[frame * 3 + plane], DataType::kUINT8, Dims4(1, 1, -1, -1));
                profile.setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4(1, 1, 1, 1));
                profile.setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4(1, 1, height >> shift, width >> shift));
                profile.setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4(1, 1, side, side));
                planes[plane] = net.addCast(*input, DataType::kFLOAT)->getOutput(0);
                if (plane > 0) {
                    color = color ? color : input;
                    planes[plane] = product(sum(resize(planes[plane], planes[0]), constant(-128)), at(CHROMA_SCALE));
                }
            }
            ITensor* y = product(sum(planes[0], product(at(LUMA), constant(-1))), at(LUMA_SCALE));
            ITensor* r = sum(y, product(at(R_V), planes[2]));
            ITensor* g = math(math(y, product(at(G_U), planes[1]), ElementWiseOperation::kSUB), product(at(G_V), planes[2]), ElementWiseOperation::kSUB);
            ITensor* b = sum(y, product(at(B_U), planes[1]));
            ITensor* channels[3] = { clip(r, 0, 1), clip(g, 0, 1), clip(b, 0, 1) };
            rgb[frame] = net.addConcatenation(channels, 3)->getOutput(0);
        }

        // the model takes sizes in steps of 64
        ITensor* size = net.addShape(*rgb[0])->getOutput(0);
        ITensor* padded = product(math(sum(size, shape({ 0, 0, 63, 63 })), shape({ 1, 1, 64, 64 }), ElementWiseOperation::kFLOOR_DIV), shape({ 1, 1, 64, 64 }));
        ITensor* plane = sum(product(padded, shape({ 1, 0, 1, 1 })), shape({ 0, 1, 0, 0 }));
        ITensor* inputs[3] = { cut(rgb[0], Dims4(0, 0, 0, 0), padded, SampleMode::kCLAMP), cut(rgb[1], Dims4(0, 0, 0, 0), padded, SampleMode::kCLAMP),
            cut(values, Dims4(0, TIME, 0, 0), plane, SampleMode::kCLAMP) };
        ITensor* joined = net.addConcatenation(inputs, 3)->getOutput(0);
        for (auto [layer, input] : readers)
            layer->setInput(input, *joined);

        net.unmarkOutput(*result);
        ITensor* luma = sum(product(size, shape({ 1, 0, 1, 1 })), shape({ 0, 1, 0, 0 }));
        ITensor* picture = clip(cut(result, Dims4(0, 0, 0, 0), size), 0, 1);
        ITensor* r = cut(picture, Dims4(0, 0, 0, 0), luma);
        ITensor* g = cut(picture, Dims4(0, 1, 0, 0), luma);
        ITensor* b = cut(picture, Dims4(0, 2, 0, 0), luma);
        ITensor* y = sum(sum(product(at(K_R), r), product(at(K_G), g)), product(at(K_B), b));
        bytes(sum(product(y, at(LUMA_RANGE)), at(LUMA)), OUTPUTS[0]);
        ITensor* chroma[2] = { product(math(b, y, ElementWiseOperation::kSUB), at(U_B)), product(math(r, y, ElementWiseOperation::kSUB), at(V_R)) };
        // at half the size the bilinear samples at half pixels are the mean of each 2x2 block
        for (int plane = 0; plane < 2; plane++)
            bytes(sum(product(resize(chroma[plane], color), at(CHROMA_RANGE)), constant(128)), OUTPUTS[plane + 1]);
        return true;
    }
};

struct Slot {
    IExecutionContext* context = nullptr;
    void* memory = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;
    int held[2] = { -1, -1 };

    ~Slot()
    {
        delete context;
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
    ICudaEngine* engine = nullptr;
    int width = 0, height = 0;
    size_t planes[3] = {}, frame = 0;
    // 1 when the color planes are half the size of the picture
    int half = 1;
    float values[VALUES] = {};
    std::vector<Slot*> slots, idle;
    std::mutex lock;
    std::condition_variable ready;

    ~Rife()
    {
        for (Slot* slot : slots)
            delete slot;
        delete engine;
        delete runtime;
    }

    Slot* open()
    {
        Slot* slot = new Slot;
        slots.push_back(slot);
        const size_t bytes = 3 * frame + sizeof(values);
        if (!(slot->context = engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED)))
            return nullptr;
        for (int input = 0; input < 6; input++) {
            const int shift = input % 3 > 0 ? half : 0;
            if (!slot->context->setInputShape(INPUTS[input], Dims4(1, 1, height >> shift, width >> shift)))
                return nullptr;
        }
        const size_t memory = slot->context->updateDeviceMemorySizeForShapes();
        if (cuda_malloc(&slot->memory, memory) || cuda_malloc((void**)&slot->device, bytes) || cuda_malloc_host((void**)&slot->host, bytes))
            return nullptr;
        slot->context->setDeviceMemoryV2(slot->memory, (int64_t)memory);
        for (int plane = 0; plane < 3; plane++)
            slot->context->setTensorAddress(OUTPUTS[plane], slot->device + 2 * frame + planes[plane]);
        slot->context->setTensorAddress("settings", slot->device + 3 * frame);
        if (cuda_stream_create(&slot->stream, NON_BLOCKING) || cuda_event_create(&slot->done, BLOCKING_SYNC | DISABLE_TIMING))
            return nullptr;
        return slot;
    }

    // a rife that ran before keeps its engine, and its slots when the size is the same
    bool prepare(int w, int h, bool bt601, bool full, bool yuv444)
    {
        forget();
        const float kr = bt601 ? 0.299f : 0.2126f, kb = bt601 ? 0.114f : 0.0722f, kg = 1 - kr - kb;
        const float luma = full ? 255 : 219, chroma = full ? 255 : 224;
        const float rv = 2 * (1 - kr), bu = 2 * (1 - kb);
        const float numbers[VALUES] = { 0, full ? 0.f : 16.f, 1 / luma, 1 / chroma, rv, bu * kb / kg, rv * kr / kg, bu, kr, kg, kb, 1 / bu, 1 / rv, luma, chroma };
        std::copy(numbers, numbers + VALUES, values);
        if (!slots.empty() && w == width && h == height && half == !yuv444) {
            // the frames in the slots are from the clip before
            for (Slot* slot : slots)
                slot->held[0] = slot->held[1] = -1;
            return true;
        }
        for (Slot* slot : slots)
            delete slot;
        slots.clear();
        idle.clear();
        width = w;
        height = h;
        half = !yuv444;
        // tensorrt wants tensors on 256 bytes
        const auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
        const size_t color = (size_t)(w >> half) * (h >> half);
        planes[1] = align((size_t)w * h);
        planes[2] = planes[1] + align(color);
        frame = planes[2] + align(color);

        if (!runtime && !(runtime = (IRuntime*)create_runtime(&logger, NV_TENSORRT_VERSION)))
            return false;
        Wrap wrap;
        // any size up to a square of this side, a larger video builds again; at 4096 the larger models ran out of gpu memory
        const int side = (std::max({ w, h, 1920 }) + 63) / 64 * 64;
        const Edit edit = [&](INetworkDefinition& network, IOptimizationProfile& profile) { return wrap(network, profile, w, h, half, side); };
        const auto fits = [&](ICudaEngine& engine) {
            if (engine.getTensorIOMode("settings") != TensorIOMode::kINPUT)
                return false;
            const Dims luma = engine.getProfileShape(INPUTS[0], 0, OptProfileSelector::kMAX);
            const Dims color = engine.getProfileShape(INPUTS[1], 0, OptProfileSelector::kMAX);
            return luma.d[2] >= h && luma.d[3] >= w && color.d[2] >= h && color.d[3] >= w;
        };
        if (!engine || !fits(*engine)) {
            delete engine;
            engine = load_engine(runtime, folder / "rife.onnx", folder / "rife.engine", edit, fits, monitor);
        }
        if (!engine)
            return false;
        // older plugins made one engine per size
        std::error_code error;
        for (const auto& file : std::filesystem::directory_iterator(folder, error)) {
            if (file.path().extension() == ".engine" && file.path().filename() != "rife.engine")
                std::filesystem::remove(file.path(), error);
        }
        // more slots gain little
        for (int count = 0; count < 2; count++) {
            if (Slot* slot = open())
                idle.push_back(slot);
            else if (idle.empty()) {
                fail("not enough gpu memory");
                return false;
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

    bool run(Slot* slot, const Picture& first, const Picture& second, float time, const Picture& output)
    {
        const Picture* pictures[2] = { &first, &second };
        int place[2] = { -1, -1 };
        for (int picture = 0; picture < 2; picture++) {
            for (int held = 0; held < 2; held++) {
                if (slot->held[held] == pictures[picture]->id)
                    place[picture] = held;
            }
        }
        for (int picture = 0; picture < 2; picture++) {
            if (place[picture] >= 0)
                continue;
            const int held = place[1 - picture] == 0;
            pack(*pictures[picture], slot->host + held * frame, true);
            if (cuda_memcpy(slot->device + held * frame, slot->host + held * frame, frame, TO_DEVICE, slot->stream))
                return false;
            slot->held[held] = pictures[picture]->id;
            place[picture] = held;
        }
        for (int input = 0; input < 6; input++)
            slot->context->setTensorAddress(INPUTS[input], slot->device + place[input / 3] * frame + planes[input % 3]);
        float* settings = (float*)(slot->host + 3 * frame);
        std::copy(values, values + VALUES, settings);
        settings[TIME] = time;
        uint8_t* result = slot->host + 2 * frame;
        const bool done = !cuda_memcpy(slot->device + 3 * frame, settings, sizeof(values), TO_DEVICE, slot->stream)
            && slot->context->enqueueV3(slot->stream)
            && !cuda_memcpy(result, slot->device + 2 * frame, frame, TO_HOST, slot->stream)
            && !cuda_event_record(slot->done, slot->stream)
            && !cuda_event_wait(slot->done);
        if (!done) {
            slot->held[0] = slot->held[1] = -1;
            return false;
        }
        pack(output, result, false);
        return true;
    }

    int process(const Picture& first, const Picture& second, float time, const Picture& output)
    {
        Slot* slot = take();
        const bool done = run(slot, first, second, time, output);
        give(slot);
        return done ? 0 : 1;
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
