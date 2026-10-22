#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "command.h"
#include "gpu.h"
#include "graph.h"
#include "net.h"

#include "yuv_in.comp.hex.h"
#include "yuv_out.comp.hex.h"

#if _WIN32
#include <windows.h>
#define EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

// the graph runs the warps, ncnn only needs to know the layer
struct Warp : ncnn::Layer {
    Warp()
    {
        one_blob_only = false;
        support_vulkan = true;
    }
};

ncnn::Layer* make_warp(void*)
{
    return new Warp;
}

struct Picture {
    uint8_t* data[3];
    ptrdiff_t stride[3];
    int id;
};

FILE* open(const char* path)
{
#if _WIN32
    std::wstring wide(MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0), 0);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wide.data(), (int)wide.size());
    return _wfopen(wide.c_str(), L"rb");
#else
    return fopen(path, "rb");
#endif
}

template <class Base>
struct Watched : Base {
    using Base::Base;
    using Base::fastFree;
    using Base::fastMalloc;
    bool lost = false;
    size_t live = 0;
    std::atomic<size_t> peak { 0 };

    ncnn::VkBufferMemory* fastMalloc(size_t size) override
    {
        ncnn::VkBufferMemory* ptr = Base::fastMalloc(size);
        if (!ptr)
            lost = true;
        else {
            live += ptr->capacity;
            peak = std::max(peak.load(), live);
        }
        return ptr;
    }

    void fastFree(ncnn::VkBufferMemory* ptr) override
    {
        live -= ptr->capacity;
        Base::fastFree(ptr);
    }
};

// what one frame in work owns
struct Slot {
    Watched<ncnn::VkBlobAllocator> blobs;
    Watched<ncnn::VkStagingAllocator> staging;

    explicit Slot(const ncnn::VulkanDevice* gpu)
        : blobs(gpu)
        , staging(gpu)
    {
    }
};

enum { DONE, FAILED, STARVED };

// windows resets a gpu after 2 s on one submit, 4 GMAC take 0.5 s at 8 GMAC/s
constexpr int64_t SUBMIT_MACS = 4LL << 30;

thread_local std::string failure;
// the nvidia driver loses the gpu when frames run on several queues, also from two rifes
std::mutex queue;

// nvenc and the motion search use cuda device 0
bool cuda_uuid(uint8_t (&uuid)[16])
{
#if _WIN32
    HMODULE library = LoadLibraryA("nvcuda.dll");
    const auto find = [&](const char* name) { return library ? (void*)GetProcAddress(library, name) : nullptr; };
#else
    void* library = dlopen("libcuda.so.1", RTLD_LAZY);
    const auto find = [&](const char* name) { return library ? dlsym(library, name) : nullptr; };
#endif
    const auto init = (int (*)(unsigned))find("cuInit");
    const auto get = (int (*)(int*, int))find("cuDeviceGet");
    const auto ask = (int (*)(uint8_t*, int))find("cuDeviceGetUuid");
    int device = 0;
    return init && get && ask && init(0) == 0 && get(&device, 0) == 0 && ask(uuid, device) == 0;
}

int pick()
{
    uint8_t wanted[16];
    const bool known = cuda_uuid(wanted);
    int best = -1;
    uint64_t score = 0;
    for (int index = 0; index < ncnn::get_gpu_count(); index++) {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(index);
        VkPhysicalDeviceIDProperties id = {};
        id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2KHR properties = {};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        properties.pNext = &id;
        ncnn::vkGetPhysicalDeviceProperties2KHR(info.physicalDevice(), &properties);
        if (known && memcmp(id.deviceUUID, wanted, sizeof(wanted)) == 0)
            return index;
        const VkPhysicalDeviceMemoryProperties& memory = info.physicalDeviceMemoryProperties();
        uint64_t largest = 0;
        for (uint32_t heap = 0; heap < memory.memoryHeapCount; heap++) {
            if (memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                largest = std::max<uint64_t>(largest, memory.memoryHeaps[heap].size);
        }
        const uint64_t kind = info.type() == 0 ? 2 : info.type() == 1 ? 1 : 0;
        if (best < 0 || (kind << 60 | largest) > score) {
            best = index;
            score = kind << 60 | largest;
        }
    }
    return best;
}

struct Held {
    int id = 0;
    ncnn::VkMat mat;
    bool ready = false, failed = false;
    ncnn::VkMat feature[2];
    bool featured[2] = {};
};

struct Rife {
    ncnn::VulkanDevice* gpu = nullptr;
    ncnn::Net flownet;
    Graph graph;
    // two frames keep the gpu busy, one runs alone after the gpu ran out of memory
    std::mutex busy;
    std::condition_variable room;
    int working = 0;
    bool solo = false;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<Slot*> idle;
    uint32_t heap = 0;
    std::atomic<bool> dead { false };
    ncnn::Pipeline* in = nullptr;
    ncnn::Pipeline* out = nullptr;
    int padding = 32;
    int width = 0, height = 0;
    int planes[3] = {}, frame = 0;
    // 1 when the color planes are half the size of the picture
    int half = 1;
    float color[15] = {};
    // guards the frames and their allocator
    std::mutex cache;
    std::condition_variable settled;
    std::vector<std::shared_ptr<Held>> held;
    std::vector<std::vector<uint8_t>> spare;
    ncnn::VkAllocator* kept = nullptr;

    struct Turn {
        Rife& rife;
        const bool alone;
        Slot* const slot;

        Turn(Rife& rife, bool alone)
            : rife(rife)
            , alone(alone)
            , slot(rife.enter(alone))
        {
        }

        ~Turn() { rife.leave(slot, alone); }
    };

    ~Rife()
    {
        held.clear();
        delete kept;
        delete in;
        delete out;
    }

    size_t available() const
    {
        if (!gpu->info.support_VK_EXT_memory_budget())
            return std::max<int64_t>(0, (int64_t)gpu->get_heap_budget() * 1024 * 1024 - (int64_t)reserved());
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2KHR properties = {};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
        properties.pNext = &budget;
        ncnn::vkGetPhysicalDeviceMemoryProperties2KHR(gpu->info.physicalDevice(), &properties);
        return budget.heapBudget[heap] > budget.heapUsage[heap] ? budget.heapBudget[heap] - budget.heapUsage[heap] : 0;
    }

    size_t reserved() const
    {
        size_t bytes = 0;
        for (const auto& slot : slots)
            bytes += slot->blobs.peak;
        return bytes;
    }

    // a second frame needs the most a slot held, less what this one holds
    Slot* enter(bool alone)
    {
        std::unique_lock<std::mutex> hold(busy);
        room.wait(hold, [&] {
            if (solo || (alone ? working > 0 : working >= 2))
                return false;
            if (alone || working == 0)
                return true;
            size_t most = 0, own = 0;
            for (const auto& slot : slots)
                most = std::max(most, slot->blobs.peak.load());
            for (const Slot* slot : idle)
                own = std::max(own, slot->blobs.peak.load());
            return available() >= most - own;
        });
        if (idle.empty()) {
            slots.push_back(std::make_unique<Slot>(gpu));
            idle.push_back(slots.back().get());
        }
        const auto best = std::max_element(idle.begin(), idle.end(), [](const Slot* a, const Slot* b) { return a->blobs.peak < b->blobs.peak; });
        Slot* slot = *best;
        idle.erase(best);
        slot->blobs.lost = slot->staging.lost = false;
        working++;
        solo = alone;
        return slot;
    }

    void leave(Slot* slot, bool alone)
    {
        {
            const std::lock_guard<std::mutex> hold(busy);
            working--;
            if (alone)
                solo = false;
            idle.push_back(slot);
        }
        room.notify_all();
    }

    void trim()
    {
        const std::lock_guard<std::mutex> turns(busy);
        {
            const std::lock_guard<std::mutex> hold(cache);
            held.clear();
        }
        kept->clear();
        for (const auto& slot : slots) {
            slot->blobs.clear();
            slot->staging.clear();
        }
    }

    std::vector<uint8_t> take()
    {
        const std::lock_guard<std::mutex> hold(cache);
        if (spare.empty())
            return std::vector<uint8_t>(frame);
        std::vector<uint8_t> bytes = std::move(spare.back());
        spare.pop_back();
        return bytes;
    }

    ncnn::Pipeline* pipeline(const char* data, int size, int x, int y, int z)
    {
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(data, size, flownet.opt, spirv))
            return nullptr;
        ncnn::Pipeline* pipeline = new ncnn::Pipeline(gpu);
        pipeline->set_optimal_local_size_xyz(x, y, z);
        pipeline->create(spirv.data(), spirv.size() * 4, {});
        return pipeline;
    }

    bool load(const std::string& folder)
    {
        gpu = ncnn::get_gpu_device(pick());
        ncnn::Option& opt = flownet.opt;
        opt.use_vulkan_compute = true;
        // the shaders of the graph read halves only from fp16 storage
        opt.use_fp16_storage = opt.use_fp16_packed = gpu->info.support_fp16_storage();
        // amd and intel gpus do fp16 math at up to twice the rate of fp32
        opt.use_fp16_arithmetic = true;
        opt.use_int8_storage = false;
        // the gemm kernels on cooperative matrices beat winograd
        opt.use_winograd_convolution = !gpu->info.support_cooperative_matrix();
        flownet.set_vulkan_device(gpu);
        const VkPhysicalDeviceMemoryProperties& memory = gpu->info.physicalDeviceMemoryProperties();
        VkDeviceSize largest = 0;
        for (uint32_t index = 0; index < memory.memoryHeapCount; index++) {
            if ((memory.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && memory.memoryHeaps[index].size > largest) {
                largest = memory.memoryHeaps[index].size;
                heap = index;
            }
        }
        flownet.register_custom_layer("rife.Warp", make_warp);
        Graph::keep(flownet);
        FILE* param = open((folder + "/flownet.param").c_str());
        FILE* model = open((folder + "/flownet.bin").c_str());
        const bool loaded = param && model && flownet.load_param(param) == 0 && flownet.load_model(model) == 0;
        if (param)
            fclose(param);
        if (model)
            fclose(model);
        // the newer models take the picture in larger steps of size
        padding = folder.find("rife-v4.25-lite") != std::string::npos ? 128
            : folder.find("rife-v4.25") != std::string::npos || folder.find("rife-v4.26") != std::string::npos ? 64
            : 32;
        in = pipeline(yuv_in_comp_data, sizeof(yuv_in_comp_data), 8, 8, 1);
        out = pipeline(yuv_out_comp_data, sizeof(yuv_out_comp_data), 64, 1, 1);
        kept = new ncnn::VkBlobAllocator(gpu);
        return loaded && in && out && graph.load(flownet);
    }

    void prepare(int w, int h, bool bt601, bool full, bool yuv444)
    {
        trim();
        {
            const std::lock_guard<std::mutex> hold(cache);
            spare.clear();
        }
        width = w;
        height = h;
        half = !yuv444;
        const auto word = [](int bytes) { return (bytes + 3) / 4 * 4; };
        const int area = (w >> half) * (h >> half);
        planes[1] = word(w * h);
        planes[2] = planes[1] + word(area);
        frame = planes[2] + word(area);
        const float kr = bt601 ? 0.299f : 0.2126f, kb = bt601 ? 0.114f : 0.0722f, kg = 1 - kr - kb;
        const float luma = full ? 255 : 219, chroma = full ? 255 : 224;
        const float rv = 2 * (1 - kr), bu = 2 * (1 - kb);
        const float values[15] = { full ? 0.f : 16.f, 1 / luma, 1 / chroma, rv, bu * kb / kg, rv * kr / kg, bu, luma, chroma, kr, kg, kb, 1 / bu, 1 / rv };
        std::copy(values, values + 15, color);
    }

    void pack(const Picture& picture, uint8_t* block, bool into)
    {
        for (int plane = 0; plane < 3; plane++) {
            const int shift = plane > 0 ? half : 0;
            const size_t line = (size_t)width >> shift;
            for (int y = 0; y < height >> shift; y++) {
                uint8_t* outside = picture.data[plane] + y * picture.stride[plane];
                uint8_t* inside = block + planes[plane] + y * line;
                if (into)
                    memcpy(inside, outside, line);
                else
                    memcpy(outside, inside, line);
            }
        }
    }

    int process(const Picture& first, const Picture& second, float time, const Picture& output)
    {
        failure.clear();
        if (dead) {
            failure = "the gpu stopped responding";
            return 1;
        }
        int code = run(first, second, time, output, false);
        if (code == STARVED)
            code = run(first, second, time, output, true);
        if (dead)
            failure = "the gpu stopped responding";
        else if (code == STARVED)
            failure = "not enough gpu memory";
        else if (code == FAILED)
            failure = "rife could not render the frame";
        return code != DONE;
    }

    int run(const Picture& first, const Picture& second, float time, const Picture& output, bool alone)
    {
        const int wide = (width + padding - 1) / padding * padding;
        const int high = (height + padding - 1) / padding * padding;
        const Picture* pictures[2] = { &first, &second };

        // the turn ends last, after this frame left the allocators
        const Turn turn(*this, alone);
        if (alone)
            trim();
        Slot& slot = *turn.slot;
        const auto starved = [&] { return slot.blobs.lost || slot.staging.lost; };
        ncnn::Option opt = flownet.opt;
        opt.blob_vkallocator = &slot.blobs;
        opt.workspace_vkallocator = &slot.blobs;
        opt.staging_vkallocator = &slot.staging;
        const size_t element = opt.use_fp16_storage ? 2u : 4u;
        ncnn::VkCompute cmd(gpu);

        std::shared_ptr<Held> frames[2];
        bool fresh[2] = {}, making[2] = {}, empty = false;
        ncnn::VkMat mats[2], features[2];
        {
            const std::lock_guard<std::mutex> hold(cache);
            for (int index = 0; index < 2; index++) {
                const auto found = std::find_if(held.begin(), held.end(), [&](const auto& frame) { return frame->id == pictures[index]->id; });
                if (found != held.end()) {
                    frames[index] = *found;
                    held.erase(found);
                } else {
                    frames[index] = std::make_shared<Held>();
                    frames[index]->id = pictures[index]->id;
                    frames[index]->mat.create(wide, high, 3, element, 1, kept);
                    empty |= frames[index]->mat.empty();
                    fresh[index] = true;
                }
                held.push_back(frames[index]);
                mats[index] = frames[index]->mat;
                if (frames[index]->feature[index].empty()) {
                    ncnn::Option kind = opt;
                    kind.blob_vkallocator = kept;
                    try {
                        frames[index]->feature[index] = graph.feature(index, wide, high, kind);
                    } catch (const Lost&) {
                    }
                    making[index] = !frames[index]->feature[index].empty();
                }
                features[index] = frames[index]->feature[index];
            }
            if (held.size() > 4)
                held.erase(held.begin());
        }
        // the allocator holds only these frames, so free them under the lock
        const auto finish = [&](bool done) {
            const std::lock_guard<std::mutex> hold(cache);
            for (int index = 0; index < 2; index++) {
                if (fresh[index])
                    frames[index]->ready = done;
                if (making[index])
                    frames[index]->featured[index] = done;
                if ((fresh[index] || making[index]) && !done) {
                    frames[index]->failed = true;
                    held.erase(std::remove(held.begin(), held.end(), frames[index]), held.end());
                }
                mats[index].release();
                features[index].release();
                frames[index].reset();
            }
            settled.notify_all();
        };
        const auto fail = [&](int code) {
            finish(false);
            return code;
        };
        if (empty)
            return fail(STARVED);

        std::vector<uint8_t> bytes[3] = { {}, {}, take() };
        for (int index = 0; index < 2; index++) {
            if (!fresh[index])
                continue;
            bytes[index] = take();
            pack(*pictures[index], bytes[index].data(), true);
            ncnn::VkMat uploaded;
            cmd.record_clone(ncnn::Mat(frame / 4, (void*)bytes[index].data(), (size_t)4u), uploaded, opt);
            if (uploaded.empty())
                return fail(STARVED);
            std::vector<ncnn::vk_constant_type> constants(15);
            const int sizes[8] = { width, height, planes[1], planes[2], half, wide, high, (int)mats[index].cstep };
            for (int i = 0; i < 8; i++)
                constants[i].i = sizes[i];
            for (int i = 0; i < 7; i++)
                constants[8 + i].f = color[i];
            ncnn::VkMat dispatcher;
            dispatcher.w = wide;
            dispatcher.h = high;
            dispatcher.c = 1;
            cmd.record_pipeline(in, { uploaded, mats[index] }, constants, dispatcher);
        }
        // ready after its convert or make call, which the first submit waits for
        bool waited = false, gone = false;
        const auto settle = [&] {
            std::unique_lock<std::mutex> hold(cache);
            settled.wait(hold, [&] {
                for (int index = 0; index < 2; index++) {
                    const Held& frame = *frames[index];
                    if (frame.failed)
                        return true;
                    if (!fresh[index] && !(index == 1 && frames[1] == frames[0]) && !frame.ready)
                        return false;
                    if (!features[index].empty() && !making[index] && !frame.featured[index])
                        return false;
                }
                return true;
            });
            return frames[0]->failed || frames[1]->failed;
        };
        const auto submit = [&](bool more) {
            if (!std::exchange(waited, true) && (gone = settle()))
                return 1;
            const std::lock_guard<std::mutex> hold(queue);
            const int failed = cmd.submit_and_wait();
            if (failed)
                dead = true;
            else if (more)
                cmd.reset();
            return failed;
        };
        const auto flush = [&] { return submit(true); };
        ncnn::VkMat result;
        const bool filled[2] = { !making[0], !making[1] };
        try {
            if (graph.run(mats, time, result, cmd, opt, features, filled, flush, SUBMIT_MACS))
                return fail(starved() || gone ? STARVED : FAILED);
        } catch (const Lost&) {
            return fail(STARVED);
        }

        ncnn::VkMat packed;
        packed.create(frame / 4, (size_t)4u, 1, &slot.blobs);
        if (packed.empty())
            return fail(STARVED);
        for (int chroma = 0; chroma < 2; chroma++) {
            std::vector<ncnn::vk_constant_type> constants(16);
            const int sizes[8] = { width, height, (int)result.cstep, result.w, chroma, planes[1], planes[2], half };
            for (int i = 0; i < 8; i++)
                constants[i].i = sizes[i];
            constants[8].f = color[0];
            for (int i = 0; i < 7; i++)
                constants[9 + i].f = color[7 + i];
            ncnn::VkMat dispatcher;
            dispatcher.w = chroma ? ((width >> half) * (height >> half) + 3) / 4 : (width * height + 3) / 4;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(out, { result, packed }, constants, dispatcher);
        }
        ncnn::Mat downloaded(frame / 4, (void*)bytes[2].data(), (size_t)4u);
        cmd.record_clone(packed, downloaded, opt);
        if (starved())
            return fail(STARVED);
        const int failed = submit(false);
        if (failed)
            return fail(gone ? STARVED : FAILED);
        finish(true);
        pack(output, bytes[2].data(), false);
        const std::lock_guard<std::mutex> guard(cache);
        for (std::vector<uint8_t>& done : bytes) {
            if (!done.empty())
                spare.push_back(std::move(done));
        }
        return DONE;
    }
};

std::mutex lock;
std::set<Rife*> alive;

// ncnn destroys the gpu at exit, and the nvidia driver corrupted its heap when pipelines were still alive then
void release()
{
    const std::lock_guard<std::mutex> guard(lock);
    for (Rife* rife : alive)
        delete rife;
    alive.clear();
}

}

EXPORT void* rife_create(const char* models)
{
    failure.clear();
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() == 0) {
        failure = "rife needs a vulkan gpu";
        return nullptr;
    }
    // exit handlers run in reverse, so this one runs before the one of ncnn
    static const int registered = atexit(release);
    (void)registered;
    Rife* rife = new Rife;
    if (!rife->load(models)) {
        delete rife;
        failure = "rife could not load its model on the gpu";
        return nullptr;
    }
    const std::lock_guard<std::mutex> guard(lock);
    alive.insert(rife);
    return rife;
}

// the app calls it before exit, so no gpu call runs at unload
EXPORT void rife_shutdown()
{
    release();
    ncnn::destroy_gpu_instance();
}

// ncnn has nothing to build
EXPORT int rife_prepare(void* rife, int width, int height, int bt601, int full, int yuv444, void (*)(void*, float), void*)
{
    ((Rife*)rife)->prepare(width, height, bt601, full, yuv444);
    return 0;
}

EXPORT int rife_process(void* rife, const Picture* first, const Picture* second, float time, const Picture* output)
{
    return ((Rife*)rife)->process(*first, *second, time, *output);
}

EXPORT void rife_destroy(void* rife)
{
    const std::lock_guard<std::mutex> guard(lock);
    alive.erase((Rife*)rife);
    delete (Rife*)rife;
}

EXPORT const char* rife_error()
{
    return failure.c_str();
}
