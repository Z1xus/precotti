#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// two frames in work keep the gpu busy, more only take its memory
struct Turn {
    std::mutex& lock;
    std::condition_variable& room;
    int& working;

    Turn(std::mutex& lock, std::condition_variable& room, int& working)
        : lock(lock)
        , room(room)
        , working(working)
    {
        std::unique_lock<std::mutex> hold(lock);
        room.wait(hold, [&] { return working < 2; });
        working++;
    }

    ~Turn()
    {
        {
            const std::lock_guard<std::mutex> hold(lock);
            working--;
        }
        room.notify_one();
    }
};

struct Held {
    int id = 0;
    ncnn::VkMat mat;
    bool ready = false, failed = false;
};

struct Rife {
    ncnn::VulkanDevice* gpu = nullptr;
    ncnn::Net flownet;
    Graph graph;
    std::mutex busy;
    std::condition_variable room;
    int working = 0;
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

    ~Rife()
    {
        held.clear();
        delete kept;
        delete in;
        delete out;
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
        gpu = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
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
        {
            const std::lock_guard<std::mutex> hold(cache);
            held.clear();
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
        const int wide = (width + padding - 1) / padding * padding;
        const int high = (height + padding - 1) / padding * padding;
        const Picture* pictures[2] = { &first, &second };

        const Turn turn(busy, room, working);
        ncnn::VkAllocator* blobs = gpu->acquire_blob_allocator();
        ncnn::VkAllocator* staging = gpu->acquire_staging_allocator();
        ncnn::Option opt = flownet.opt;
        opt.blob_vkallocator = blobs;
        opt.workspace_vkallocator = blobs;
        opt.staging_vkallocator = staging;
        const size_t element = opt.use_fp16_storage ? 2u : 4u;
        ncnn::VkCompute cmd(gpu);

        std::shared_ptr<Held> frames[2];
        bool fresh[2] = {};
        ncnn::VkMat mats[2];
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
                    fresh[index] = true;
                }
                held.push_back(frames[index]);
                mats[index] = frames[index]->mat;
            }
            if (held.size() > 4)
                held.erase(held.begin());
        }
        // the allocator holds only these frames, so free them under the lock
        const auto finish = [&](bool done) {
            const std::lock_guard<std::mutex> hold(cache);
            for (int index = 0; index < 2; index++) {
                if (fresh[index]) {
                    frames[index]->ready = done;
                    frames[index]->failed = !done;
                    if (!done)
                        held.erase(std::remove(held.begin(), held.end(), frames[index]), held.end());
                }
                mats[index].release();
                frames[index].reset();
            }
            settled.notify_all();
        };
        const auto fail = [&] {
            finish(false);
            gpu->reclaim_blob_allocator(blobs);
            gpu->reclaim_staging_allocator(staging);
            return 1;
        };

        std::vector<uint8_t> bytes[3] = { {}, {}, take() };
        for (int index = 0; index < 2; index++) {
            if (!fresh[index])
                continue;
            bytes[index] = take();
            pack(*pictures[index], bytes[index].data(), true);
            ncnn::VkMat uploaded;
            cmd.record_clone(ncnn::Mat(frame / 4, (void*)bytes[index].data(), (size_t)4u), uploaded, opt);
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
        ncnn::VkMat result;
        if (graph.run(mats, time, result, cmd, opt))
            return fail();

        ncnn::VkMat packed;
        packed.create(frame / 4, (size_t)4u, 1, blobs);
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
        // the nvidia driver lost the gpu with a scheduler error when frames ran on several queues at once
        static std::mutex queue;
        // a frame that another call converts is ready once that call has run
        std::unique_lock<std::mutex> hold(cache);
        settled.wait(hold, [&] {
            for (int index = 0; index < 2; index++) {
                if (!fresh[index] && !(index == 1 && frames[1] == frames[0]) && !frames[index]->ready && !frames[index]->failed)
                    return false;
            }
            return true;
        });
        const bool lost = (!fresh[0] && frames[0]->failed) || (!fresh[1] && frames[1]->failed);
        hold.unlock();
        if (lost)
            return fail();
        std::unique_lock<std::mutex> submit(queue);
        const int failed = cmd.submit_and_wait();
        submit.unlock();
        finish(!failed);
        gpu->reclaim_blob_allocator(blobs);
        gpu->reclaim_staging_allocator(staging);
        if (failed)
            return 1;
        pack(output, bytes[2].data(), false);
        const std::lock_guard<std::mutex> guard(cache);
        for (std::vector<uint8_t>& done : bytes) {
            if (!done.empty())
                spare.push_back(std::move(done));
        }
        return 0;
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
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() == 0)
        return nullptr;
    // exit handlers run in reverse, so this one runs before the one of ncnn
    static const int registered = atexit(release);
    (void)registered;
    Rife* rife = new Rife;
    if (!rife->load(models)) {
        delete rife;
        return nullptr;
    }
    const std::lock_guard<std::mutex> guard(lock);
    alive.insert(rife);
    return rife;
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

// ncnn tells nothing more than the result of a call
EXPORT const char* rife_error()
{
    return "";
}
