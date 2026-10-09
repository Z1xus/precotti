#include <cstdint>
#include <cstring>

#include "tensorrt.h"

namespace {

struct Rife {
    std::filesystem::path folder;
    Monitor monitor;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    int width = 0, height = 0;
    // the size that the model takes, a multiple of 64
    int wide = 0, high = 0;
    // two rgb frames and the time, and in older models 4 more planes that depend on the size only
    int channels = 0;
    float* input = nullptr;
    float* output = nullptr;
    // the source frames that are on the gpu now
    std::vector<uint8_t> held[2];
    float time = -1;
    // page-locked memory, which moves to and from the gpu faster
    float* planes = nullptr;

    ~Rife()
    {
        delete context;
        delete engine;
        delete runtime;
        if (input)
            cuda_free(input);
        if (output)
            cuda_free(output);
        if (planes)
            cuda_free_host(planes);
    }

    bool prepare(int w, int h)
    {
        width = w;
        height = h;
        wide = (w + 63) / 64 * 64;
        high = (h + 63) / 64 * 64;
        // an engine fits one size, one gpu model and one tensorrt version
        const std::filesystem::path name = folder / (std::to_string(wide) + "x" + std::to_string(high) + ".engine");
        runtime = (IRuntime*)create_runtime(&logger, NV_TENSORRT_VERSION);
        if (!runtime)
            return false;
        engine = load_engine(runtime, folder / "rife.onnx", name, high, wide, monitor);
        if (!engine || !(context = engine->createExecutionContext()))
            return false;

        const size_t pixels = (size_t)wide * high;
        const char* in = engine->getIOTensorName(0);
        const char* out = engine->getIOTensorName(1);
        channels = engine->getTensorShape(in).d[1];
        if (cuda_malloc((void**)&input, channels * pixels * sizeof(float)) || cuda_malloc((void**)&output, 3 * pixels * sizeof(float)))
            return false;
        if (!context->setInputShape(in, Dims4(1, channels, high, wide)) || !context->setTensorAddress(in, input) || !context->setTensorAddress(out, output))
            return false;

        if (cuda_malloc_host((void**)&planes, 4 * pixels * sizeof(float)))
            return false;
        for (int y = 0; y < high; y++) {
            for (int x = 0; x < wide; x++) {
                const size_t at = (size_t)y * wide + x;
                planes[at] = 2.f * x / (wide - 1) - 1;
                planes[pixels + at] = 2.f * y / (high - 1) - 1;
                planes[2 * pixels + at] = 2.f / (wide - 1);
                planes[3 * pixels + at] = 2.f / (high - 1);
            }
        }
        held[0].assign((size_t)w * h * 3, 0);
        held[1].assign((size_t)w * h * 3, 0);
        // the two frame slots start black, like the frames that they are compared with
        std::vector<float> black(7 * pixels, 0.f);
        return cuda_memcpy(input, black.data(), 7 * pixels * sizeof(float), TO_DEVICE) == 0
            && (channels == 7 || cuda_memcpy(input + 7 * pixels, planes, 4 * pixels * sizeof(float), TO_DEVICE) == 0);
    }

    // one source frame goes to the gpu once, the edge pixels fill the padding
    bool upload(int slot, const uint8_t* frame)
    {
        const size_t pixels = (size_t)wide * high;
        const size_t size = held[slot].size();
        if (memcmp(held[slot].data(), frame, size) == 0)
            return true;
        if (slot == 0 && memcmp(held[1].data(), frame, size) == 0) {
            // the clip moved on by one frame, so the second frame is the first one now
            if (cuda_memcpy(input, input + 3 * pixels, 3 * pixels * sizeof(float), ON_DEVICE))
                return false;
            held[0] = held[1];
            return true;
        }
        bands(high, [&](int from, int to) {
            for (int y = from; y < to; y++) {
                const uint8_t* line = frame + (size_t)std::min(y, height - 1) * width * 3;
                float* row = planes + (size_t)y * wide;
                for (int x = 0; x < wide; x++) {
                    const uint8_t* pixel = line + std::min(x, width - 1) * 3;
                    row[x] = pixel[0] / 255.f;
                    row[pixels + x] = pixel[1] / 255.f;
                    row[2 * pixels + x] = pixel[2] / 255.f;
                }
            }
        });
        memcpy(held[slot].data(), frame, size);
        return cuda_memcpy(input + slot * 3 * pixels, planes, 3 * pixels * sizeof(float), TO_DEVICE) == 0;
    }

    int process(const uint8_t* first, const uint8_t* second, int w, int h, float at, uint8_t* out)
    {
        if (!context && !prepare(w, h))
            return 1;
        if (w != width || h != height)
            return 1;
        const size_t pixels = (size_t)wide * high;
        if (!upload(0, first) || !upload(1, second))
            return 1;
        if (at != time) {
            std::fill_n(planes, pixels, at);
            if (cuda_memcpy(input + 6 * pixels, planes, pixels * sizeof(float), TO_DEVICE))
                return 1;
            time = at;
        }
        // the default stream, so the copy that follows waits for the result
        if (!context->enqueueV3(nullptr))
            return 1;
        if (cuda_memcpy(planes, output, 3 * pixels * sizeof(float), TO_HOST))
            return 1;
        bands(height, [&](int from, int to) {
            for (int y = from; y < to; y++) {
                const float* row = planes + (size_t)y * wide;
                uint8_t* line = out + (size_t)y * width * 3;
                for (int x = 0; x < width; x++) {
                    for (int plane = 0; plane < 3; plane++)
                        line[x * 3 + plane] = (uint8_t)(std::clamp(row[plane * pixels + x], 0.f, 1.f) * 255.f + 0.5f);
                }
            }
        });
        return 0;
    }
};

}

EXPORT void* rife_create(const char* models)
{
    if (!load_libraries())
        return nullptr;
    Rife* rife = new Rife;
    rife->folder = std::filesystem::u8path(models);
    return rife;
}

// loads the engine for a size or builds it, and tells how far the build is
EXPORT int rife_prepare(void* rife, int width, int height, void (*report)(void*, float), void* user)
{
    Rife* self = (Rife*)rife;
    self->monitor.report = report;
    self->monitor.user = user;
    const bool ready = self->prepare(width, height);
    self->monitor.report = nullptr;
    return ready ? 0 : 1;
}

EXPORT int rife_process(void* rife, const uint8_t* first, const uint8_t* second, int width, int height, float time, uint8_t* output)
{
    return ((Rife*)rife)->process(first, second, width, height, time, output);
}

EXPORT void rife_destroy(void* rife)
{
    delete (Rife*)rife;
}
