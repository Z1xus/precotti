#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#if _WIN32
#define NOMINMAX
#include <windows.h>
#define EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

namespace {

// the last error on this thread, for the app to show
thread_local std::string failure;

struct Logger : ILogger {
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kERROR)
            failure = message;
    }
} logger;

// tensorrt and cuda are loaded at run time, so the build needs the headers only
void* (*create_builder)(void*, int32_t);
void* (*create_runtime)(void*, int32_t);
void* (*create_parser)(void*, void*, int);
int (*cuda_malloc)(void**, size_t);
int (*cuda_free)(void*);
int (*cuda_malloc_host)(void**, size_t);
int (*cuda_free_host)(void*);
int (*cuda_memcpy)(void*, const void*, size_t, int, cudaStream_t);
int (*cuda_stream_create)(cudaStream_t*, unsigned);
int (*cuda_stream_destroy)(cudaStream_t);
int (*cuda_event_create)(cudaEvent_t*, unsigned);
int (*cuda_event_record)(cudaEvent_t, cudaStream_t);
int (*cuda_event_wait)(cudaEvent_t);
int (*cuda_event_destroy)(cudaEvent_t);

const int TO_DEVICE = 1, TO_HOST = 2;
const unsigned NON_BLOCKING = 1, BLOCKING_SYNC = 1, DISABLE_TIMING = 2;

#if _WIN32
const wchar_t* const CUDA = L"cudart64_12.dll";
const wchar_t* const INFER = L"nvinfer_10.dll";
const wchar_t* const ONNX = L"nvonnxparser_10.dll";

// the folder of this library
std::filesystem::path here()
{
    HMODULE module = nullptr;
    wchar_t name[32768];
    const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, (LPCWSTR)&here, &module) || !GetModuleFileNameW(module, name, 32768))
        return {};
    return std::filesystem::path(name).parent_path();
}

void* open(const std::filesystem::path& file)
{
    // the library finds the libraries that it needs in its own folder
    return LoadLibraryExW(file.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}

void* symbol(void* library, const char* name)
{
    return (void*)GetProcAddress((HMODULE)library, name);
}
#else
const char* const CUDA = "libcudart.so.12";
const char* const INFER = "libnvinfer.so.10";
const char* const ONNX = "libnvonnxparser.so.10";

std::filesystem::path here()
{
    Dl_info info;
    if (!dladdr((void*)&here, &info))
        return {};
    return std::filesystem::path(info.dli_fname).parent_path();
}

void* open(const std::filesystem::path& file)
{
    return dlopen(file.c_str(), RTLD_NOW | RTLD_GLOBAL);
}

void* symbol(void* library, const char* name)
{
    return dlsym(library, name);
}
#endif

bool load_libraries()
{
    static int state = 0;
    if (state != 0)
        return state > 0;
    state = -1;
    const std::filesystem::path folder = here() / "tensorrt";
    void* cuda = open(folder / CUDA);
    void* infer = open(folder / INFER);
    void* onnx = open(folder / ONNX);
    if (!cuda || !infer || !onnx)
        return false;
#if _WIN32
    // tensorrt loads its builder resource by name only when it makes an engine
    SetDllDirectoryW(folder.c_str());
#endif
    create_builder = (decltype(create_builder))symbol(infer, "createInferBuilder_INTERNAL");
    create_runtime = (decltype(create_runtime))symbol(infer, "createInferRuntime_INTERNAL");
    create_parser = (decltype(create_parser))symbol(onnx, "createNvOnnxParser_INTERNAL");
    cuda_malloc = (decltype(cuda_malloc))symbol(cuda, "cudaMalloc");
    cuda_free = (decltype(cuda_free))symbol(cuda, "cudaFree");
    cuda_malloc_host = (decltype(cuda_malloc_host))symbol(cuda, "cudaMallocHost");
    cuda_free_host = (decltype(cuda_free_host))symbol(cuda, "cudaFreeHost");
    cuda_memcpy = (decltype(cuda_memcpy))symbol(cuda, "cudaMemcpyAsync");
    cuda_stream_create = (decltype(cuda_stream_create))symbol(cuda, "cudaStreamCreateWithFlags");
    cuda_stream_destroy = (decltype(cuda_stream_destroy))symbol(cuda, "cudaStreamDestroy");
    cuda_event_create = (decltype(cuda_event_create))symbol(cuda, "cudaEventCreateWithFlags");
    cuda_event_record = (decltype(cuda_event_record))symbol(cuda, "cudaEventRecord");
    cuda_event_wait = (decltype(cuda_event_wait))symbol(cuda, "cudaEventSynchronize");
    cuda_event_destroy = (decltype(cuda_event_destroy))symbol(cuda, "cudaEventDestroy");
    if (create_builder && create_runtime && create_parser && cuda_malloc && cuda_free && cuda_malloc_host && cuda_free_host && cuda_memcpy
        && cuda_stream_create && cuda_stream_destroy && cuda_event_create && cuda_event_record && cuda_event_wait && cuda_event_destroy)
        state = 1;
    return state > 0;
}

// these two phases take almost all of the build
struct Monitor : IProgressMonitor {
    void (*report)(void*, float) = nullptr;
    void* user = nullptr;
    int steps = 0;
    float done = 0;

    static bool timing(const char* name)
    {
        return !strcmp(name, "Timing graph nodes") || !strcmp(name, "Timing tensor reformats");
    }

    void phaseStart(const char* name, const char*, int32_t count) noexcept override
    {
        if (timing(name))
            steps = count;
    }

    bool stepComplete(const char* name, int32_t step) noexcept override
    {
        if (report && timing(name) && steps > 0)
            report(user, std::min(done + 0.5f * (step + 1) / steps, 1.f));
        return true;
    }

    void phaseFinish(const char* name) noexcept override
    {
        if (timing(name))
            done += 0.5f;
    }
};

using Edit = std::function<bool(INetworkDefinition&, IOptimizationProfile&)>;

std::vector<char> read(const std::filesystem::path& file)
{
    std::ifstream stream(file, std::ios::binary);
    return { std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
}

std::vector<char> build(const std::filesystem::path& onnx, const std::filesystem::path& name, const Edit& edit, Monitor& monitor)
{
    std::vector<char> plan;
    IBuilder* builder = (IBuilder*)create_builder(&logger, NV_TENSORRT_VERSION);
    if (!builder)
        return plan;
    INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = (nvonnxparser::IParser*)create_parser(network, &logger, NV_ONNX_PARSER_VERSION);
    IBuilderConfig* config = builder->createBuilderConfig();
    // the parser gets the bytes, it cannot open a path with letters outside of ascii on windows
    const std::vector<char> model = read(onnx);
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (parser && parser->parse(model.data(), model.size()) && edit(*network, *profile)) {
        config->addOptimizationProfile(profile);
        config->setFlag(BuilderFlag::kFP16);
        config->setProgressMonitor(&monitor);
        // what the builder learned about the gpu in other builds makes this one shorter
        const std::filesystem::path learned = onnx.parent_path().parent_path() / "timing.cache";
        const std::vector<char> known = read(learned);
        ITimingCache* cache = config->createTimingCache(known.data(), known.size());
        config->setTimingCache(*cache, true);
        if (IHostMemory* memory = builder->buildSerializedNetwork(*network, *config)) {
            plan.assign((const char*)memory->data(), (const char*)memory->data() + memory->size());
            delete memory;
            // the app can build the same engine two times at once, so the file appears complete
            std::filesystem::path part = name;
            part += ".part";
            std::ofstream(part, std::ios::binary).write(plan.data(), (std::streamsize)plan.size());
            std::error_code failed;
            std::filesystem::rename(part, name, failed);
            if (IHostMemory* timing = cache->serialize()) {
                std::ofstream(learned, std::ios::binary).write((const char*)timing->data(), (std::streamsize)timing->size());
                delete timing;
            }
        }
        delete cache;
    }
    delete config;
    delete parser;
    delete network;
    delete builder;
    return plan;
}

// the engine from its file, or built and kept in it
ICudaEngine* load_engine(IRuntime* runtime, const std::filesystem::path& onnx, const std::filesystem::path& name, const Edit& edit,
    const std::function<bool(ICudaEngine&)>& fits, Monitor& monitor)
{
    // two builds at once need twice the gpu memory, and one that waits can load the engine of the other
    static std::mutex building;
    const std::lock_guard<std::mutex> guard(building);
    std::vector<char> plan = read(name);
    ICudaEngine* engine = plan.empty() ? nullptr : runtime->deserializeCudaEngine(plan.data(), plan.size());
    if (engine && !fits(*engine)) {
        delete engine;
        engine = nullptr;
    }
    if (!engine) {
        plan = build(onnx, name, edit, monitor);
        if (!plan.empty())
            engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
    }
    return engine;
}

}
