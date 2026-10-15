#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
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
        if (severity <= Severity::kERROR) {
            fprintf(stderr, "tensorrt: %s\n", message);
            failure = message;
        }
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
int (*cuda_memcpy)(void*, const void*, size_t, int);

const int TO_DEVICE = 1, TO_HOST = 2, ON_DEVICE = 3;

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
    create_builder = (decltype(create_builder))symbol(infer, "createInferBuilder_INTERNAL");
    create_runtime = (decltype(create_runtime))symbol(infer, "createInferRuntime_INTERNAL");
    create_parser = (decltype(create_parser))symbol(onnx, "createNvOnnxParser_INTERNAL");
    cuda_malloc = (decltype(cuda_malloc))symbol(cuda, "cudaMalloc");
    cuda_free = (decltype(cuda_free))symbol(cuda, "cudaFree");
    cuda_malloc_host = (decltype(cuda_malloc_host))symbol(cuda, "cudaMallocHost");
    cuda_free_host = (decltype(cuda_free_host))symbol(cuda, "cudaFreeHost");
    cuda_memcpy = (decltype(cuda_memcpy))symbol(cuda, "cudaMemcpy");
    if (create_builder && create_runtime && create_parser && cuda_malloc && cuda_free && cuda_malloc_host && cuda_free_host && cuda_memcpy)
        state = 1;
    return state > 0;
}

template <typename Rows>
void bands(int rows, Rows work)
{
    const int count = std::clamp((int)std::thread::hardware_concurrency(), 1, 8);
    std::vector<std::thread> threads;
    for (int band = 0; band < count; band++)
        threads.emplace_back(work, rows * band / count, rows * (band + 1) / count);
    for (std::thread& thread : threads)
        thread.join();
}

// tells how far the build of an engine is, from the steps of its first phase
struct Monitor : IProgressMonitor {
    void (*report)(void*, float) = nullptr;
    void* user = nullptr;
    std::string phase;
    int steps = 0;

    void phaseStart(const char* name, const char* parent, int32_t count) noexcept override
    {
        if (!parent && phase.empty()) {
            phase = name;
            steps = count;
        }
    }

    bool stepComplete(const char* name, int32_t step) noexcept override
    {
        if (report && phase == name && steps > 0)
            report(user, (float)(step + 1) / steps);
        return true;
    }

    void phaseFinish(const char*) noexcept override { }
};

std::vector<char> read(const std::filesystem::path& file)
{
    std::ifstream stream(file, std::ios::binary);
    return { std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
}

std::vector<char> build(const std::filesystem::path& onnx, const std::filesystem::path& name, int high, int wide, Monitor& monitor)
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
    if (parser && parser->parse(model.data(), model.size())) {
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (OptProfileSelector kind : { OptProfileSelector::kMIN, OptProfileSelector::kOPT, OptProfileSelector::kMAX })
            profile->setDimensions(network->getInput(0)->getName(), kind, Dims4(1, network->getInput(0)->getDimensions().d[1], high, wide));
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

// the engine for a model and a size, from its file or built and kept in it
ICudaEngine* load_engine(IRuntime* runtime, const std::filesystem::path& onnx, const std::filesystem::path& name, int high, int wide, Monitor& monitor)
{
    std::vector<char> plan = read(name);
    ICudaEngine* engine = plan.empty() ? nullptr : runtime->deserializeCudaEngine(plan.data(), plan.size());
    if (!engine) {
        plan = build(onnx, name, high, wide, monitor);
        if (!plan.empty())
            engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
    }
    return engine;
}

}
