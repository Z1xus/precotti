#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
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
#define PROCESS GetCurrentProcessId()
#define EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#include <unistd.h>
#define PROCESS getpid()
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#include "NvInfer.h"

#include "kernels.h"
#include "kernels.ptx.h"

using namespace nvinfer1;

namespace {

// the first error since the last forget, later ones follow from it
// tensorrt can tell its errors from any thread
std::mutex failure_lock;
std::string failure;

// tensorrt and the guard report from noexcept code, where a failed allocation would end the app
void fail(const char* message) noexcept
{
    try {
        const std::lock_guard<std::mutex> guard(failure_lock);
        if (failure.empty())
            failure = message;
    } catch (...) {
    }
}

void fail(const std::string& message)
{
    fail(message.c_str());
}

void forget()
{
    const std::lock_guard<std::mutex> guard(failure_lock);
    failure.clear();
}

const char* failure_text()
{
    thread_local std::string copy;
    const std::lock_guard<std::mutex> guard(failure_lock);
    copy = failure;
    return copy.c_str();
}

struct Logger : ILogger {
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kERROR)
            fail(message);
    }
} logger;

// tensorrt and cuda are loaded at run time, so the build needs the headers only
int32_t (*library_version)();
void* (*create_builder)(void*, int32_t);
void* (*create_runtime)(void*, int32_t);
int (*cuda_malloc)(void**, size_t);
int (*cuda_free)(void*);
int (*cuda_mem_info)(size_t*, size_t*);
int (*cuda_malloc_host)(void**, size_t);
int (*cuda_free_host)(void*);
int (*cuda_memcpy)(void*, const void*, size_t, int, cudaStream_t);
int (*cuda_stream_create)(cudaStream_t*, unsigned);
int (*cuda_stream_destroy)(cudaStream_t);
int (*cuda_stream_wait)(cudaStream_t);
int (*cuda_event_create)(cudaEvent_t*, unsigned);
int (*cuda_event_record)(cudaEvent_t, cudaStream_t);
int (*cuda_event_wait)(cudaEvent_t);
int (*cuda_event_destroy)(cudaEvent_t);
int (*cuda_stream_wait_event)(cudaStream_t, cudaEvent_t, unsigned);
int (*cuda_library_load)(void**, const void*, void*, void**, unsigned, void**, unsigned*, unsigned);
int (*cuda_library_kernel)(void**, void*, const char*);
IPluginRegistry* (*plugin_registry)();

struct Dim3 {
    unsigned x, y, z;
};
int (*cuda_launch)(const void*, Dim3, Dim3, void**, size_t, cudaStream_t);

struct Kernels {
    void* compose = nullptr;
    void* simple = nullptr;
    void* yuv_in = nullptr;
    void* yuv_out = nullptr;
} kernels;

const int TO_DEVICE = 1, TO_HOST = 2;
const unsigned NON_BLOCKING = 1, BLOCKING_SYNC = 1, DISABLE_TIMING = 2;

#define NAME_(text) #text
#define NAME(text) NAME_(text)
#if _WIN32
const char* const CUDA = "cudart64_12.dll";
const char* const INFER = "nvinfer_" NAME(NV_TENSORRT_MAJOR) ".dll";
// tensorrt loads its builder resource by name when it makes an engine, so it must be loaded already
const char* const RESOURCE = "nvinfer_builder_resource_" NAME(NV_TENSORRT_MAJOR) ".dll";

// the folder of this library
std::filesystem::path here()
{
    HMODULE module = nullptr;
    wchar_t name[32768];
    const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, (LPCWSTR)&here, &module) || !GetModuleFileNameW(module, name, 32768))
        return { };
    return std::filesystem::path(name).parent_path();
}

// the library finds the libraries that it needs in its own folder
void* open(const std::filesystem::path& file, std::string& why)
{
    void* library = LoadLibraryExW(file.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!library) {
        const DWORD code = GetLastError();
        char text[512] = { };
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), text,
            sizeof(text), nullptr);
        why = text;
        why.erase(why.find_last_not_of(" \r\n") + 1);
        why = "windows error " + std::to_string(code) + ", " + why;
    }
    return library;
}

void* symbol(void* library, const char* name)
{
    return (void*)GetProcAddress((HMODULE)library, name);
}
#else
const char* const CUDA = "libcudart.so.12";
const char* const INFER = "libnvinfer.so." NAME(NV_TENSORRT_MAJOR);

std::filesystem::path here()
{
    Dl_info info;
    if (!dladdr((void*)&here, &info))
        return { };
    return std::filesystem::path(info.dli_fname).parent_path();
}

void* open(const std::filesystem::path& file, std::string& why)
{
    void* library = dlopen(file.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!library)
        why = dlerror();
    return library;
}

void* symbol(void* library, const char* name)
{
    return dlsym(library, name);
}
#endif

void* load(const std::filesystem::path& folder, const char* name)
{
    const std::filesystem::path file = folder / name;
    std::error_code error;
    if (!std::filesystem::exists(file, error)) {
        fail(std::string(name) + " is not in " + folder.u8string());
        return nullptr;
    }
    std::string why;
    void* library = open(file, why);
    if (!library)
        fail(std::string("cannot load ") + name + ", " + why);
    return library;
}

bool cuda_ok(int code)
{
    if (code == 0)
        return true;
    // cudaErrorMemoryAllocation and cudaErrorInsufficientDriver
    fail(code == 2 ? "not enough gpu memory" : code == 35 ? "the nvidia driver is too old for cuda 12" : "cuda error " + std::to_string(code));
    return false;
}

template <class F>
bool bind(F& function, void* library, const char* name)
{
    function = (F)symbol(library, name);
    if (!function)
        fail(std::string(name) + " is missing, the libraries are not made for this plugin");
    return function != nullptr;
}

bool load_libraries()
{
    static std::mutex lock;
    static bool loaded = false;
    const std::lock_guard<std::mutex> guard(lock);
    if (loaded)
        return true;
    const std::filesystem::path folder = here() / "tensorrt";
    void* cuda = load(folder, CUDA);
    void* infer = cuda ? load(folder, INFER) : nullptr;
#if _WIN32
    if (infer && !load(folder, RESOURCE))
        return false;
#endif
    if (!infer || !bind(library_version, infer, "getInferLibVersion"))
        return false;
    if (library_version() != NV_TENSORRT_VERSION) {
        fail(std::string(INFER) + " is version " + std::to_string(library_version()) + ", the plugin needs " + std::to_string(NV_TENSORRT_VERSION));
        return false;
    }
    loaded = bind(create_builder, infer, "createInferBuilder_INTERNAL") && bind(create_runtime, infer, "createInferRuntime_INTERNAL")
        && bind(plugin_registry, infer, "getPluginRegistry") && bind(cuda_malloc, cuda, "cudaMalloc") && bind(cuda_free, cuda, "cudaFree")
        && bind(cuda_mem_info, cuda, "cudaMemGetInfo") && bind(cuda_malloc_host, cuda, "cudaMallocHost") && bind(cuda_free_host, cuda, "cudaFreeHost")
        && bind(cuda_memcpy, cuda, "cudaMemcpyAsync") && bind(cuda_stream_create, cuda, "cudaStreamCreateWithFlags")
        && bind(cuda_stream_destroy, cuda, "cudaStreamDestroy") && bind(cuda_stream_wait, cuda, "cudaStreamSynchronize")
        && bind(cuda_event_create, cuda, "cudaEventCreateWithFlags") && bind(cuda_event_record, cuda, "cudaEventRecord")
        && bind(cuda_event_wait, cuda, "cudaEventSynchronize") && bind(cuda_event_destroy, cuda, "cudaEventDestroy")
        && bind(cuda_stream_wait_event, cuda, "cudaStreamWaitEvent") && bind(cuda_library_load, cuda, "cudaLibraryLoadData")
        && bind(cuda_library_kernel, cuda, "cudaLibraryGetKernel") && bind(cuda_launch, cuda, "cudaLaunchKernel");
    if (loaded) {
        // the first call makes the context
        void* library = nullptr;
        loaded = cuda_ok(cuda_free(nullptr)) && cuda_ok(cuda_library_load(&library, kernels_ptx, nullptr, nullptr, 0, nullptr, nullptr, 0))
            && cuda_ok(cuda_library_kernel(&kernels.compose, library, "compose")) && cuda_ok(cuda_library_kernel(&kernels.simple, library, "simple"))
            && cuda_ok(cuda_library_kernel(&kernels.yuv_in, library, "yuv_in")) && cuda_ok(cuda_library_kernel(&kernels.yuv_out, library, "yuv_out"));
    }
    return loaded;
}

// these two phases take almost all of the build
struct Monitor : IProgressMonitor {
    void (*report)(void*, float) = nullptr;
    void* user = nullptr;
    float base = 0, span = 1;
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
            report(user, base + span * std::min(done + 0.5f * (step + 1) / steps, 1.f));
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

// a file appears complete or not at all, also when two processes write it
bool write(const std::filesystem::path& file, const void* data, size_t size)
{
    std::filesystem::path part = file;
    part += "." + std::to_string(PROCESS) + ".part";
    std::ofstream stream(part, std::ios::binary);
    stream.write((const char*)data, (std::streamsize)size);
    stream.close();
    std::error_code error;
    if (stream.good())
        std::filesystem::rename(part, file, error);
    if (!stream.good() || error) {
        std::filesystem::remove(part, error);
        return false;
    }
    return true;
}

std::vector<char> build(const std::filesystem::path& name, const std::filesystem::path& learned, const Edit& edit, Monitor& monitor)
{
    std::vector<char> plan;
    // the workspace defaults to all gpu memory, other programs hold some
    size_t available = 0, total = 0;
    const bool known = cuda_ok(cuda_mem_info(&available, &total));
    IBuilder* builder = (IBuilder*)create_builder(&logger, NV_TENSORRT_VERSION);
    if (!builder) {
        fail("tensorrt could not start");
        return plan;
    }
    INetworkDefinition* network = builder->createNetworkV2(1U << (int)NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    IBuilderConfig* config = builder->createBuilderConfig();
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!network || !config || !profile) {
        fail("tensorrt could not set up the build");
    } else if (edit(*network, *profile)) {
        config->addOptimizationProfile(profile);
        if (known)
            config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, available);
        monitor.done = 0;
        config->setProgressMonitor(&monitor);
        // what the builder learned about the gpu in other builds makes this one shorter
        const std::vector<char> remembered = read(learned);
        ITimingCache* cache = config->createTimingCache(remembered.data(), remembered.size());
        // a cache that is torn or from another gpu is not used
        bool set = cache && config->setTimingCache(*cache, false);
        if (!set) {
            delete cache;
            cache = config->createTimingCache(nullptr, 0);
            set = cache && config->setTimingCache(*cache, false);
        }
        if (!set) {
            fail("tensorrt could not make a timing cache");
        } else if (IHostMemory* memory = builder->buildSerializedNetwork(*network, *config)) {
            plan.assign((const char*)memory->data(), (const char*)memory->data() + memory->size());
            delete memory;
            // the app can build the same engine two times at once, so the file appears complete
            write(name, plan.data(), plan.size());
            if (IHostMemory* timing = cache->serialize()) {
                write(learned, timing->data(), timing->size());
                delete timing;
            }
        }
        delete cache;
    }
    delete config;
    delete network;
    delete builder;
    return plan;
}

// the engine from its file, or built and kept in it
ICudaEngine* load_engine(IRuntime* runtime, const std::filesystem::path& name, const std::filesystem::path& learned, const Edit& edit,
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
        plan = build(name, learned, edit, monitor);
        if (!plan.empty())
            engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
    }
    // a stale engine that did not load is not a later failure
    if (engine)
        forget();
    return engine;
}

}
