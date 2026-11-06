#pragma once

#include <mutex>

#include "tensorrt.h"

namespace {

const char* const GLUE = "RifeGlue";

struct Shape {
    int inputs, timed, channels, height, width, count;
    int simple;
    int ref_height, ref_width;
};

class Glue : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    explicit Glue(std::vector<char> spec)
        : spec(std::move(spec))
    {
        fields.nbFields = 1;
        fields.fields = &field;
        refresh();
    }

    Glue(const Glue& other)
        : Glue(other.spec)
    {
    }

    ~Glue() override
    {
        if (device)
            cuda_free(device);
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kBUILD)
            return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME)
            return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }

    IPluginV3* clone() noexcept override
    {
        try {
            return new Glue(*this);
        } catch (...) {
            return nullptr;
        }
    }

    const char* getPluginName() const noexcept override
    {
        return GLUE;
    }
    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }
    const char* getPluginNamespace() const noexcept override
    {
        return "interpolini";
    }

    int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t, const DynamicPluginTensorDesc*, int32_t) noexcept override
    {
        return 0;
    }

    int32_t getOutputDataTypes(DataType* types, int32_t, const DataType*, int32_t) const noexcept override
    {
        types[0] = DataType::kHALF;
        return 0;
    }

    int32_t getOutputShapes(
        const DimsExprs* inputs, int32_t, const DimsExprs*, int32_t, DimsExprs* outputs, int32_t, IExprBuilder& builder) noexcept override
    {
        const auto scaled = [&](const IDimensionExpr* from, int size, int reference) {
            return builder.operation(DimensionOperation::kFLOOR_DIV, *builder.operation(DimensionOperation::kPROD, *from, *builder.constant(size)),
                *builder.constant(reference));
        };
        outputs[0].nbDims = 4;
        outputs[0].d[0] = builder.constant(1);
        outputs[0].d[1] = builder.constant(shape.channels);
        outputs[0].d[2] = scaled(inputs[0].d[2], shape.height, shape.ref_height);
        outputs[0].d[3] = scaled(inputs[0].d[3], shape.width, shape.ref_width);
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* inOut, int32_t, int32_t) noexcept override
    {
        const PluginTensorDesc& desc = inOut[pos].desc;
        if (shape.timed && pos == shape.inputs)
            return desc.type == DataType::kFLOAT && desc.format == TensorFormat::kLINEAR;
        return desc.type == DataType::kHALF && (desc.format == TensorFormat::kLINEAR || (desc.format == TensorFormat::kHWC8 && desc.dims.d[1] >= 8));
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc* out, int32_t) noexcept override
    {
        return fit((int)out[0].dims.d[3], (int)out[0].dims.d[2]) ? 0 : 1;
    }

    static Bind bind(const PluginTensorDesc& desc, const void* data)
    {
        const int c = (int)desc.dims.d[1], h = (int)desc.dims.d[2], w = (int)desc.dims.d[3];
        if (desc.format == TensorFormat::kHWC8)
            return { data, 1, (c + 7) / 8 * 8, w, h };
        return { data, h * w, 1, w, h };
    }

    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void*,
        cudaStream_t stream) noexcept override
    {
        if (!device)
            return 1;
        Compose args = { };
        for (int i = 0; i < shape.inputs; i++)
            args.in[i] = bind(inputDesc[i], inputs[i]);
        args.out = bind(outputDesc[0], outputs[0]);
        args.time = shape.timed ? (const float*)inputs[shape.inputs] : nullptr;
        args.parts = device;
        args.count = shape.count;
        args.channels = shape.channels;
        void* list[] = { &args };
        const Dim3 block = { 32, 4, 1 }, grid = { (unsigned)(args.out.w + 31) / 32, (unsigned)(args.out.h + 3) / 4, 1 };
        return cuda_launch(shape.simple ? kernels.simple : kernels.compose, grid, block, list, shape.count * sizeof(Part), stream);
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    const PluginFieldCollection* getFieldsToSerialize() noexcept override
    {
        return &fields;
    }

private:
    std::vector<char> spec;
    Shape shape = { };
    std::vector<Part> parts;
    Part* device = nullptr;
    int fitted[2] = { 0, 0 };
    PluginField field = { };
    PluginFieldCollection fields = { };

    void refresh()
    {
        memcpy(&shape, spec.data(), sizeof(shape));
        field = PluginField("spec", spec.data(), PluginFieldType::kINT8, (int32_t)spec.size());
        parts.resize(shape.count);
        memcpy(parts.data(), spec.data() + sizeof(shape), shape.count * sizeof(Part));
        if (cuda_malloc((void**)&device, shape.count * sizeof(Part)) == 0 && !fit(shape.width, shape.height)) {
            cuda_free(device);
            device = nullptr;
        }
    }

    bool fit(int width, int height)
    {
        if (!device)
            return true;
        if (width == fitted[0] && height == fitted[1])
            return true;
        std::vector<Part> moved = parts;
        for (Part& part : moved) {
            for (Src& src : part.s) {
                for (int i = 0; i < 3; i++) {
                    src.w[i] = src.w[i] * width / shape.width;
                    src.h[i] = src.h[i] * height / shape.height;
                }
                for (int i = 0; i < 2; i++) {
                    if (src.w[i + 1]) {
                        src.sx[i] = (float)src.w[i] / src.w[i + 1];
                        src.sy[i] = (float)src.h[i] / src.h[i + 1];
                    }
                }
            }
        }
        // the kernels run in other streams
        if (cuda_memcpy(device, moved.data(), moved.size() * sizeof(Part), TO_DEVICE, nullptr) || cuda_stream_wait(nullptr))
            return false;
        fitted[0] = width;
        fitted[1] = height;
        return true;
    }
};

std::vector<char> glue_spec(
    int inputs, bool timed, int channels, int height, int width, int ref_height, int ref_width, const std::vector<Part>& parts)
{
    const bool simple = parts.size() == 1 && parts[0].kind == LIN && parts[0].n <= 8 && parts[0].terms > 0 && !parts[0].sigmoid;
    const Shape shape = { inputs, timed, channels, height, width, (int)parts.size(), simple, ref_height, ref_width };
    std::vector<char> spec(sizeof(shape) + parts.size() * sizeof(Part));
    memcpy(spec.data(), &shape, sizeof(shape));
    memcpy(spec.data() + sizeof(shape), parts.data(), parts.size() * sizeof(Part));
    return spec;
}

struct GlueCreator : IPluginCreatorV3One {
    PluginFieldCollection names = { 0, nullptr };

    IPluginV3* createPlugin(const char*, const PluginFieldCollection* fc, TensorRTPhase) noexcept override
    {
        try {
            if (!fc || fc->nbFields != 1)
                return nullptr;
            const char* data = (const char*)fc->fields[0].data;
            const size_t length = fc->fields[0].length;
            Shape shape;
            if (!data || length < sizeof(shape))
                return nullptr;
            memcpy(&shape, data, sizeof(shape));
            const bool sized = shape.inputs > 0 && shape.inputs <= MAX_INPUTS && shape.channels > 0 && shape.width > 0 && shape.height > 0
                && shape.ref_width > 0 && shape.ref_height > 0;
            if (!sized || shape.count < 1 || length != sizeof(shape) + shape.count * sizeof(Part))
                return nullptr;
            return new Glue(std::vector<char>(data, data + length));
        } catch (...) {
            return nullptr;
        }
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &names;
    }
    const char* getPluginName() const noexcept override
    {
        return GLUE;
    }
    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }
    const char* getPluginNamespace() const noexcept override
    {
        return "interpolini";
    }
};

void register_glue()
{
    static GlueCreator creator;
    static std::mutex lock;
    static bool registered = false;
    const std::lock_guard<std::mutex> guard(lock);
    if (!registered)
        registered = plugin_registry()->registerCreator(creator, "interpolini");
}

}
