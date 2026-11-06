#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <vector>

#include "command.h"
#include "gpu.h"
#include "layer.h"
#include "modelbin.h"
#include "net.h"
#include "paramdict.h"
#include "pipeline.h"

#include "network.h"

#include "compose.comp.hex.h"

namespace {

struct Lost { };

// the flow net of rife as convolutions of ncnn with one shader step between them, in one command buffer
class Graph : public Backend {
public:
    struct Step {
        // a convolution, or null for a step of the shader
        ncnn::Layer* layer = nullptr;
        std::vector<int> inputs;
        int output = 0;
        int64_t macs = 0;
        std::vector<Part> parts;
        ncnn::VkMat table;
        int count = 0, w = 0, h = 0, channels = 0;
        ncnn::Pipeline* shader = nullptr;
        // the channel of the output where the step writes
        int base = 0;
        // the tensors that no later step reads
        std::vector<int> dead;
    };

    struct Program {
        std::vector<Step> steps;
        int tensors = 0;
        int output = -1;
    };

    // the features of a frame, if the model has them, and the net of a pair
    Program frame, pair;

    ~Graph()
    {
        clear();
        for (const auto& [channels, shader] : shaders)
            delete shader;
    }

    void load(const std::filesystem::path& folder, const ncnn::VulkanDevice* device, const ncnn::Option& options)
    {
        gpu = device;
        opt = options;
        std::string text(compose_comp_data, sizeof(compose_comp_data));
        if (ncnn::compile_spirv_module(text.data(), (int)text.size(), opt, spirv))
            throw std::runtime_error("rife could not load its model on the gpu");
        model = read_model(folder);
        plan = plan_of(model);
    }

    // the channels of the picture of a frame, and of its features after them
    int channels() const
    {
        return plan.cached[0] >= 0 ? 3 + plan.features : 3;
    }

    void lower(int wide, int high)
    {
        if (wide == width && high == height)
            return;
        clear();
        width = height = 0;
        weights = new ncnn::VkWeightAllocator(gpu);
        if (plan.cached[0] >= 0) {
            current = &frame;
            const Leaf features = Lowering(*this, model, plan).frame(leaf(3, wide, high));
            if (frame.steps.empty() || frame.steps.back().layer || frame.steps.back().output != features.id || features.c != plan.features)
                throw std::runtime_error("the model has no frame features");
            frame.steps.back().base = 3;
            seal(frame, features.id, 1);
        }
        current = &pair;
        const Leaf first = leaf(channels(), wide, high);
        const Leaf second = leaf(channels(), wide, high);
        seal(pair, Lowering(*this, model, plan).pair(first, second).id, 2);
        upload();
        width = wide;
        height = high;
    }

    // the given tensors of the program are set, the others are made here and let go after their last step
    int run(const Program& program, std::vector<ncnn::VkMat>& mats, float time, ncnn::VkCompute& cmd, const ncnn::Option& options,
        const std::function<int()>& flush, int64_t& work, int64_t submit) const
    {
        for (const Step& step : program.steps) {
            // a submit ends after this many multiply-adds
            if (work >= submit) {
                work = 0;
                if (flush())
                    return 1;
            }
            ncnn::VkMat& out = mats[step.output];
            if (step.layer) {
                if (step.layer->forward(mats[step.inputs[0]], out, cmd, options))
                    return 1;
                work += step.macs;
            } else {
                if (out.empty())
                    out = tensor(step.w, step.h, step.channels, options);
                record(step, mats, time, cmd);
            }
            for (int id : step.dead)
                mats[id].release();
        }
        return 0;
    }

    static ncnn::VkMat tensor(int w, int h, int channels, const ncnn::Option& options)
    {
        const int pack = channels % 4 == 0 ? 4 : 1;
        ncnn::VkMat mat;
        mat.create(w, h, channels / pack, (size_t)(pack * (options.use_fp16_storage ? 2 : 4)), pack, options.blob_vkallocator);
        if (mat.empty())
            throw Lost();
        return mat;
    }

    Leaf convolution(const std::string&, const Leaf& input, const std::vector<float>& weight, const std::vector<float>& bias, int outputs, int stride,
        float slope, bool activated) override
    {
        Step step;
        step.layer = make_convolution(weight, bias, outputs, stride, slope, activated);
        layers.push_back(step.layer);
        step.inputs = { input.id };
        const int h = stride == 2 ? (input.h - 1) / 2 + 1 : input.h, w = stride == 2 ? (input.w - 1) / 2 + 1 : input.w;
        step.macs = (int64_t)w * h * (int64_t)weight.size();
        return emit(step, outputs, w, h);
    }

    Leaf compose(const std::vector<Leaf>& inputs, const std::vector<Part>& parts, bool timed, int channels, int w, int h) override
    {
        if (timed && current == &frame)
            throw std::runtime_error("the model reads the time in the frame step");
        Step step;
        for (const Leaf& input : inputs)
            step.inputs.push_back(input.id);
        step.parts = parts;
        step.count = (int)parts.size();
        const bool simple = parts.size() == 1 && parts[0].kind == LIN && parts[0].n <= 8 && parts[0].terms > 0;
        step.shader = shader(simple ? parts[0].n : 0);
        step.w = w;
        step.h = h;
        step.channels = channels;
        return emit(step, channels, w, h);
    }

private:
    const ncnn::VulkanDevice* gpu = nullptr;
    ncnn::Option opt;
    std::vector<uint32_t> spirv;
    // the shader of the steps for any step, and for each number of channels of a step that sums its sources
    std::map<int, ncnn::Pipeline*> shaders;
    ncnn::VkAllocator* weights = nullptr;
    Model model;
    Plan plan;
    std::vector<ncnn::Layer*> layers;
    Program* current = nullptr;
    int width = 0, height = 0;

    // 0 channels for a step of any parts, else one part that sums its sources
    ncnn::Pipeline* shader(int channels)
    {
        ncnn::Pipeline*& made = shaders[channels];
        if (!made) {
            ncnn::vk_specialization_type simple, count;
            simple.i = channels > 0;
            count.i = channels;
            made = new ncnn::Pipeline(gpu);
            made->set_optimal_local_size_xyz(32, 4, 1);
            if (made->create(spirv.data(), spirv.size() * 4, { simple, count }))
                throw std::runtime_error("rife could not load its model on the gpu");
        }
        return made;
    }

    void clear()
    {
        for (Program* program : { &frame, &pair })
            *program = Program();
        for (ncnn::Layer* layer : layers) {
            layer->destroy_pipeline(opt);
            delete layer;
        }
        layers.clear();
        // the allocator of the weights frees its memory only when it goes
        delete weights;
        weights = nullptr;
    }

    Leaf leaf(int channels, int w, int h)
    {
        return { current->tensors++, channels, h, w };
    }

    Leaf emit(Step& step, int channels, int w, int h)
    {
        const Leaf made = leaf(channels, w, h);
        step.output = made.id;
        current->steps.push_back(std::move(step));
        return made;
    }

    // the given tensors are the first ones
    static void seal(Program& program, int output, int given)
    {
        program.output = output;
        std::vector<int> last(program.tensors, -1);
        for (size_t index = 0; index < program.steps.size(); index++) {
            for (int id : program.steps[index].inputs)
                last[id] = (int)index;
        }
        for (int id = given; id < (int)last.size(); id++) {
            if (id != output && last[id] >= 0)
                program.steps[last[id]].dead.push_back(id);
        }
    }

    ncnn::Layer* make_convolution(
        const std::vector<float>& weight, const std::vector<float>& bias, int outputs, int stride, float slope, bool activated)
    {
        ncnn::Layer* made = ncnn::create_layer_vulkan("Convolution");
        made->vkdev = gpu;
        ncnn::ParamDict pd;
        pd.set(0, outputs);
        pd.set(1, 3);
        pd.set(3, stride);
        pd.set(4, 1);
        pd.set(5, 1);
        pd.set(6, (int)weight.size());
        if (activated) {
            pd.set(9, 2);
            ncnn::Mat params(1);
            params[0] = slope;
            pd.set(10, params);
        }
        const ncnn::Mat parts[2] = { ncnn::Mat((int)weight.size(), (void*)weight.data()).clone(), ncnn::Mat(outputs, (void*)bias.data()).clone() };
        ncnn::VkWeightStagingAllocator staging(gpu);
        ncnn::Option options = opt;
        options.blob_vkallocator = weights;
        options.staging_vkallocator = &staging;
        ncnn::VkTransfer cmd(gpu);
        if (made->load_param(pd) || made->load_model(ncnn::ModelBinFromMatArray(parts)) || made->create_pipeline(opt)
            || made->upload_model(cmd, options) || cmd.submit_and_wait()) {
            made->destroy_pipeline(opt);
            delete made;
            throw std::runtime_error("rife could not load its model on the gpu");
        }
        return made;
    }

    // the tables of the steps go to the gpu as they are
    void upload()
    {
        ncnn::VkAllocator* staging = gpu->acquire_staging_allocator();
        ncnn::Option options = opt;
        options.use_fp16_packed = options.use_fp16_storage = false;
        options.blob_vkallocator = weights;
        options.staging_vkallocator = staging;
        ncnn::VkCompute cmd(gpu);
        for (Program* program : { &frame, &pair }) {
            for (Step& step : program->steps) {
                if (step.layer)
                    continue;
                // ncnn copies a mat in blocks of 16 bytes
                ncnn::Mat words((int)(step.parts.size() * sizeof(Part) / 4));
                memcpy(words.data, step.parts.data(), step.parts.size() * sizeof(Part));
                cmd.record_upload(words, step.table, options);
            }
        }
        const int failed = cmd.submit_and_wait();
        gpu->reclaim_staging_allocator(staging);
        if (failed)
            throw std::runtime_error("rife could not load its model on the gpu");
        for (Program* program : { &frame, &pair }) {
            for (Step& step : program->steps) {
                step.parts.clear();
                if (!step.layer && step.table.empty())
                    throw std::runtime_error("rife could not load its model on the gpu");
            }
        }
    }

    void record(const Step& step, const std::vector<ncnn::VkMat>& mats, float time, ncnn::VkCompute& cmd) const
    {
        const auto meta = [](const ncnn::VkMat& mat) { return (int)mat.cstep * 2 + (mat.elempack == 4); };
        std::vector<ncnn::VkMat> bindings(MAX_INPUTS + 2);
        std::vector<ncnn::vk_constant_type> constants(19);
        for (size_t index = 0; index < step.inputs.size(); index++) {
            bindings[index] = mats[step.inputs[index]];
            constants[index].i = meta(bindings[index]);
        }
        const ncnn::VkMat& out = mats[step.output];
        bindings[MAX_INPUTS] = out;
        bindings[MAX_INPUTS + 1] = step.table;
        const int ints[6] = { meta(out), step.w, step.h, step.base, step.count, step.channels };
        for (int index = 0; index < 6; index++)
            constants[MAX_INPUTS + index].i = ints[index];
        constants[MAX_INPUTS + 6].f = time;
        ncnn::VkMat dispatcher;
        dispatcher.w = step.w;
        dispatcher.h = step.h;
        dispatcher.c = 1;
        cmd.record_pipeline(step.shader, bindings, constants, dispatcher);
    }
};

}
