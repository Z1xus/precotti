#include "graph.h"

#include <algorithm>
#include <string>

#include "layer/binaryop.h"
#include "layer/concat.h"
#include "layer/convolution.h"
#include "layer/crop.h"
#include "layer/deconvolution.h"
#include "layer/eltwise.h"
#include "layer/interp.h"
#include "layer/memorydata.h"
#include "layer/pixelshuffle.h"
#include "layer/relu.h"
#include "layer/slice.h"
#include "layer/vulkan/convolution_vulkan.h"
#include "layer/vulkan/deconvolution_vulkan.h"
#include "layer/vulkan/memorydata_vulkan.h"
#include "modelbin.h"
#include "paramdict.h"

#include "interp.comp.hex.h"
#include "map.comp.hex.h"
#include "shuffle.comp.hex.h"
#include "warp.comp.hex.h"

namespace {

// ncnn frees the weights once they are on the gpu, the graph needs them to fold the residual sums
struct Kept : ncnn::Convolution_vulkan {
    ncnn::Mat weight, bias;

    int load_model(const ncnn::ModelBin& mb) override
    {
        const int failed = ncnn::Convolution_vulkan::load_model(mb);
        weight = weight_data.clone();
        bias = bias_data.clone();
        return failed;
    }
};

struct KeptDeconv : ncnn::Deconvolution_vulkan {
    ncnn::Mat weight, bias;

    int load_model(const ncnn::ModelBin& mb) override
    {
        const int failed = ncnn::Deconvolution_vulkan::load_model(mb);
        weight = weight_data.clone();
        bias = bias_data.clone();
        return failed;
    }
};

struct KeptData : ncnn::MemoryData_vulkan {
    ncnn::Mat kept;

    int load_model(const ncnn::ModelBin& mb) override
    {
        const int failed = ncnn::MemoryData_vulkan::load_model(mb);
        kept = data.clone();
        return failed;
    }
};

ncnn::Layer* convolution(void*)
{
    return new Kept;
}

ncnn::Layer* memory(void*)
{
    return new KeptData;
}

ncnn::Layer* deconvolution(void*)
{
    return new KeptDeconv;
}

enum { ADD, SUB, MUL, DIV, MAX, MIN, RSUB = 7, RDIV = 8, COPY = 99, FILL = 100 };
enum { NONE, LEAKY, SIGMOID };

bool is(const ncnn::Layer* layer, const char* type)
{
    return layer->type == type;
}

// the channels that a crop keeps, from the first one to the one after the last
std::pair<int, int> kept(const ncnn::Crop* crop, int channels)
{
    int from = ((const int*)crop->starts)[0], to = ((const int*)crop->ends)[0];
    from = from < 0 ? from + channels : from;
    to = std::min(to < 0 ? to + channels : to, channels);
    return { from, to };
}

std::pair<int, int> resized(const ncnn::Interp* resize, int w, int h)
{
    return { resize->output_width ? resize->output_width : (int)(w * resize->width_scale),
        resize->output_height ? resize->output_height : (int)(h * resize->height_scale) };
}

bool plain(const ncnn::Interp* resize)
{
    return resize->resize_type == 2 && !resize->align_corner && !resize->dynamic_target_size;
}

int part(const ncnn::Slice* slice, int index, int from, int channels)
{
    const int size = ((const int*)slice->slices)[index];
    return size == -233 ? channels - from : size;
}

// a thread writes four channels of a pixel, four pixels of a plane, or the few channels of a pixel at any offset of a pack
int lanes_of(const ncnn::VkMat& out, int at, int channels)
{
    if (out.elempack == 1)
        return 1;
    return at % 4 == 0 && channels % 4 == 0 ? 4 : 0;
}

}

Graph::~Graph()
{
    constants.clear();
    for (ncnn::Layer* layer : replaced) {
        if (layer) {
            layer->destroy_pipeline(net->opt);
            delete layer;
        }
    }
    delete weights;
    delete map;
    delete interp;
    delete warp;
    delete shuffle;
}

ncnn::Pipeline* Graph::pipeline(const char* data, int size, int x, int y, int z)
{
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(data, size, net->opt, spirv))
        return nullptr;
    ncnn::Pipeline* made = new ncnn::Pipeline(gpu);
    made->set_optimal_local_size_xyz(x, y, z);
    made->create(spirv.data(), spirv.size() * 4, {});
    return made;
}

bool Graph::load(ncnn::Net& model)
{
    net = &model;
    gpu = model.vulkan_device();
    map = pipeline(map_comp_data, sizeof(map_comp_data), 256, 1, 1);
    interp = pipeline(interp_comp_data, sizeof(interp_comp_data), 8, 8, 1);
    warp = pipeline(warp_comp_data, sizeof(warp_comp_data), 8, 8, 1);
    shuffle = pipeline(shuffle_comp_data, sizeof(shuffle_comp_data), 8, 8, 1);
    if (!map || !interp || !warp || !shuffle)
        return false;

    const std::vector<ncnn::Blob>& blobs = model.blobs();
    const std::vector<ncnn::Layer*>& layers = model.layers();
    for (int index = 0; index < (int)blobs.size(); index++) {
        const std::string& name = blobs[index].name;
        if (name == "in0" || name == "in1" || name == "in2")
            inputs[name[2] - '0'] = index;
        if (name == "out0")
            output = index;
    }
    readers.assign(blobs.size(), 0);
    for (const ncnn::Layer* layer : layers) {
        for (int bottom : layer->bottoms)
            readers[bottom]++;
    }
    folded.assign(layers.size(), 0);
    replaced.assign(layers.size(), nullptr);
    weights = new ncnn::VkWeightAllocator(gpu);
    for (int index = 0; index < (int)layers.size(); index++) {
        if (!is(layers[index], "Convolution"))
            continue;
        if (!fuse(index))
            return false;
        ((Kept*)layers[index])->weight.release();
        ((Kept*)layers[index])->bias.release();
    }
    // a resize to the same size passes its input on
    for (int index = 0; index < (int)layers.size(); index++) {
        const ncnn::Interp* resize = (const ncnn::Interp*)layers[index];
        if (is(resize, "Interp") && plain(resize) && !resize->output_width && !resize->output_height && resize->width_scale == 1
            && resize->height_scale == 1)
            folded[index] = 1;
    }
    spread.assign(layers.size(), 0);
    for (int index = 0; index < (int)layers.size(); index++) {
        if (!is(layers[index], "Deconvolution"))
            continue;
        if (!unfold(index))
            return false;
        ((KeptDeconv*)layers[index])->weight.release();
        ((KeptDeconv*)layers[index])->bias.release();
    }
    activation.assign(layers.size(), NONE);
    scale.assign(layers.size(), 1.f);
    const auto next = [&](int top) { return blobs[top].consumer; };
    for (int index = 0; index < (int)layers.size(); index++) {
        const ncnn::Layer* layer = layers[index];
        if (folded[index] || layer->tops.size() != 1 || readers[layer->tops[0]] != 1)
            continue;
        const int reader = next(layer->tops[0]);
        const ncnn::Layer* after = layers[reader];
        if (folded[reader])
            continue;
        if (is(layer, "BinaryOp") && (is(after, "ReLU") || is(after, "Sigmoid"))) {
            folded[reader] = 1;
            activation[index] = is(after, "ReLU") ? LEAKY : SIGMOID;
        }
        if (is(layer, "Interp") && is(after, "BinaryOp")) {
            const ncnn::BinaryOp* op = (const ncnn::BinaryOp*)after;
            if (op->with_scalar && (op->op_type == MUL || op->op_type == DIV)) {
                folded[reader] = 1;
                scale[index] = op->op_type == MUL ? op->b : 1 / op->b;
            }
        }
    }

    // the channels of each blob, so a concat can be made before the layers that write into it run
    width.assign(blobs.size(), 0);
    width[inputs[0]] = width[inputs[1]] = 3;
    width[inputs[2]] = 1;
    for (const ncnn::Layer* layer : layers) {
        const std::vector<int>& tops = layer->tops;
        const int first = layer->bottoms.empty() ? 0 : width[layer->bottoms[0]];
        if (is(layer, "Input"))
            continue;
        if (is(layer, "Split")) {
            for (int top : tops)
                width[top] = first;
        } else if (is(layer, "MemoryData")) {
            width[tops[0]] = ((const ncnn::MemoryData*)layer)->c;
        } else if (is(layer, "Crop")) {
            const auto [from, to] = kept((const ncnn::Crop*)layer, first);
            width[tops[0]] = to - from;
        } else if (is(layer, "Slice")) {
            int from = 0;
            for (size_t top = 0; top < tops.size(); top++) {
                width[tops[top]] = part((const ncnn::Slice*)layer, (int)top, from, first);
                from += width[tops[top]];
            }
        } else if (is(layer, "Concat")) {
            for (int bottom : layer->bottoms)
                width[tops[0]] += width[bottom];
        } else if (is(layer, "BinaryOp") && layer->bottoms.size() == 2) {
            width[tops[0]] = std::max(first, width[layer->bottoms[1]]);
        } else if (is(layer, "PixelShuffle")) {
            width[tops[0]] = first / 4;
        } else if (is(layer, "Convolution")) {
            width[tops[0]] = ((const ncnn::Convolution*)layer)->num_output;
        } else if (is(layer, "Deconvolution")) {
            width[tops[0]] = ((const ncnn::Deconvolution*)layer)->num_output;
        } else {
            width[tops[0]] = first;
        }
    }
    // resizing is the same for each channel, so the resize of a concat is the concat of the resized parts
    pushed.assign(layers.size(), -1);
    for (int index = 0; index < (int)layers.size(); index++) {
        const int top = layers[index]->tops.empty() ? 0 : layers[index]->tops[0];
        if (!is(layers[index], "Concat") || readers[top] != 1)
            continue;
        const int reader = blobs[top].consumer;
        if (is(layers[reader], "Interp") && !folded[reader] && plain((const ncnn::Interp*)layers[reader])) {
            pushed[index] = reader;
            folded[reader] = 1;
        }
    }
    // a layer of the graph that alone feeds a concat writes its part of it in place
    into.assign(layers.size(), -1);
    offset.assign(layers.size(), 0);
    for (int index = 0; index < (int)layers.size(); index++) {
        if (!is(layers[index], "Concat") || pushed[index] >= 0)
            continue;
        int at = 0;
        for (int bottom : layers[index]->bottoms) {
            int writer = blobs[bottom].producer;
            bool alone = readers[bottom] == 1;
            while (folded[writer]) {
                writer = blobs[layers[writer]->bottoms[folded[writer] - 1]].producer;
                alone = alone && readers[layers[writer]->tops[0]] == 1;
            }
            const ncnn::Layer* layer = layers[writer];
            const bool ours = is(layer, "BinaryOp") || is(layer, "Eltwise") || is(layer, "ReLU") || is(layer, "Sigmoid")
                || is(layer, "Interp") || is(layer, "rife.Warp") || is(layer, "PixelShuffle") || is(layer, "Concat")
                || (is(layer, "Deconvolution") && spread[writer]);
            if (alone && ours) {
                into[writer] = index;
                offset[writer] = at;
            }
            at += width[bottom];
        }
    }

    // a moved image that only a concat reads is moved and resized in one step
    lazy.assign(layers.size(), 0);
    for (int index = 0; index < (int)layers.size(); index++) {
        if (!is(layers[index], "rife.Warp") || width[layers[index]->tops[0]] > 4)
            continue;
        int top = layers[index]->tops[0], reader = blobs[top].consumer;
        if (readers[top] != 1)
            continue;
        const ncnn::Crop* crop = (const ncnn::Crop*)layers[reader];
        if (is(crop, "Crop") && crop->axes.w == 1 && ((const int*)crop->axes)[0] == 0 && kept(crop, width[top]) == std::pair<int, int> { 0, width[top] }) {
            top = crop->tops[0];
            reader = blobs[top].consumer;
            if (readers[top] != 1)
                continue;
        }
        if (is(layers[reader], "Concat") && pushed[reader] >= 0)
            lazy[index] = 1;
    }

    // layers that depend on one frame give the same blob each time, so make it once
    std::vector<int> depends(blobs.size(), 0);
    prefix.assign(layers.size(), 0);
    for (int index = 0; index < (int)layers.size(); index++) {
        const ncnn::Layer* layer = layers[index];
        int mask = 0;
        for (int bottom : layer->bottoms)
            mask |= depends[bottom];
        if (is(layer, "Input"))
            mask = 1 << (layer->tops[0] == inputs[0] ? 0 : layer->tops[0] == inputs[1] ? 1 : 2);
        for (int top : layer->tops)
            depends[top] = mask;
        if ((mask == 1 || mask == 2) && !is(layer, "Input") && !is(layer, "Split") && !is(layer, "Crop") && !is(layer, "Slice") && !layer->bottoms.empty())
            prefix[index] = mask;
    }
    for (int role = 0; role < 2; role++) {
        std::vector<int> leaving;
        for (int index = 0; index < (int)layers.size(); index++) {
            for (int bottom : layers[index]->bottoms) {
                const bool leaves = prefix[blobs[bottom].producer] == 1 << role && prefix[index] != 1 << role;
                if (leaves && std::find(leaving.begin(), leaving.end(), bottom) == leaving.end())
                    leaving.push_back(bottom);
            }
        }
        const int writer = leaving.size() == 1 ? blobs[leaving[0]].producer : -1;
        if (writer >= 0 && spread[writer] && !folded[writer] && into[writer] < 0) {
            cached[role] = leaving[0];
            made_by[role] = writer;
        } else {
            for (int& mask : prefix) {
                if (mask == 1 << role)
                    mask = 0;
            }
        }
    }

    constants.resize(layers.size());
    ncnn::VkAllocator* staging = gpu->acquire_staging_allocator();
    ncnn::Option opt = model.opt;
    opt.blob_vkallocator = weights;
    opt.staging_vkallocator = staging;
    ncnn::VkCompute cmd(gpu);
    for (int index = 0; index < (int)layers.size(); index++) {
        if (!is(layers[index], "MemoryData"))
            continue;
        ncnn::Mat& data = ((KeptData*)layers[index])->kept;
        ncnn::Mat packed;
        ncnn::convert_packing(data, packed, data.c % 4 == 0 ? 4 : 1, opt);
        cmd.record_upload(packed, constants[index], opt);
        data.release();
    }
    const bool uploaded = cmd.submit_and_wait() == 0;
    gpu->reclaim_staging_allocator(staging);
    return uploaded;
}

// a slope of 1 means no activation
ncnn::Layer* Graph::make_convolution(const ncnn::Mat& weight, const ncnn::Mat& bias, int outputs, float slope)
{
    ncnn::Layer* made = ncnn::create_layer_vulkan("Convolution");
    made->vkdev = gpu;
    ncnn::ParamDict pd;
    pd.set(0, outputs);
    pd.set(1, 3);
    pd.set(4, 1);
    pd.set(5, 1);
    pd.set(6, weight.w);
    if (slope != 1) {
        pd.set(9, 2);
        ncnn::Mat params(1);
        params[0] = slope;
        pd.set(10, params);
    }
    const ncnn::Mat parts[2] = { weight, bias };
    ncnn::VkWeightStagingAllocator staging(gpu);
    ncnn::Option opt = net->opt;
    opt.blob_vkallocator = weights;
    opt.staging_vkallocator = &staging;
    ncnn::VkTransfer cmd(gpu);
    if (made->load_param(pd) || made->load_model(ncnn::ModelBinFromMatArray(parts)) || made->create_pipeline(net->opt)
        || made->upload_model(cmd, opt) || cmd.submit_and_wait()) {
        delete made;
        return nullptr;
    }
    return made;
}

// a deconvolution with kernel 4 and stride 2 is a 3x3 convolution with four outputs for each output, one for each pixel
// of a 2x2 block, that a shuffle then spreads; a shuffle by 2 after it joins into one shuffle by 4
bool Graph::unfold(int index)
{
    const std::vector<ncnn::Blob>& blobs = net->blobs();
    const std::vector<ncnn::Layer*>& layers = net->layers();
    KeptDeconv* deconv = (KeptDeconv*)layers[index];
    const int outputs = deconv->num_output;
    const int inputs = deconv->weight_data_size / outputs / 16;
    const bool plain = deconv->kernel_w == 4 && deconv->kernel_h == 4 && deconv->stride_w == 2 && deconv->stride_h == 2
        && deconv->dilation_w == 1 && deconv->dilation_h == 1 && deconv->pad_left == 1 && deconv->pad_right == 1 && deconv->pad_top == 1
        && deconv->pad_bottom == 1 && deconv->output_pad_right == 0 && deconv->output_pad_bottom == 0 && deconv->output_w == 0
        && deconv->activation_type == 0;
    if (!plain)
        return true;
    const int top = deconv->tops[0], reader = readers[top] == 1 ? blobs[top].consumer : -1;
    const bool joined = reader >= 0 && is(layers[reader], "PixelShuffle") && ((const ncnn::PixelShuffle*)layers[reader])->upscale_factor == 2
        && ((const ncnn::PixelShuffle*)layers[reader])->mode == 0 && outputs % 4 == 0;
    const int factor = joined ? 4 : 2;
    // the channel of each output pixel in its block of factor x factor
    const auto spot = [&](int p, int py, int px) {
        if (!joined)
            return p * 4 + py * 2 + px;
        const int c = p / 4, dy = p % 4 / 2, dx = p % 2;
        return c * 16 + (py * 2 + dy) * 4 + px * 2 + dx;
    };
    ncnn::Mat weight(outputs * 4 * inputs * 9);
    ncnn::Mat bias(outputs * 4);
    weight.fill(0.f);
    const float* w = deconv->weight;
    for (int p = 0; p < outputs; p++) {
        for (int py = 0; py < 2; py++) {
            for (int px = 0; px < 2; px++) {
                const int o = spot(p, py, px);
                bias[o] = deconv->bias_term ? deconv->bias[p] : 0;
                for (int q = 0; q < inputs; q++) {
                    for (int ty = 0; ty < 3; ty++) {
                        for (int tx = 0; tx < 3; tx++) {
                            const int ky = py + 3 - 2 * ty, kx = px + 3 - 2 * tx;
                            if (ky >= 0 && ky < 4 && kx >= 0 && kx < 4)
                                weight[((o * inputs + q) * 3 + ty) * 3 + tx] = w[(p * inputs + q) * 16 + ky * 4 + kx];
                        }
                    }
                }
            }
        }
    }
    ncnn::Layer* made = make_convolution(weight, bias, outputs * 4, 1);
    if (!made)
        return false;
    replaced[index] = made;
    spread[index] = factor;
    if (joined)
        folded[reader] = 1;
    return true;
}

void Graph::keep(ncnn::Net& net)
{
    net.register_custom_layer("Convolution", convolution);
    net.register_custom_layer("MemoryData", memory);
    net.register_custom_layer("Deconvolution", deconvolution);
}

// leaky(conv(x) * beta + x) is one convolution: beta scales the weights, and x adds one to the middle tap of each channel
bool Graph::fuse(int index)
{
    const std::vector<ncnn::Blob>& blobs = net->blobs();
    const std::vector<ncnn::Layer*>& layers = net->layers();
    Kept* conv = (Kept*)layers[index];
    const int channels = conv->num_output, x = conv->bottoms[0];
    const bool plain = conv->kernel_w == 3 && conv->kernel_h == 3 && conv->stride_w == 1 && conv->stride_h == 1 && conv->dilation_w == 1
        && conv->dilation_h == 1 && conv->pad_left == 1 && conv->pad_right == 1 && conv->pad_top == 1 && conv->pad_bottom == 1
        && conv->activation_type == 0 && conv->weight_data_size == channels * channels * 9;
    int top = conv->tops[0];
    if (!plain || readers[top] != 1)
        return true;
    const auto binary = [&](int layer, int op) {
        const ncnn::Layer* found = layers[layer];
        return is(found, "BinaryOp") && found->bottoms.size() == 2 && ((const ncnn::BinaryOp*)found)->op_type == op;
    };
    int at = blobs[top].consumer, mul = -1;
    const ncnn::Mat* beta = nullptr;
    if (binary(at, MUL) && layers[at]->bottoms[0] == top) {
        const ncnn::Layer* constant = layers[blobs[layers[at]->bottoms[1]].producer];
        const KeptData* data = (const KeptData*)constant;
        if (!is(constant, "MemoryData") || data->w != 1 || data->h != 1 || data->c != channels)
            return true;
        beta = &data->kept;
        mul = at;
        top = layers[at]->tops[0];
        if (readers[top] != 1)
            return true;
        at = blobs[top].consumer;
    }
    if (!binary(at, ADD))
        return true;
    const int pass = layers[at]->bottoms[0] == top ? 0 : 1;
    const int other = layers[at]->bottoms[1 - pass], source = blobs[x].producer;
    if (other != x && (blobs[other].producer != source || !is(layers[source], "Split")))
        return true;
    const int add = at;
    top = layers[add]->tops[0];
    const int relu = blobs[top].consumer;
    if (readers[top] != 1 || !is(layers[relu], "ReLU"))
        return true;

    ncnn::Mat weight = conv->weight;
    ncnn::Mat bias(channels);
    float* w = weight;
    for (int o = 0; o < channels; o++) {
        const float scale = beta ? beta->channel(o)[0] : 1;
        bias[o] = conv->bias_term ? conv->bias[o] * scale : 0;
        for (int k = 0; k < channels * 9; k++)
            w[o * channels * 9 + k] *= scale;
        w[(o * channels + o) * 9 + 4] += 1;
    }
    ncnn::Layer* made = make_convolution(weight, bias, channels, ((const ncnn::ReLU*)layers[relu])->slope);
    if (!made)
        return false;
    replaced[index] = made;
    if (mul >= 0)
        folded[mul] = 1;
    folded[add] = 1 + pass;
    folded[relu] = 1;
    return true;
}

ncnn::VkMat Graph::tensor(int w, int h, int channels, const ncnn::Option& opt) const
{
    const int pack = channels % 4 == 0 ? 4 : 1;
    ncnn::VkMat mat;
    mat.create(w, h, channels / pack, (size_t)(pack * (opt.use_fp16_storage ? 2 : 4)), pack, opt.blob_vkallocator);
    return mat;
}

void Graph::apply(const Value& a, const Value* b, int op, float ka, float kb, int act, float slope, const ncnn::VkMat& out, int at, ncnn::VkCompute& cmd)
{
    const int size = a.mat.w * a.mat.h;
    const int channels = b ? std::max(a.channels, b->channels) : a.channels;
    const int lanes = lanes_of(out, at, channels);
    // a fixed row of threads keeps the groups of the dispatch under the limit of vulkan
    const int row = 65536;
    // a group of four pixels past the end writes into the padding of the plane
    const int total = lanes == 4 ? channels / 4 * size : lanes == 1 ? channels * ((size + 3) / 4) : size;
    std::vector<ncnn::vk_constant_type> constants(20);
    const int ints[17] = { size, channels, a.mat.elempack, (int)a.mat.cstep, a.from,
        b ? b->mat.elempack : 0, b ? (int)b->mat.cstep : 0, b ? b->from : 0, b ? b->channels : 1, b ? b->mat.w * b->mat.h : 1,
        out.elempack, (int)out.cstep, at, op, act, lanes, row };
    for (int i = 0; i < 17; i++)
        constants[i].i = ints[i];
    constants[17].f = ka;
    constants[18].f = kb;
    constants[19].f = slope;
    ncnn::VkMat dispatcher;
    dispatcher.w = std::min(total, row);
    dispatcher.h = (total + row - 1) / row;
    dispatcher.c = 1;
    cmd.record_pipeline(map, { a.mat, b ? b->mat : a.mat, out }, constants, dispatcher);
}

void Graph::resample(const Value& a, int outw, int outh, float k, const ncnn::VkMat& result, int at, ncnn::VkCompute& cmd)
{
    const int w = a.mat.w, h = a.mat.h, channels = a.channels;
    const int lanes = lanes_of(result, at, channels) == 4 ? 4 : 1;
    std::vector<ncnn::vk_constant_type> constants(15);
    const int ints[12] = { w, h, a.mat.elempack, (int)a.mat.cstep, a.from, outw, outh, result.elempack, (int)result.cstep, at, channels, lanes };
    for (int i = 0; i < 12; i++)
        constants[i].i = ints[i];
    constants[12].f = (float)w / outw;
    constants[13].f = (float)h / outh;
    constants[14].f = k;
    ncnn::VkMat dispatcher;
    dispatcher.w = outw;
    dispatcher.h = outh;
    dispatcher.c = lanes == 4 ? channels / 4 : 1;
    cmd.record_pipeline(interp, { a.mat, result }, constants, dispatcher);
}

void Graph::move(const Value& image, const Value& flow, int outw, int outh, float k, const ncnn::VkMat& result, int at, ncnn::VkCompute& cmd)
{
    const int w = image.mat.w, h = image.mat.h;
    // the shader moves four channels at most
    for (int done = 0; done < image.channels; done += 4) {
        std::vector<ncnn::vk_constant_type> constants(17);
        const int ints[14] = { w, h, std::min(4, image.channels - done), image.mat.elempack, (int)image.mat.cstep, image.from + done,
            flow.mat.elempack, (int)flow.mat.cstep, flow.from, result.elempack, (int)result.cstep, at + done, outw, outh };
        for (int i = 0; i < 14; i++)
            constants[i].i = ints[i];
        constants[14].f = (float)w / outw;
        constants[15].f = (float)h / outh;
        constants[16].f = k;
        ncnn::VkMat dispatcher;
        dispatcher.w = outw;
        dispatcher.h = outh;
        dispatcher.c = 1;
        cmd.record_pipeline(warp, { image.mat, flow.mat, result }, constants, dispatcher);
    }
}

void Graph::spread_out(const Value& source, int factor, const ncnn::VkMat& result, int at, ncnn::VkCompute& cmd)
{
    const int w = source.mat.w, h = source.mat.h, channels = source.channels / factor / factor;
    const int lanes = lanes_of(result, at, channels) == 4 ? 4 : 1;
    std::vector<ncnn::vk_constant_type> constants(10);
    const int ints[10] = { w, h, channels, source.mat.elempack, (int)source.mat.cstep, result.elempack, (int)result.cstep, at, lanes, factor };
    for (int i = 0; i < 10; i++)
        constants[i].i = ints[i];
    ncnn::VkMat dispatcher;
    dispatcher.w = w * factor * lanes;
    dispatcher.h = h * factor;
    dispatcher.c = channels / lanes;
    cmd.record_pipeline(shuffle, { source.mat, result }, constants, dispatcher);
}

void Graph::copy(const Value& from, const ncnn::VkMat& to, int at, ncnn::VkCompute& cmd)
{
    apply(from, nullptr, COPY, 1, 0, NONE, 0, to, at, cmd);
}

Graph::Value Graph::whole(const Value& value, ncnn::VkCompute& cmd, const ncnn::Option& opt)
{
    if (value.from == 0 && value.channels == value.mat.c * value.mat.elempack)
        return value;
    Value made { tensor(value.mat.w, value.mat.h, value.channels, opt), 0, value.channels };
    copy(value, made.mat, 0, cmd);
    return made;
}

ncnn::VkMat Graph::feature(int role, int w, int h, const ncnn::Option& opt) const
{
    return cached[role] < 0 ? ncnn::VkMat() : tensor(w, h, width[cached[role]], opt);
}

int Graph::run(const ncnn::VkMat (&frames)[2], float time, ncnn::VkMat& out, ncnn::VkCompute& cmd, const ncnn::Option& opt,
    const ncnn::VkMat (&features)[2], const bool (&filled)[2])
{
    const std::vector<ncnn::Layer*>& layers = net->layers();
    std::vector<Value> values(net->blobs().size());
    std::vector<int> left = readers;
    std::vector<ncnn::VkMat> joined(layers.size());
    const auto made = [&](const ncnn::VkMat& mat) { return Value { mat, 0, mat.c * mat.elempack }; };
    // a new tensor, or the part of the concat that the layer feeds
    const auto target = [&](int index, int w, int h, int channels) -> std::pair<ncnn::VkMat, int> {
        for (int role = 0; role < 2; role++) {
            if (index == made_by[role] && !features[role].empty() && !filled[role])
                return { features[role], 0 };
        }
        int concat = into[index], at = offset[index];
        if (concat < 0)
            return { tensor(w, h, channels, opt), 0 };
        // a concat that feeds another one is a part of it
        while (into[concat] >= 0) {
            at += offset[concat];
            concat = into[concat];
        }
        if (joined[concat].empty())
            joined[concat] = tensor(w, h, width[layers[concat]->tops[0]], opt);
        return { joined[concat], at };
    };
    for (int index = 0; index < 2; index++)
        values[inputs[index]] = made(frames[index]);
    ncnn::VkMat moment = tensor(frames[0].w, frames[0].h, 1, opt);
    apply({ frames[0], 0, 1 }, nullptr, FILL, 1, time, NONE, 0, moment, 0, cmd);
    values[inputs[2]] = made(moment);
    for (int role = 0; role < 2; role++) {
        if (!features[role].empty() && filled[role])
            values[cached[role]] = made(features[role]);
    }

    for (int index = 0; index < (int)layers.size(); index++) {
        const ncnn::Layer* layer = layers[index];
        if (prefix[index] && !features[prefix[index] - 1].empty() && filled[prefix[index] - 1])
            continue;
        const std::vector<int>& tops = layer->tops;
        std::vector<Value> bottoms;
        for (int bottom : layer->bottoms)
            bottoms.push_back(values[bottom]);
        for (int bottom : layer->bottoms) {
            if (--left[bottom] == 0)
                values[bottom] = Value();
        }
        if (folded[index] || is(layer, "Input")) {
            if (folded[index])
                values[tops[0]] = bottoms[folded[index] - 1];
            continue;
        }
        const Value& a = bottoms.empty() ? Value() : bottoms[0];
        const int w = a.mat.w, h = a.mat.h, channels = tops.empty() ? 0 : width[tops[0]];

        if (is(layer, "Split")) {
            for (int top : tops)
                values[top] = a;
        } else if (is(layer, "MemoryData")) {
            values[tops[0]] = made(constants[index]);
        } else if (is(layer, "Crop")) {
            const ncnn::Crop* crop = (const ncnn::Crop*)layer;
            if (crop->axes.w != 1 || ((const int*)crop->axes)[0] != 0)
                return 1;
            values[tops[0]] = { a.mat, a.from + kept(crop, a.channels).first, channels, a.flow };
        } else if (is(layer, "Slice")) {
            if (((const ncnn::Slice*)layer)->axis != 0)
                return 1;
            int from = 0;
            for (int top : tops) {
                values[top] = { a.mat, a.from + from, width[top] };
                from += width[top];
            }
        } else if (is(layer, "Concat")) {
            if (((const ncnn::Concat*)layer)->axis != 0)
                return 1;
            if (pushed[index] >= 0) {
                const int resize = pushed[index];
                const auto [outw, outh] = resized((const ncnn::Interp*)layers[resize], w, h);
                const auto [result, from] = target(index, outw, outh, channels);
                int at = from;
                for (const Value& part : bottoms) {
                    if (part.flow)
                        move(part, *part.flow, outw, outh, scale[resize], result, at, cmd);
                    else
                        resample(part, outw, outh, scale[resize], result, at, cmd);
                    at += part.channels;
                }
                values[tops[0]] = { result, from, channels };
                continue;
            }
            const auto [result, from] = target(index, w, h, channels);
            if (into[index] < 0)
                joined[index] = ncnn::VkMat();
            int at = from;
            for (const Value& part : bottoms) {
                if (part.mat.data != result.data || part.from != at)
                    copy(part, result, at, cmd);
                at += part.channels;
            }
            values[tops[0]] = { result, from, channels };
        } else if (is(layer, "BinaryOp")) {
            const ncnn::BinaryOp* op = (const ncnn::BinaryOp*)layer;
            const float slope = activation[index] == LEAKY ? ((const ncnn::ReLU*)layers[net->blobs()[tops[0]].consumer])->slope : 0;
            if (op->with_scalar) {
                const auto [result, at] = target(index, w, h, channels);
                apply(a, nullptr, op->op_type, 1, op->b, activation[index], slope, result, at, cmd);
                values[tops[0]] = { result, at, channels };
            } else {
                // the operand with more channels or pixels goes first, so the other one broadcasts
                const bool swap = bottoms[1].channels > a.channels || bottoms[1].mat.w * bottoms[1].mat.h > w * h;
                const Value& first = swap ? bottoms[1] : a;
                const Value& second = swap ? a : bottoms[1];
                int kind = op->op_type;
                if (swap && kind == SUB)
                    kind = RSUB;
                else if (swap && kind == DIV)
                    kind = RDIV;
                const auto [result, at] = target(index, first.mat.w, first.mat.h, channels);
                apply(first, &second, kind, 1, 1, activation[index], slope, result, at, cmd);
                values[tops[0]] = { result, at, channels };
            }
        } else if (is(layer, "Eltwise")) {
            const ncnn::Eltwise* eltwise = (const ncnn::Eltwise*)layer;
            if (eltwise->op_type != 1 || bottoms.size() != 2)
                return 1;
            const float* coeffs = eltwise->coeffs.empty() ? nullptr : (const float*)eltwise->coeffs;
            const auto [result, at] = target(index, w, h, channels);
            apply(a, &bottoms[1], ADD, coeffs ? coeffs[0] : 1, coeffs ? coeffs[1] : 1, NONE, 0, result, at, cmd);
            values[tops[0]] = { result, at, channels };
        } else if (is(layer, "ReLU") || is(layer, "Sigmoid")) {
            const float slope = is(layer, "ReLU") ? ((const ncnn::ReLU*)layer)->slope : 0;
            const auto [result, at] = target(index, w, h, channels);
            apply(a, nullptr, COPY, 1, 0, is(layer, "ReLU") ? LEAKY : SIGMOID, slope, result, at, cmd);
            values[tops[0]] = { result, at, channels };
        } else if (is(layer, "Interp")) {
            const ncnn::Interp* resize = (const ncnn::Interp*)layer;
            if (!plain(resize))
                return 1;
            const auto [outw, outh] = resized(resize, w, h);
            const auto [result, at] = target(index, outw, outh, channels);
            resample(a, outw, outh, scale[index], result, at, cmd);
            values[tops[0]] = { result, at, channels };
        } else if (is(layer, "rife.Warp")) {
            const Value& flow = bottoms[1];
            if (lazy[index]) {
                values[tops[0]] = { a.mat, a.from, channels, std::make_shared<const Value>(flow) };
                continue;
            }
            const auto [result, at] = target(index, w, h, channels);
            move(a, flow, w, h, 1, result, at, cmd);
            values[tops[0]] = { result, at, channels };
        } else if (is(layer, "PixelShuffle")) {
            const ncnn::PixelShuffle* shuffled = (const ncnn::PixelShuffle*)layer;
            if (shuffled->upscale_factor != 2 || shuffled->mode != 0)
                return 1;
            const auto [result, at] = target(index, w * 2, h * 2, channels);
            spread_out(whole(a, cmd, opt), 2, result, at, cmd);
            values[tops[0]] = { result, at, channels };
        } else if (spread[index]) {
            ncnn::VkMat blocks;
            if (replaced[index]->forward(whole(a, cmd, opt).mat, blocks, cmd, opt))
                return 1;
            const int factor = spread[index], last = factor == 4 ? net->blobs()[tops[0]].consumer : index;
            const int result_channels = width[layers[last]->tops[0]];
            const auto [result, at] = target(index, w * factor, h * factor, result_channels);
            spread_out(made(blocks), factor, result, at, cmd);
            values[tops[0]] = { result, at, result_channels };
        } else if (is(layer, "Convolution") || is(layer, "Deconvolution")) {
            ncnn::VkMat result;
            const ncnn::Layer* run = replaced[index] ? replaced[index] : layer;
            if (run->forward(whole(a, cmd, opt).mat, result, cmd, opt))
                return 1;
            values[tops[0]] = made(result);
        } else {
            return 1;
        }
    }
    out = whole(values[output], cmd, opt).mat;
    return 0;
}
