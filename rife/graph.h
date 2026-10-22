#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "command.h"
#include "net.h"
#include "pipeline.h"

struct Lost { };

// the flow net of rife with its own steps between the convolutions of ncnn, in one command buffer
class Graph {
public:
    ~Graph();
    // the net must make its layers with this, before it loads the model
    static void keep(ncnn::Net& net);
    bool load(ncnn::Net& net);
    ncnn::VkMat feature(int role, int w, int h, const ncnn::Option& opt) const;
    // a submit ends after this many multiply-adds
    int run(const ncnn::VkMat (&frames)[2], float time, ncnn::VkMat& output, ncnn::VkCompute& cmd, const ncnn::Option& opt,
        const ncnn::VkMat (&features)[2], const bool (&filled)[2], const std::function<int()>& flush, int64_t submit);

private:
    struct Value {
        ncnn::VkMat mat;
        int from = 0;
        int channels = 0;
        std::shared_ptr<const Value> flow;
    };

    ncnn::Net* net = nullptr;
    const ncnn::VulkanDevice* gpu = nullptr;
    ncnn::VkAllocator* weights = nullptr;
    ncnn::Pipeline* map = nullptr;
    ncnn::Pipeline* interp = nullptr;
    ncnn::Pipeline* warp = nullptr;
    ncnn::Pipeline* shuffle = nullptr;
    int inputs[3] = {}, output = 0;
    // a layer that runs inside the one before it (one more than the input it passes on), the activation or scale it adds there,
    // and how many layers read each blob
    std::vector<int> folded, activation, readers;
    // the convolutions that also do the residual sum and the activation after them
    std::vector<ncnn::Layer*> replaced;
    // a deconvolution that runs as a convolution and a shuffle by this factor
    std::vector<int> spread;
    // the channels of each blob, and the concat that a layer writes into with the channel where its part starts
    std::vector<int> width, into, offset;
    // a concat that only a resize reads, its parts are resized into the output of that resize
    std::vector<int> pushed;
    std::vector<float> scale;
    std::vector<int> lazy;
    // layers that depend on one frame alone, and per frame the blob the rest reads and its layer
    std::vector<int> prefix;
    int cached[2] = { -1, -1 }, made_by[2] = { -1, -1 };
    std::vector<ncnn::VkMat> constants;

    ncnn::Pipeline* pipeline(const char* data, int size, int x, int y, int z);
    bool fuse(int index);
    bool unfold(int index);
    ncnn::Layer* make_convolution(const ncnn::Mat& weight, const ncnn::Mat& bias, int outputs, float slope);
    void move(const Value& image, const Value& flow, int outw, int outh, float k, const ncnn::VkMat& result, int at, ncnn::VkCompute& cmd);
    void spread_out(const Value& source, int factor, const ncnn::VkMat& result, int at, ncnn::VkCompute& cmd);
    ncnn::VkMat tensor(int w, int h, int channels, const ncnn::Option& opt) const;
    Value whole(const Value& value, ncnn::VkCompute& cmd, const ncnn::Option& opt);
    void resample(const Value& from, int outw, int outh, float k, const ncnn::VkMat& to, int at, ncnn::VkCompute& cmd);
    void copy(const Value& from, const ncnn::VkMat& to, int at, ncnn::VkCompute& cmd);
    void apply(const Value& a, const Value* b, int op, float ka, float kb, int act, float slope, const ncnn::VkMat& out, int at, ncnn::VkCompute& cmd);
};
