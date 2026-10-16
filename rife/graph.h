#pragma once

#include <vector>

#include "command.h"
#include "net.h"
#include "pipeline.h"

// the flow net of rife with its own steps between the convolutions of ncnn, in one command buffer
class Graph {
public:
    ~Graph();
    // the net must make its layers with this, before it loads the model
    static void keep(ncnn::Net& net);
    bool load(ncnn::Net& net);
    int run(const ncnn::VkMat (&frames)[2], float time, ncnn::VkMat& output, ncnn::VkCompute& cmd, const ncnn::Option& opt);

private:
    struct Value {
        ncnn::VkMat mat;
        int from = 0;
        int channels = 0;
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
    std::vector<ncnn::VkMat> constants;

    ncnn::Pipeline* pipeline(const char* data, int size, int x, int y, int z);
    bool fuse(int index);
    bool unfold(int index);
    ncnn::Layer* make_convolution(const ncnn::Mat& weight, const ncnn::Mat& bias, int outputs, float slope);
    void spread_out(const Value& source, int factor, const ncnn::VkMat& result, int at, ncnn::VkCompute& cmd);
    ncnn::VkMat tensor(int w, int h, int channels, const ncnn::Option& opt) const;
    Value whole(const Value& value, ncnn::VkCompute& cmd, const ncnn::Option& opt);
    void resample(const Value& from, int outw, int outh, float k, const ncnn::VkMat& to, int at, ncnn::VkCompute& cmd);
    void copy(const Value& from, const ncnn::VkMat& to, int at, ncnn::VkCompute& cmd);
    void apply(const Value& a, const Value* b, int op, float ka, float kb, int act, float slope, const ncnn::VkMat& out, int at, ncnn::VkCompute& cmd);
};
