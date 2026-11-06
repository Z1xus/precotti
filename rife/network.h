#pragma once

#include <algorithm>
#include <memory>

#include "kernels.h"
#include "model.h"

namespace {

enum { ADD, SUB, MUL, DIV, RSUB = 7 };

// a tensor of a runtime, which names it by an id
struct Leaf {
    int id = -1;
    int c = 0, h = 0, w = 0;

    bool operator==(const Leaf& other) const
    {
        return id == other.id;
    }
};

struct Backend {
    virtual ~Backend() = default;
    // a 3x3 convolution with a leaky activation when it is activated
    virtual Leaf convolution(const std::string& name, const Leaf& input, const std::vector<float>& weight, const std::vector<float>& bias,
        int outputs, int stride, float slope, bool activated) = 0;
    // one step that writes the parts into a new tensor
    virtual Leaf compose(const std::vector<Leaf>& inputs, const std::vector<Part>& parts, bool timed, int channels, int w, int h) = 0;
};

struct View {
    Leaf leaf;
    int c0 = 0, shift = 0, stages = 0;
    int w[3] = { }, h[3] = { };
};

struct Term {
    View view;
    float k = 1;
};

struct Val;
using Ref = std::shared_ptr<Val>;

struct Val {
    enum Kind { LIN, WARP, CONCAT, SIGMOID, WEIGHTED, BLEND } kind = LIN;
    int n = 0;
    int w = 0, h = 0;
    std::vector<Term> terms;
    float bias = 0, time = 0;
    View image, flow;
    float k = 1;
    std::vector<Ref> parts;
    bool inverted = false;
    Leaf stored;
};

struct Plan {
    int inputs[3] = { };
    int output = -1;
    std::vector<int> frame;
    int cached[2] = { -1, -1 };
    int features = 0;
};

bool unfoldable(const Layer& layer)
{
    return layer.is("Deconvolution") && layer.integer(1, 0) == 4 && layer.integer(11, 4) == 4 && layer.integer(3, 1) == 2 && layer.integer(13, 2) == 2
        && layer.integer(2, 1) == 1 && layer.integer(12, 1) == 1 && layer.integer(4, 0) == 1 && layer.integer(15, 1) == 1 && layer.integer(14, 1) == 1
        && layer.integer(16, 1) == 1 && layer.integer(18, 0) == 0 && layer.integer(19, 0) == 0 && layer.integer(20, 0) == 0
        && layer.integer(9, 0) == 0;
}

// the channels from and to of a crop, negative ones count from the end
std::pair<int, int> cropped(const Layer& layer, int n)
{
    const float start = layer.array(9)[0], end = layer.array(10)[0];
    int from = (int)start, to = end >= (float)n ? n : (int)end;
    from = from < 0 ? from + n : from;
    to = to < 0 ? to + n : to;
    if (from < 0 || from >= to || to > n)
        throw std::runtime_error("the model has a crop that is not supported");
    return { from, to };
}

// the channel where each part of a slice starts, and the end, -233 takes the rest
std::vector<int> sliced(const Layer& layer, int n)
{
    std::vector<int> ends = { 0 };
    for (float size : layer.array(0)) {
        const int end = size == -233.f ? n : ends.back() + (int)size;
        if (end <= ends.back() || end > n)
            throw std::runtime_error("the model has a slice that is not supported");
        ends.push_back(end);
    }
    return ends;
}

Plan plan_of(const Model& model)
{
    Plan plan;
    for (int blob = 0; blob < model.blobs; blob++) {
        for (int input = 0; input < 3; input++) {
            if (model.names[blob] == "in" + std::to_string(input))
                plan.inputs[input] = blob;
        }
        if (model.names[blob] == "out0")
            plan.output = blob;
    }
    std::vector<int> depends(model.blobs, 0);
    plan.frame.assign(model.layers.size(), 0);
    for (size_t index = 0; index < model.layers.size(); index++) {
        const Layer& layer = model.layers[index];
        int mask = 0;
        for (int bottom : layer.bottoms)
            mask |= depends[bottom];
        if (layer.is("Input"))
            mask = 1 << (layer.tops[0] == plan.inputs[0] ? 0 : layer.tops[0] == plan.inputs[1] ? 1 : 2);
        for (int top : layer.tops)
            depends[top] = mask;
        if ((mask == 1 || mask == 2) && !layer.is("Input") && !layer.is("Split") && !layer.is("Crop") && !layer.is("Slice") && !layer.bottoms.empty())
            plan.frame[index] = mask;
    }
    std::vector<char> needed(model.layers.size(), 0);
    for (size_t index = model.layers.size(); index-- > 0;) {
        needed[index] |= plan.frame[index] && unfoldable(model.layers[index]);
        for (int bottom : model.layers[index].bottoms) {
            const int writer = model.producer[bottom];
            needed[writer] |= needed[index] && plan.frame[writer];
        }
    }
    for (size_t index = 0; index < model.layers.size(); index++) {
        if (!needed[index])
            plan.frame[index] = 0;
    }
    for (int role = 0; role < 2; role++) {
        std::vector<int> leaving;
        for (size_t index = 0; index < model.layers.size(); index++) {
            for (int bottom : model.layers[index].bottoms) {
                const int writer = model.producer[bottom];
                if (plan.frame[writer] == role + 1 && plan.frame[index] != role + 1
                    && std::find(leaving.begin(), leaving.end(), bottom) == leaving.end())
                    leaving.push_back(bottom);
            }
        }
        const int writer = leaving.size() == 1 ? model.producer[leaving[0]] : -1;
        if (writer < 0 || !unfoldable(model.layers[writer])) {
            plan.frame.assign(model.layers.size(), 0);
            plan.cached[0] = plan.cached[1] = -1;
            return plan;
        }
        plan.cached[role] = leaving[0];
        plan.features = model.layers[writer].integer(0, 0);
    }
    std::vector<const Layer*> frames[2];
    for (size_t index = 0; index < model.layers.size(); index++) {
        if (plan.frame[index])
            frames[plan.frame[index] - 1].push_back(&model.layers[index]);
    }
    bool same = frames[0].size() == frames[1].size();
    for (size_t index = 0; same && index < frames[0].size(); index++) {
        const Layer &a = *frames[0][index], &b = *frames[1][index];
        same = a.type == b.type && a.params == b.params && a.weight == b.weight && a.bias == b.bias;
    }
    if (!same)
        throw std::runtime_error("the model does not encode both frames the same way");
    return plan;
}

struct Lowering {
    Backend& backend;
    const Model& model;
    const Plan& plan;
    std::vector<Ref> values;
    std::vector<char> folded;

    Lowering(Backend& backend, const Model& model, const Plan& plan)
        : backend(backend)
        , model(model)
        , plan(plan)
        , values(model.blobs)
        , folded(model.layers.size(), 0)
    {
    }

    static Ref view_of(const Leaf& leaf, int c0, int n, int shift = 0)
    {
        auto value = std::make_shared<Val>();
        value->n = n;
        value->w = leaf.w << shift;
        value->h = leaf.h << shift;
        Term term;
        term.view.leaf = leaf;
        term.view.c0 = c0;
        term.view.shift = shift;
        term.view.w[0] = value->w;
        term.view.h[0] = value->h;
        value->terms.push_back(term);
        return value;
    }

    static Ref altered(const Ref& value)
    {
        Ref result = std::make_shared<Val>(*value);
        result->stored = Leaf();
        return result;
    }

    Ref stored(const Ref& value)
    {
        return view_of(memory(value), 0, value->n);
    }

    Ref crop(const Ref& value, int from, int to)
    {
        Ref result = altered(value);
        result->n = to - from;
        if (value->kind == Val::LIN) {
            for (Term& term : result->terms)
                term.view.c0 += from;
        } else if (value->kind == Val::WARP) {
            result->image.c0 += from;
        } else if (value->kind == Val::CONCAT) {
            result->parts.clear();
            int at = 0;
            for (const Ref& part : value->parts) {
                const int low = std::max(from, at), high = std::min(to, at + part->n);
                if (low < high)
                    result->parts.push_back(crop(part, low - at, high - at));
                at += part->n;
            }
            if (result->parts.size() == 1)
                return result->parts[0];
        } else {
            throw std::runtime_error("the model crops a value that is not supported");
        }
        return result;
    }

    bool pure(const Ref& value, bool whole = false)
    {
        return value->kind == Val::LIN && value->terms.size() == 1 && value->bias == 0 && value->time == 0 && value->terms[0].view.stages == 0
            && (!whole || (value->terms[0].k == 1 && value->terms[0].view.shift == 0));
    }

    Leaf memory(const Ref& value)
    {
        if (value->stored.id >= 0)
            return value->stored;
        if (pure(value, true) && value->terms[0].view.c0 == 0 && value->terms[0].view.leaf.c == value->n)
            return value->terms[0].view.leaf;
        value->stored = compose({ value }, value->w, value->h, value->n);
        return value->stored;
    }

    Ref warp(const Ref& image, const Ref& flow)
    {
        if (flow->n != 2)
            throw std::runtime_error("the model moves an image by a flow that is not two channels");
        auto value = std::make_shared<Val>();
        value->kind = Val::WARP;
        value->n = image->n;
        Ref plain = image;
        if (pure(image))
            value->k = image->terms[0].k;
        else
            plain = stored(image);
        value->image = plain->terms[0].view;
        value->w = plain->w;
        value->h = plain->h;
        value->flow = (pure(flow, true) ? flow : stored(flow))->terms[0].view;
        if (value->flow.w[0] != value->image.w[0] || value->flow.h[0] != value->image.h[0])
            throw std::runtime_error("the model moves an image by a flow of another size");
        return value;
    }

    Ref resize(const Ref& value, int w, int h)
    {
        if (value->kind == Val::CONCAT) {
            Ref result = altered(value);
            for (Ref& part : result->parts)
                part = resize(part, w, h);
            result->w = w;
            result->h = h;
            return result;
        }
        if (!value->w || (value->w == w && value->h == h))
            return value;
        Ref result = altered(value);
        result->w = w;
        result->h = h;
        if (value->kind == Val::WARP && value->w == value->image.w[0] && value->h == value->image.h[0])
            return result;
        if (value->kind != Val::LIN)
            return resize(stored(value), w, h);
        for (const Term& term : value->terms) {
            if (term.view.stages == 2)
                return resize(stored(value), w, h);
        }
        for (Term& term : result->terms) {
            term.view.stages++;
            term.view.w[term.view.stages] = w;
            term.view.h[term.view.stages] = h;
        }
        return result;
    }

    Ref scaled(const Ref& value, float k, float bias)
    {
        if (value->kind == Val::WARP && bias == 0) {
            Ref result = altered(value);
            result->k *= k;
            return result;
        }
        if (value->kind != Val::LIN)
            return scaled(stored(value), k, bias);
        Ref result = altered(value);
        for (Term& term : result->terms)
            term.k *= k;
        if (k == 0)
            result->terms.clear();
        result->bias = value->bias * k + bias;
        result->time = value->time * k;
        return result;
    }

    Ref sum(const Ref& a, float ka, const Ref& b, float kb)
    {
        Ref left = a->kind == Val::LIN ? a : stored(a), right = b->kind == Val::LIN ? b : stored(b);
        if (left->w && right->w && (left->w != right->w || left->h != right->h))
            throw std::runtime_error("the model adds values of different sizes");
        if (left->terms.size() + right->terms.size() > 2) {
            if (right->terms.size() > 1 || left->terms.size() == 1)
                right = stored(right);
            if (left->terms.size() + right->terms.size() > 2)
                left = stored(left);
        }
        Ref result = altered(left);
        for (Term& term : result->terms)
            term.k *= ka;
        for (Term term : right->terms) {
            term.k *= kb;
            result->terms.push_back(term);
        }
        result->bias = left->bias * ka + right->bias * kb;
        result->time = left->time * ka + right->time * kb;
        result->n = std::max(left->n, right->n);
        // only a constant spreads over the channels
        if ((!left->terms.empty() && left->n != result->n) || (!right->terms.empty() && right->n != result->n))
            throw std::runtime_error("the model adds values of different channels");
        if (!result->w) {
            result->w = right->w;
            result->h = right->h;
        }
        return result;
    }

    static Src bind(const View& view, std::vector<Leaf>& inputs, int stages)
    {
        Src src = { };
        const auto at = std::find(inputs.begin(), inputs.end(), view.leaf);
        src.input = (int)(at - inputs.begin());
        if (at == inputs.end()) {
            if (inputs.size() >= MAX_INPUTS)
                throw std::runtime_error("the model needs more tensors in one step than a step takes");
            inputs.push_back(view.leaf);
        }
        src.c0 = view.c0;
        src.shuffle = 1 << view.shift;
        src.shift = view.shift;
        src.stages = stages;
        src.k = 1;
        for (int i = 0; i < 3; i++) {
            src.w[i] = view.w[i];
            src.h[i] = view.h[i];
        }
        for (int i = 0; i < 2; i++) {
            if (src.w[i + 1]) {
                src.sx[i] = (float)src.w[i] / src.w[i + 1];
                src.sy[i] = (float)src.h[i] / src.h[i + 1];
            }
        }
        return src;
    }

    static Src bind_image(const View& view, std::vector<Leaf>& inputs, int w, int h, float k)
    {
        View sized = view;
        sized.w[1] = w;
        sized.h[1] = h;
        Src src = bind(sized, inputs, 0);
        src.k = k;
        return src;
    }

    static Part part_of(const Ref& value, int at, std::vector<Leaf>& inputs, bool& timed, int w, int h)
    {
        Part part = { };
        part.out_c = at;
        part.n = value->n;
        if (value->kind == Val::LIN) {
            part.kind = LIN;
            part.terms = (int)value->terms.size();
            for (size_t i = 0; i < value->terms.size(); i++) {
                part.s[i] = bind(value->terms[i].view, inputs, value->terms[i].view.stages);
                part.s[i].k = value->terms[i].k;
            }
            part.bias = value->bias;
            part.time = value->time;
            timed |= value->time != 0;
        } else if (value->kind == Val::WARP) {
            part.kind = WARP;
            part.s[0] = bind_image(value->image, inputs, w, h, value->k);
            part.s[1] = bind(value->flow, inputs, 0);
        } else if (value->kind == Val::SIGMOID) {
            part = part_of(value->parts[0], at, inputs, timed, w, h);
            part.sigmoid = 1;
        } else if (value->kind == Val::BLEND) {
            part.kind = BLEND;
            for (int side = 0; side < 2; side++) {
                const Ref& moved = value->parts[side];
                part.s[side * 2] = bind_image(moved->image, inputs, w, h, moved->k);
                part.s[side * 2 + 1] = bind(moved->flow, inputs, 0);
            }
            const Ref& mask = value->parts[2];
            if (mask->terms.size() != 1 || mask->bias != 0 || mask->time != 0 || mask->terms[0].k != 1)
                throw std::runtime_error("the model blends with a mask that is not supported");
            part.s[4] = bind(mask->terms[0].view, inputs, mask->terms[0].view.stages);
        } else {
            throw std::runtime_error("the model has a value that is not supported");
        }
        return part;
    }

    Leaf compose(const std::vector<Ref>& parts, int w, int h, int channels)
    {
        std::vector<Leaf> inputs;
        std::vector<Part> list;
        bool timed = false;
        int at = 0;
        for (const Ref& value : parts) {
            list.push_back(part_of(value, at, inputs, timed, w, h));
            at += value->n;
        }
        if (at != channels || !w || !h || inputs.empty())
            throw std::runtime_error("the model has a value without a size");
        return backend.compose(inputs, list, timed, channels, w, h);
    }

    Leaf input_of(const Ref& value)
    {
        if (!value)
            throw std::runtime_error("the model uses a value that is not there");
        if (value->kind == Val::CONCAT)
            return compose(value->parts, value->w, value->h, value->n);
        return memory(value);
    }

    int root(int blob) const
    {
        while (model.layers[model.producer[blob]].is("Split"))
            blob = model.layers[model.producer[blob]].bottoms[0];
        return blob;
    }

    bool binary(int index, int op) const
    {
        const Layer& layer = model.layers[index];
        return layer.is("BinaryOp") && layer.bottoms.size() == 2 && layer.integer(0, 0) == op;
    }

    void convolve(int index)
    {
        const Layer& layer = model.layers[index];
        const int outputs = layer.integer(0, 0), stride = layer.integer(3, 1);
        const bool square = layer.integer(1, 0) == 3 && layer.integer(11, 3) == 3 && layer.integer(2, 1) == 1 && layer.integer(12, 1) == 1
            && stride == layer.integer(13, stride) && layer.integer(4, 0) == 1 && layer.integer(14, 1) == 1 && layer.integer(15, 1) == 1
            && layer.integer(16, 1) == 1 && layer.integer(7, 1) == 1 && (stride == 1 || stride == 2);
        const Leaf input = input_of(values[layer.bottoms[0]]);
        const int inputs = input.c;
        const int activation = layer.integer(9, 0);
        if (!square || (activation != 0 && activation != 2) || (int)layer.weight.size() != outputs * inputs * 9)
            throw std::runtime_error("the model has a convolution that is not supported");
        const std::vector<float> bias = layer.bias.empty() ? std::vector<float>(outputs, 0.f) : layer.bias;
        const float slope = activation == 2 ? layer.array(10)[0] : 0.f;

        int top = layer.tops[0];
        int fused[3] = { -1, -1, -1 };
        std::vector<float> weight = layer.weight, shifted = bias;
        bool residual = false;
        if (stride == 1 && activation == 0 && inputs == outputs && model.readers[top].size() == 1) {
            int at = model.readers[top][0];
            const float* beta = nullptr;
            bool possible = true;
            if (binary(at, MUL) && model.layers[at].bottoms[0] == top) {
                const Layer& constant = model.layers[model.producer[model.layers[at].bottoms[1]]];
                possible
                    = constant.is("MemoryData") && constant.integer(0, 0) == 1 && constant.integer(1, 1) == 1 && constant.integer(2, 1) == outputs;
                beta = possible ? constant.data.data() : nullptr;
                fused[0] = at;
                top = model.layers[at].tops[0];
                possible = possible && model.readers[top].size() == 1;
                if (possible)
                    at = model.readers[top][0];
            }
            if (possible && binary(at, ADD)) {
                const Layer& add = model.layers[at];
                const int other = add.bottoms[add.bottoms[0] == top ? 1 : 0];
                top = add.tops[0];
                if (root(other) == root(layer.bottoms[0]) && model.readers[top].size() == 1 && model.layers[model.readers[top][0]].is("ReLU")) {
                    fused[1] = at;
                    fused[2] = model.readers[top][0];
                    residual = true;
                }
            }
            if (residual) {
                for (int o = 0; o < outputs; o++) {
                    const float scale = beta ? beta[o] : 1.f;
                    shifted[o] = bias[o] * scale;
                    for (int k = 0; k < outputs * 9; k++)
                        weight[(size_t)o * outputs * 9 + k] *= scale;
                    weight[((size_t)o * outputs + o) * 9 + 4] += 1;
                }
            } else {
                fused[0] = -1;
            }
        }
        if (residual) {
            for (int index : fused) {
                if (index >= 0)
                    folded[index] = 1;
            }
            const Layer& relu = model.layers[fused[2]];
            values[relu.tops[0]] = view_of(backend.convolution(layer.name, input, weight, shifted, outputs, 1, relu.number(0, 0), true), 0, outputs);
            return;
        }
        values[layer.tops[0]] = view_of(backend.convolution(layer.name, input, weight, bias, outputs, stride, slope, activation == 2), 0, outputs);
    }

    void deconvolve(int index)
    {
        const Layer& layer = model.layers[index];
        if (!unfoldable(layer))
            throw std::runtime_error("the model has a deconvolution that is not supported");
        const int outputs = layer.integer(0, 0);
        const Leaf input = input_of(values[layer.bottoms[0]]);
        const int inputs = input.c;
        if ((int)layer.weight.size() != outputs * inputs * 16)
            throw std::runtime_error("the model has a deconvolution that is not supported");
        const int top = layer.tops[0];
        const int reader = model.readers[top].size() == 1 ? model.readers[top][0] : -1;
        const bool joined = reader >= 0 && model.layers[reader].is("PixelShuffle") && model.layers[reader].integer(0, 0) == 2
            && model.layers[reader].integer(1, 0) == 0 && outputs % 4 == 0;
        const int shift = joined ? 2 : 1;
        const auto spot = [&](int p, int py, int px) {
            if (!joined)
                return p * 4 + py * 2 + px;
            const int c = p / 4, dy = p % 4 / 2, dx = p % 2;
            return c * 16 + (py * 2 + dy) * 4 + px * 2 + dx;
        };
        std::vector<float> weight((size_t)outputs * 4 * inputs * 9, 0.f), bias(outputs * 4, 0.f);
        for (int p = 0; p < outputs; p++) {
            for (int py = 0; py < 2; py++) {
                for (int px = 0; px < 2; px++) {
                    const int o = spot(p, py, px);
                    bias[o] = layer.bias.empty() ? 0.f : layer.bias[p];
                    for (int q = 0; q < inputs; q++) {
                        for (int ty = 0; ty < 3; ty++) {
                            for (int tx = 0; tx < 3; tx++) {
                                const int ky = py + 3 - 2 * ty, kx = px + 3 - 2 * tx;
                                if (ky >= 0 && ky < 4 && kx >= 0 && kx < 4)
                                    weight[(((size_t)o * inputs + q) * 3 + ty) * 3 + tx] = layer.weight[((size_t)p * inputs + q) * 16 + ky * 4 + kx];
                            }
                        }
                    }
                }
            }
        }
        const Leaf leaf = backend.convolution(layer.name, input, weight, bias, outputs * 4, 1, 0, false);
        const int channels = outputs * 4 >> (2 * shift);
        if (joined) {
            folded[reader] = 1;
            values[model.layers[reader].tops[0]] = view_of(leaf, 0, channels, shift);
        } else {
            values[top] = view_of(leaf, 0, channels, shift);
        }
    }

    std::vector<int> uses(int blob, int n) const
    {
        std::vector<int> count(n, 0);
        const auto add = [&](const std::vector<int>& more, int from) {
            for (size_t i = 0; i < more.size(); i++)
                count[from + i] += more[i];
        };
        for (int reader : model.readers[blob]) {
            const Layer& layer = model.layers[reader];
            if (layer.is("Split")) {
                for (int top : layer.tops)
                    add(uses(top, n), 0);
            } else if (layer.is("Crop")) {
                const auto [from, to] = cropped(layer, n);
                add(uses(layer.tops[0], to - from), from);
            } else if (layer.is("Slice")) {
                const std::vector<int> ends = sliced(layer, n);
                for (size_t top = 0; top < layer.tops.size(); top++)
                    add(uses(layer.tops[top], ends[top + 1] - ends[top]), ends[top]);
            } else {
                for (int& item : count)
                    item += layer.is("rife.Warp") ? 2 : 1;
            }
        }
        return count;
    }

    void settle(const Layer& layer)
    {
        if (layer.is("Split") || layer.is("Concat") || layer.is("Convolution") || layer.is("Deconvolution"))
            return;
        for (int top : layer.tops) {
            Ref& value = values[top];
            if (!value || (value->kind != Val::LIN && value->kind != Val::WARP) || (pure(value, true) && value->terms.size() == 1))
                continue;
            if (value->w == 0 && value->terms.empty())
                continue;
            const std::vector<int> count = uses(top, value->n);
            const bool shared = std::all_of(count.begin(), count.end(), [](int item) { return item != 1; });
            if (shared && std::any_of(count.begin(), count.end(), [](int item) { return item > 1; }))
                value = stored(value);
        }
    }

    void walk(bool pair)
    {
        for (size_t index = 0; index < model.layers.size(); index++) {
            const Layer& layer = model.layers[index];
            if (folded[index] || layer.is("Input") || layer.is("MemoryData") || (pair && plan.frame[index]))
                continue;
            bool ready = true;
            for (int bottom : layer.bottoms)
                ready = ready && values[bottom];
            if (!ready) {
                if (pair)
                    throw std::runtime_error("the model uses a value that is not there");
                continue;
            }
            layer_of((int)index);
            settle(layer);
        }
    }

    void layer_of(int index)
    {
        const Layer& layer = model.layers[index];
        const auto input = [&](int at) -> const Ref& { return values[layer.bottoms[at]]; };
        if (layer.is("Split")) {
            for (int top : layer.tops)
                values[top] = input(0);
        } else if (layer.is("Crop")) {
            const auto [from, to] = cropped(layer, input(0)->n);
            values[layer.tops[0]] = crop(input(0), from, to);
        } else if (layer.is("Slice")) {
            const std::vector<int> ends = sliced(layer, input(0)->n);
            for (size_t top = 0; top < layer.tops.size(); top++)
                values[layer.tops[top]] = crop(input(0), ends[top], ends[top + 1]);
        } else if (layer.is("Concat")) {
            if (layer.integer(0, 0) != 0)
                throw std::runtime_error("the model has a concat that is not supported");
            auto value = std::make_shared<Val>();
            value->kind = Val::CONCAT;
            for (int bottom : layer.bottoms) {
                const Ref& part = values[bottom];
                const std::vector<Ref> list = part->kind == Val::CONCAT ? part->parts : std::vector<Ref> { part };
                for (const Ref& item : list) {
                    value->parts.push_back(item);
                    value->n += item->n;
                    if (item->w && value->w && (item->w != value->w || item->h != value->h))
                        throw std::runtime_error("the model joins values of different sizes");
                    if (item->w) {
                        value->w = item->w;
                        value->h = item->h;
                    }
                }
            }
            values[layer.tops[0]] = value;
        } else if (layer.is("Interp")) {
            if (layer.integer(0, 0) != 2 || layer.integer(6, 0) || layer.integer(5, 0))
                throw std::runtime_error("the model has a resize that is not supported");
            const Ref& value = input(0);
            const int w = layer.integer(4, 0) ? layer.integer(4, 0) : (int)((float)value->w * layer.number(2, 1.f));
            const int h = layer.integer(3, 0) ? layer.integer(3, 0) : (int)((float)value->h * layer.number(1, 1.f));
            values[layer.tops[0]] = resize(value, w, h);
        } else if (layer.is("BinaryOp")) {
            binary_op(layer);
        } else if (layer.is("Eltwise")) {
            const std::vector<float> factors = layer.array(1);
            if (layer.integer(0, 0) != 1)
                throw std::runtime_error("the model has an operation that is not supported");
            values[layer.tops[0]] = sum(input(0), factors.empty() ? 1.f : factors[0], input(1), factors.empty() ? 1.f : factors[1]);
        } else if (layer.is("Sigmoid")) {
            auto value = std::make_shared<Val>();
            value->kind = Val::SIGMOID;
            value->n = input(0)->n;
            value->parts = { input(0) };
            values[layer.tops[0]] = value;
        } else if (layer.is("rife.Warp")) {
            values[layer.tops[0]] = warp(input(0), input(1));
        } else if (layer.is("Convolution")) {
            convolve(index);
        } else if (layer.is("Deconvolution")) {
            deconvolve(index);
        } else {
            throw std::runtime_error("the model has a layer that is not supported: " + layer.type);
        }
    }

    void binary_op(const Layer& layer)
    {
        const int op = layer.integer(0, 0);
        const auto input = [&](int at) -> const Ref& { return values[layer.bottoms[at]]; };
        const int top = layer.tops[0];
        if (layer.integer(1, 0)) {
            const float b = layer.number(2, 0);
            const Ref& a = input(0);
            if (a->kind == Val::SIGMOID && op == RSUB && b == 1) {
                values[top] = altered(a);
                values[top]->inverted = !a->inverted;
                return;
            }
            if (op == ADD || op == SUB)
                values[top] = scaled(a, 1, op == ADD ? b : -b);
            else if (op == MUL || op == DIV)
                values[top] = scaled(a, op == MUL ? b : 1 / b, 0);
            else if (op == RSUB)
                values[top] = scaled(a, -1, b);
            else
                throw std::runtime_error("the model has an operation that is not supported");
            return;
        }
        const Ref &a = input(0), &b = input(1);
        if (op == ADD && a->kind == Val::WEIGHTED && b->kind == Val::WEIGHTED) {
            const Ref &first = a->parts[1]->inverted ? b : a, &second = a->parts[1]->inverted ? a : b;
            if (first->parts[1]->inverted == second->parts[1]->inverted || first->parts[1]->parts[0] != second->parts[1]->parts[0]
                || first->n != second->n || first->parts[1]->n != 1 || first->parts[0]->w != second->parts[0]->w
                || first->parts[0]->h != second->parts[0]->h || first->parts[1]->parts[0]->w != first->parts[0]->w
                || first->parts[1]->parts[0]->h != first->parts[0]->h)
                throw std::runtime_error("the model blends in a way that is not supported");
            auto value = std::make_shared<Val>();
            value->kind = Val::BLEND;
            value->n = first->parts[0]->n;
            value->w = first->parts[0]->w;
            value->h = first->parts[0]->h;
            // vulkan takes the sigmoid of the mask sum before it rounds to fp16
            auto mask = std::make_shared<Val>();
            mask->kind = Val::SIGMOID;
            mask->parts = first->parts[1]->parts;
            mask->n = mask->parts[0]->n;
            mask->w = mask->parts[0]->w;
            mask->h = mask->parts[0]->h;
            value->parts = { first->parts[0], second->parts[0], stored(mask) };
            values[top] = value;
        } else if (op == ADD || op == SUB) {
            values[top] = sum(a, 1, b, op == ADD ? 1.f : -1.f);
        } else if (op == MUL) {
            const bool constant = a->kind == Val::LIN && a->terms.empty() && a->time == 0;
            if (constant || (b->kind == Val::LIN && b->terms.empty() && b->time == 0)) {
                values[top] = scaled(constant ? b : a, (constant ? a : b)->bias, 0);
            } else if (a->kind == Val::WARP && b->kind == Val::SIGMOID && a->w == a->image.w[0]) {
                auto value = std::make_shared<Val>();
                value->kind = Val::WEIGHTED;
                value->n = a->n;
                value->parts = { a, b };
                values[top] = value;
            } else {
                throw std::runtime_error("the model has an operation that is not supported");
            }
        } else {
            throw std::runtime_error("the model has an operation that is not supported");
        }
    }

    Leaf frame(const Leaf& picture)
    {
        values[plan.inputs[0]] = view_of(picture, 0, 3);
        walk(false);
        const Ref& features = values[plan.cached[0]];
        if (!features)
            throw std::runtime_error("the model has no frame features");
        return memory(features);
    }

    Leaf pair(const Leaf& first, const Leaf& second)
    {
        const Leaf pictures[2] = { first, second };
        for (int role = 0; role < 2; role++) {
            values[plan.inputs[role]] = view_of(pictures[role], 0, 3);
            if (plan.cached[role] >= 0)
                values[plan.cached[role]] = view_of(pictures[role], 3, plan.features);
        }
        values[plan.inputs[2]] = std::make_shared<Val>();
        values[plan.inputs[2]]->n = 1;
        values[plan.inputs[2]]->time = 1;
        walk(true);
        const Ref& out = plan.output < 0 ? nullptr : values[plan.output];
        if (!out || out->n != 3 || out->w != first.w || out->h != first.h)
            throw std::runtime_error("the model has no output of three channels at the size of the picture");
        return input_of(out);
    }
};

}
