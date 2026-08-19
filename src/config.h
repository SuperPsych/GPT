#pragma once
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr int NUM_LAYERS   = 4;
constexpr int NUM_HEADS    = 4;
constexpr int DIM_HEAD     = 32;
constexpr int DIM_MODEL    = 128;
constexpr int HIDDEN_DIM   = 352;
constexpr int TARGET_VOCAB = 4096;
constexpr int MAX_SEQ_LEN  = 512;

constexpr int    EPOCHS       = 1;
constexpr int    BATCH_SIZE   = 64;
constexpr int    VAL_HOLDOUT  = 2000;
constexpr int    NUM_EXAMPLES = 500000;

constexpr int    LOG_EVERY        = 1024;
constexpr int    CHECKPOINT_EVERY = 8192;

constexpr double LR_PEAK      = 2e-3;
constexpr double LR_MIN_RATIO = 0.1;
constexpr double BETA1        = 0.9;
constexpr double BETA2        = 0.95;
constexpr double EPS          = 1e-8;
constexpr double WEIGHT_DECAY = 0.1;
constexpr double GRAD_CLIP    = 1.0;

constexpr double WARMUP_FRAC  = 0.02;
constexpr int    WARMUP_MIN   = 300;

constexpr float INIT_STD_EMBED = 0.02f;

inline float init_residual_scale(int num_layers) {
    return 1.0f / std::sqrt(2.0f * (float)num_layers);
}

inline int warmup_steps(int total_steps) {
    return std::max(WARMUP_MIN, (int)(WARMUP_FRAC * total_steps));
}

inline double lr_at(int step, int warmup, int total_steps) {
    if (step < warmup) {
        return LR_PEAK * (double)(step + 1) / (double)warmup;
    }
    double t = (double)(step - warmup) / (double)std::max(1, total_steps - warmup);
    t = std::min(1.0, std::max(0.0, t));
    double cosine = 0.5 * (1.0 + std::cos(M_PI * t));
    return LR_PEAK * (LR_MIN_RATIO + (1.0 - LR_MIN_RATIO) * cosine);
}

static_assert(NUM_HEADS * DIM_HEAD == DIM_MODEL, "W_o must stay square");
static_assert(HIDDEN_DIM % 32 == 0, "keep HIDDEN_DIM SIMD-friendly");
