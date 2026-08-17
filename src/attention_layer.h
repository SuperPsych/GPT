#pragma once
#include "attention_head.h"

struct AttentionLayer {
    vector<AttentionHead> heads;

    AttentionLayer(int num_heads, int dim_head, int dim_model, int max_seq_len) {
        heads.reserve(num_heads);
        for (int h = 0; h < num_heads; h++) heads.emplace_back(dim_head, dim_model, max_seq_len);
    }
};
