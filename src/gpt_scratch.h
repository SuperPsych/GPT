#pragma once
#include "matrix.h"
#include "attention_scratch.h"
#include "ffn_scratch.h"

struct GPTScratch {
    int n = 0;
    vector<int> tokens;

    vector<vector<vector<float>>> x_pre_attn;
    vector<vector<vector<float>>> x_pre_ffn;
    vector<vector<float>> x_pre_final;
    vector<Matrix> concat;

    Matrix x_buf;
    Matrix normed_buf;
    Matrix proj_buf;
    Matrix normed_final_buf;
    Matrix logits_buf;
    Matrix grad_logits;

    Matrix dx_buf;
    Matrix dnormed_final_buf;
    vector<float> dattn_norm_scratch;
    vector<float> dffn_norm_scratch;
    Matrix dconcat_buf;
    Matrix dnormed_attn_buf;

    vector<vector<AttentionScratch>> attn_scratch;
    vector<FFNActivations> ffn_cache;
    FFNDelta ffn_delta;

    Matrix dW_e, dW_u;
    vector<Matrix> dW_o;
    vector<Matrix> dW_ffn_gate, dW_ffn_up, dW_ffn_down;
    vector<vector<float>> dg_attn_norms;
    vector<vector<float>> dg_ffn_norms;
    vector<float> dg_final_norm;

    GPTScratch(int max_seq_len, int num_layers, int num_heads, int dim_head, int dim_model, int hidden_dim, int vocab_size) :
        x_pre_attn(num_layers, vector<vector<float>>(max_seq_len, vector<float>(dim_model))),
        x_pre_ffn(num_layers, vector<vector<float>>(max_seq_len, vector<float>(dim_model))),
        x_pre_final(max_seq_len, vector<float>(dim_model)),
        x_buf(max_seq_len, dim_model),
        normed_buf(max_seq_len, dim_model),
        proj_buf(max_seq_len, dim_model),
        normed_final_buf(max_seq_len, dim_model),
        logits_buf(max_seq_len, vocab_size),
        grad_logits(max_seq_len, vocab_size),
        dx_buf(max_seq_len, dim_model),
        dnormed_final_buf(max_seq_len, dim_model),
        dattn_norm_scratch(dim_model),
        dffn_norm_scratch(dim_model),
        dconcat_buf(max_seq_len, num_heads * dim_head),
        dnormed_attn_buf(max_seq_len, dim_model),
        ffn_delta(max_seq_len, dim_model, hidden_dim),
        dW_e(vocab_size, dim_model), dW_u(vocab_size, dim_model),
        dg_attn_norms(num_layers, vector<float>(dim_model, 0.0f)),
        dg_ffn_norms(num_layers, vector<float>(dim_model, 0.0f)),
        dg_final_norm(dim_model, 0.0f) {

        concat.reserve(num_layers);
        attn_scratch.reserve(num_layers);
        ffn_cache.reserve(num_layers);
        dW_o.reserve(num_layers);
        dW_ffn_gate.reserve(num_layers);
        dW_ffn_up.reserve(num_layers);
        dW_ffn_down.reserve(num_layers);

        for (int l = 0; l < num_layers; l++) {
            concat.emplace_back(max_seq_len, num_heads * dim_head);

            vector<AttentionScratch> heads_scratch;
            heads_scratch.reserve(num_heads);
            for (int h = 0; h < num_heads; h++) {
                heads_scratch.emplace_back(max_seq_len, dim_head, dim_model);
            }
            attn_scratch.push_back(move(heads_scratch));

            ffn_cache.emplace_back(max_seq_len, dim_model, hidden_dim);

            dW_o.emplace_back(dim_model, num_heads * dim_head);
            dW_ffn_gate.emplace_back(hidden_dim, dim_model);
            dW_ffn_up.emplace_back(hidden_dim, dim_model);
            dW_ffn_down.emplace_back(dim_model, hidden_dim);
        }
    }

    void set_n(int new_n) {
        n = new_n;
        for (auto& layer : attn_scratch) {
            for (auto& s : layer) s.n = new_n;
        }
    }

    void clear() {
        Matrix::clear(dW_e);
        Matrix::clear(dW_u);
        for (auto& m : dW_o) Matrix::clear(m);
        for (auto& m : dW_ffn_gate) Matrix::clear(m);
        for (auto& m : dW_ffn_up) Matrix::clear(m);
        for (auto& m : dW_ffn_down) Matrix::clear(m);
        for (auto& v : dg_attn_norms) Matrix::clear(v);
        for (auto& v : dg_ffn_norms) Matrix::clear(v);
        Matrix::clear(dg_final_norm);
        for (auto& layer : attn_scratch) for (auto& s : layer) s.clear();
    }
};
