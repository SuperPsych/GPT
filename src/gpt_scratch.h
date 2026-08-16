#pragma once
#include "matrix.h"
#include "attention_scratch.h"
#include "ffn_scratch.h"

struct GPTScratch {
    int n = 0;
    vector<int> tokens;

    vector<vector<vector<double>>> x_pre_attn;
    vector<vector<vector<double>>> x_pre_ffn;
    vector<vector<double>> x_pre_final;
    vector<vector<vector<double>>> concat;

    vector<vector<AttentionScratch>> attn_scratch;
    vector<vector<FFNScratch>> ffn_scratch;

    Matrix dW_e, dW_u;
    vector<Matrix> dW_o;
    vector<Matrix> dW_ffn_gate, dW_ffn_up, dW_ffn_down;
    vector<vector<double>> dg_attn_norms;
    vector<vector<double>> dg_ffn_norms;
    vector<double> dg_final_norm;

    GPTScratch(int max_seq_len, int num_layers, int num_heads, int dim_head, int dim_model, int hidden_dim, int vocab_size) :
        x_pre_attn(num_layers, vector<vector<double>>(max_seq_len, vector<double>(dim_model))),
        x_pre_ffn(num_layers, vector<vector<double>>(max_seq_len, vector<double>(dim_model))),
        x_pre_final(max_seq_len, vector<double>(dim_model)),
        concat(num_layers, vector<vector<double>>(max_seq_len, vector<double>(num_heads * dim_head))),
        dW_e(vocab_size, dim_model), dW_u(vocab_size, dim_model),
        dg_attn_norms(num_layers, vector<double>(dim_model, 0.0)),
        dg_ffn_norms(num_layers, vector<double>(dim_model, 0.0)),
        dg_final_norm(dim_model, 0.0) {

        attn_scratch.reserve(num_layers);
        ffn_scratch.reserve(num_layers);
        dW_o.reserve(num_layers);
        dW_ffn_gate.reserve(num_layers);
        dW_ffn_up.reserve(num_layers);
        dW_ffn_down.reserve(num_layers);

        for (int l = 0; l < num_layers; l++) {
            vector<AttentionScratch> heads_scratch;
            heads_scratch.reserve(num_heads);
            for (int h = 0; h < num_heads; h++) {
                heads_scratch.emplace_back(max_seq_len, dim_head, dim_model);
            }
            attn_scratch.push_back(move(heads_scratch));

            vector<FFNScratch> tok_scratch;
            tok_scratch.reserve(max_seq_len);
            for (int i = 0; i < max_seq_len; i++) {
                tok_scratch.emplace_back(dim_model, hidden_dim);
            }
            ffn_scratch.push_back(move(tok_scratch));

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
