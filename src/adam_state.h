#pragma once
#include "matrix.h"

struct AdamState {
    Matrix m_W_e, v_W_e, m_W_u, v_W_u;
    vector<Matrix> m_W_o, v_W_o;
    vector<Matrix> m_ffn_gate, v_ffn_gate, m_ffn_up, v_ffn_up, m_ffn_down, v_ffn_down;
    vector<vector<double>> m_attn_norms, v_attn_norms, m_ffn_norms, v_ffn_norms;
    vector<double> m_final_norm, v_final_norm;
    vector<vector<Matrix>> m_W_q, v_W_q, m_W_k, v_W_k, m_W_v, v_W_v;
    int t = 0;

    AdamState(int num_layers, int num_heads, int dim_head, int dim_model, int hidden_dim, int vocab_size) :
        m_W_e(vocab_size, dim_model), v_W_e(vocab_size, dim_model),
        m_W_u(vocab_size, dim_model), v_W_u(vocab_size, dim_model),
        m_attn_norms(num_layers, vector<double>(dim_model, 0.0)),
        v_attn_norms(num_layers, vector<double>(dim_model, 0.0)),
        m_ffn_norms(num_layers, vector<double>(dim_model, 0.0)),
        v_ffn_norms(num_layers, vector<double>(dim_model, 0.0)),
        m_final_norm(dim_model, 0.0), v_final_norm(dim_model, 0.0) {

        for (int l = 0; l < num_layers; l++) {
            m_W_o.emplace_back(dim_model, num_heads * dim_head);
            v_W_o.emplace_back(dim_model, num_heads * dim_head);
            m_ffn_gate.emplace_back(hidden_dim, dim_model);
            v_ffn_gate.emplace_back(hidden_dim, dim_model);
            m_ffn_up.emplace_back(hidden_dim, dim_model);
            v_ffn_up.emplace_back(hidden_dim, dim_model);
            m_ffn_down.emplace_back(dim_model, hidden_dim);
            v_ffn_down.emplace_back(dim_model, hidden_dim);

            vector<Matrix> mq, vq, mk, vk, mv, vv;
            for (int h = 0; h < num_heads; h++) {
                mq.emplace_back(dim_head, dim_model); vq.emplace_back(dim_head, dim_model);
                mk.emplace_back(dim_head, dim_model); vk.emplace_back(dim_head, dim_model);
                mv.emplace_back(dim_head, dim_model); vv.emplace_back(dim_head, dim_model);
            }
            m_W_q.push_back(move(mq)); v_W_q.push_back(move(vq));
            m_W_k.push_back(move(mk)); v_W_k.push_back(move(vk));
            m_W_v.push_back(move(mv)); v_W_v.push_back(move(vv));
        }
    }
};
