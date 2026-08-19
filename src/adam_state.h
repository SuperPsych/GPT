#pragma once
#include "matrix.h"

struct AdamState {
    Matrix m_W_e, v_W_e;
    vector<Matrix> m_W_o, v_W_o;
    vector<Matrix> m_ffn_gate, v_ffn_gate, m_ffn_up, v_ffn_up, m_ffn_down, v_ffn_down;
    vector<vector<float>> m_attn_norms, v_attn_norms, m_ffn_norms, v_ffn_norms;
    vector<float> m_final_norm, v_final_norm;
    vector<vector<Matrix>> m_W_q, v_W_q, m_W_k, v_W_k, m_W_v, v_W_v;
    int t = 0;

    AdamState(int num_layers, int num_heads, int dim_head, int dim_model, int hidden_dim, int vocab_size) :
        m_W_e(vocab_size, dim_model), v_W_e(vocab_size, dim_model),
        m_attn_norms(num_layers, vector<float>(dim_model, 0.0f)),
        v_attn_norms(num_layers, vector<float>(dim_model, 0.0f)),
        m_ffn_norms(num_layers, vector<float>(dim_model, 0.0f)),
        v_ffn_norms(num_layers, vector<float>(dim_model, 0.0f)),
        m_final_norm(dim_model, 0.0f), v_final_norm(dim_model, 0.0f) {

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

    void save(ostream& out, int num_layers, int num_heads) const {
        out.write((const char*)&t, sizeof(t));
        m_W_e.save_bin(out); v_W_e.save_bin(out);
        for (int l = 0; l < num_layers; l++) {
            m_W_o[l].save_bin(out); v_W_o[l].save_bin(out);
            m_ffn_gate[l].save_bin(out); v_ffn_gate[l].save_bin(out);
            m_ffn_up[l].save_bin(out); v_ffn_up[l].save_bin(out);
            m_ffn_down[l].save_bin(out); v_ffn_down[l].save_bin(out);
            Matrix::save_vec_bin(out, m_attn_norms[l]); Matrix::save_vec_bin(out, v_attn_norms[l]);
            Matrix::save_vec_bin(out, m_ffn_norms[l]); Matrix::save_vec_bin(out, v_ffn_norms[l]);
            for (int h = 0; h < num_heads; h++) {
                m_W_q[l][h].save_bin(out); v_W_q[l][h].save_bin(out);
                m_W_k[l][h].save_bin(out); v_W_k[l][h].save_bin(out);
                m_W_v[l][h].save_bin(out); v_W_v[l][h].save_bin(out);
            }
        }
        Matrix::save_vec_bin(out, m_final_norm); Matrix::save_vec_bin(out, v_final_norm);
    }

    void load(istream& in, int num_layers, int num_heads) {
        in.read((char*)&t, sizeof(t));
        m_W_e.load_bin(in); v_W_e.load_bin(in);
        for (int l = 0; l < num_layers; l++) {
            m_W_o[l].load_bin(in); v_W_o[l].load_bin(in);
            m_ffn_gate[l].load_bin(in); v_ffn_gate[l].load_bin(in);
            m_ffn_up[l].load_bin(in); v_ffn_up[l].load_bin(in);
            m_ffn_down[l].load_bin(in); v_ffn_down[l].load_bin(in);
            Matrix::load_vec_bin(in, m_attn_norms[l]); Matrix::load_vec_bin(in, v_attn_norms[l]);
            Matrix::load_vec_bin(in, m_ffn_norms[l]); Matrix::load_vec_bin(in, v_ffn_norms[l]);
            for (int h = 0; h < num_heads; h++) {
                m_W_q[l][h].load_bin(in); v_W_q[l][h].load_bin(in);
                m_W_k[l][h].load_bin(in); v_W_k[l][h].load_bin(in);
                m_W_v[l][h].load_bin(in); v_W_v[l][h].load_bin(in);
            }
        }
        Matrix::load_vec_bin(in, m_final_norm); Matrix::load_vec_bin(in, v_final_norm);
    }
};
