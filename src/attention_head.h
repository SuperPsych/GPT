#pragma once
#include <cmath>
#include <limits>
#include "matrix.h"
#include "attention_scratch.h"
#include "rope.h"

struct AttentionHead {
    Matrix W_q, W_k, W_v;
    Matrix K, V;
    AttentionHead(int dim_head, int dim_model, int max_seq_len) :
        W_q(dim_head, dim_model), W_k(dim_head, dim_model), W_v(dim_head, dim_model),
        K(0, dim_head), V(0, dim_head) {
        W_q.randomize_xavier();
        W_k.randomize_xavier();
        W_v.randomize_xavier();
        K.reserve_rows(max_seq_len);
        V.reserve_rows(max_seq_len);
    }
    vector<float> add_token(const vector<float>& x){
        span<float> k_row = K.add_row();
        W_k.times(x, k_row);
        int t = K.num_rows - 1;
        RoPE::rotate(k_row, t);

        W_v.times(x, V.add_row());

        int dim_head = W_q.num_rows;
        float scale = (float)(1.0 / sqrt((double)dim_head));
        vector<float> q(dim_head);
        W_q.times(x, q);
        RoPE::rotate(q, t);

        vector<float> scores(t+1);
        float max_score = numeric_limits<float>::lowest();
        for (int j = 0; j <= t; j++) {
            float dot = 0;
            for (int d = 0; d < dim_head; d++) dot += q[d] * K.at(j,d);
            scores[j] = dot * scale;
            max_score = max(max_score, scores[j]);
        }
        float sum = 0;
        for (int j = 0; j <= t; j++) {
            scores[j] = exp(scores[j] - max_score);
            sum += scores[j];
        }
        vector<float> out(dim_head, 0.0f);
        for (int j = 0; j <= t; j++) {
            float p = scores[j] / sum;
            for (int d = 0; d < dim_head; d++) out[d] += p * V.at(j,d);
        }
        return out;
    }

    Matrix& forward(const Matrix& seq, AttentionScratch& s) const {
        int n = s.n;
        int dim_head = W_q.num_rows;
        int dim_model = W_q.num_cols;
        float scale = (float)(1.0 / sqrt((double)dim_head));

        copy(seq.data.begin(), seq.data.begin() + (size_t)n * dim_model, s.x.data.begin());
        W_q.times(seq, s.Q, n);
        W_k.times(seq, s.K, n);
        W_v.times(seq, s.V, n);

        for (int i = 0; i < n; i++) {
            span<float> q_row = s.Q.row(i);
            span<float> k_row = s.K.row(i);
            RoPE::rotate(q_row, i);
            RoPE::rotate(k_row, i);
        }

        for (int i = 0; i < n; i++) {
            float max_score = numeric_limits<float>::lowest();
            for (int j = 0; j <= i; j++) {
                float dot = 0;
                for (int d = 0; d < dim_head; d++) dot += s.Q.at(i,d) * s.K.at(j,d);
                s.scores[j] = dot * scale;
                max_score = max(max_score, s.scores[j]);
            }
            float sum = 0;
            for (int j = 0; j <= i; j++) {
                s.scores[j] = exp(s.scores[j] - max_score);
                sum += s.scores[j];
            }
            for (int d = 0; d < dim_head; d++) s.out.at(i,d) = 0.0f;
            for (int j = 0; j <= i; j++) {
                float p = s.scores[j] / sum;
                s.P.at(i,j) = p;
                for (int d = 0; d < dim_head; d++) s.out.at(i,d) += p * s.V.at(j,d);
            }
        }
        return s.out;
    }

    Matrix& backward(const Matrix& grad_out, AttentionScratch& s) const {
        int n = s.n;
        int dim_head = W_q.num_rows;
        int dim_model = W_q.num_cols;
        float scale = (float)(1.0 / sqrt((double)dim_head));

        fill(s.dQ.data.begin(), s.dQ.data.begin() + (size_t)n * dim_head, 0.0f);
        fill(s.dK.data.begin(), s.dK.data.begin() + (size_t)n * dim_head, 0.0f);
        fill(s.dV.data.begin(), s.dV.data.begin() + (size_t)n * dim_head, 0.0f);

        thread_local vector<float> grad_out_row, q_row, dq_acc;
        if ((int)grad_out_row.size() < dim_head) {
            grad_out_row.resize(dim_head);
            q_row.resize(dim_head);
            dq_acc.resize(dim_head);
        }

        for (int i = 0; i < n; i++) {
            const float* grad_out_i = &grad_out.data[(size_t)i * grad_out.num_cols];
            copy(grad_out_i, grad_out_i + dim_head, grad_out_row.begin());
            const float* q_i = &s.Q.data[(size_t)i * s.Q.num_cols];
            copy(q_i, q_i + dim_head, q_row.begin());

            for (int j = 0; j <= i; j++) {
                float p = s.P.at(i,j);
                float dp = 0;
                const float* v_j = &s.V.data[(size_t)j * s.V.num_cols];
                float* dv_j = &s.dV.data[(size_t)j * s.dV.num_cols];
                for (int d = 0; d < dim_head; d++) {
                    dv_j[d] += p * grad_out_row[d];
                    dp += grad_out_row[d] * v_j[d];
                }
                s.dP[j] = dp;
            }
            float weighted = 0;
            for (int j = 0; j <= i; j++) weighted += s.P.at(i,j) * s.dP[j];

            fill(dq_acc.begin(), dq_acc.begin() + dim_head, 0.0f);
            for (int j = 0; j <= i; j++) {
                float dscore = s.P.at(i,j) * (s.dP[j] - weighted) * scale;
                const float* k_j = &s.K.data[(size_t)j * s.K.num_cols];
                float* dk_j = &s.dK.data[(size_t)j * s.dK.num_cols];
                for (int d = 0; d < dim_head; d++) {
                    dq_acc[d] += dscore * k_j[d];
                    dk_j[d] += dscore * q_row[d];
                }
            }
            float* dq_i = &s.dQ.data[(size_t)i * s.dQ.num_cols];
            for (int d = 0; d < dim_head; d++) dq_i[d] += dq_acc[d];
        }

        for (int i = 0; i < n; i++) {
            span<float> dq_row = s.dQ.row(i);
            RoPE::rotate(dq_row, i, -1.0);
            span<float> dk_row = s.dK.row(i);
            RoPE::rotate(dk_row, i, -1.0);
        }

        Matrix::accumulate_weight_grad(s.dQ, s.x, s.dW_q, n);
        Matrix::accumulate_weight_grad(s.dK, s.x, s.dW_k, n);
        Matrix::accumulate_weight_grad(s.dV, s.x, s.dW_v, n);

        W_q.backward_input(s.dQ, s.dx_out, n, false);
        W_k.backward_input(s.dK, s.dx_out, n, true);
        W_v.backward_input(s.dV, s.dx_out, n, true);
        return s.dx_out;
    }
};
