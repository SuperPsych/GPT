#pragma once
#include <cmath>
#include "matrix.h"
#include "attention_scratch.h"
#include "rope.h"

struct AttentionHead {
    Matrix W_q, W_k, W_v;
    Matrix K, V;
    AttentionHead(int dim_head, int dim_model) :
        W_q(dim_head, dim_model), W_k(dim_head, dim_model), W_v(dim_head, dim_model),
        K(0, dim_head), V(0, dim_head) {
        W_q.randomize();
        W_k.randomize();
        W_v.randomize();
    }
    vector<double> add_token(const vector<double>& x){
        Span<double> k_row = K.add_row();
        W_k.times(x, k_row);
        int t = K.num_rows - 1;
        RoPE::rotate(k_row, t);

        W_v.times(x, V.add_row());

        int dim_head = W_q.num_rows;
        double scale = 1.0 / sqrt((double)dim_head);
        vector<double> q(dim_head);
        W_q.times(x, q);
        RoPE::rotate(q, t);

        vector<double> scores(t+1);
        double max_score = -INFINITY;
        for (int j = 0; j <= t; j++) {
            double dot = 0;
            for (int d = 0; d < dim_head; d++) dot += q[d] * K.at(j,d);
            scores[j] = dot * scale;
            max_score = max(max_score, scores[j]);
        }
        double sum = 0;
        for (int j = 0; j <= t; j++) {
            scores[j] = exp(scores[j] - max_score);
            sum += scores[j];
        }
        vector<double> out(dim_head, 0.0);
        for (int j = 0; j <= t; j++) {
            double p = scores[j] / sum;
            for (int d = 0; d < dim_head; d++) out[d] += p * V.at(j,d);
        }
        return out;
    }

    vector<vector<double>> forward(const vector<vector<double>>& seq, AttentionScratch& s) const {
        int n = seq.size();
        int dim_head = W_q.num_rows;
        double scale = 1.0 / sqrt((double)dim_head);

        for (int i = 0; i < n; i++) {
            s.x[i] = seq[i];
            Span<double> q_row = s.Q.row(i);
            Span<double> k_row = s.K.row(i);
            W_q.times(seq[i], q_row);
            W_k.times(seq[i], k_row);
            W_v.times(seq[i], s.V.row(i));
            RoPE::rotate(q_row, i);
            RoPE::rotate(k_row, i);
        }

        vector<vector<double>> out(n, vector<double>(dim_head, 0.0));
        vector<double> scores(n);
        for (int i = 0; i < n; i++) {
            double max_score = -INFINITY;
            for (int j = 0; j <= i; j++) {
                double dot = 0;
                for (int d = 0; d < dim_head; d++) dot += s.Q.at(i,d) * s.K.at(j,d);
                scores[j] = dot * scale;
                max_score = max(max_score, scores[j]);
            }
            double sum = 0;
            for (int j = 0; j <= i; j++) {
                scores[j] = exp(scores[j] - max_score);
                sum += scores[j];
            }
            for (int j = 0; j <= i; j++) {
                double p = scores[j] / sum;
                s.P.at(i,j) = p;
                for (int d = 0; d < dim_head; d++) out[i][d] += p * s.V.at(j,d);
            }
        }
        return out;
    }

    vector<vector<double>> backward(const vector<vector<double>>& grad_out, AttentionScratch& s) const {
        int n = s.n;
        int dim_head = W_q.num_rows;
        int dim_model = W_q.num_cols;
        double scale = 1.0 / sqrt((double)dim_head);

        vector<vector<double>> dQ(n, vector<double>(dim_head, 0.0));
        vector<vector<double>> dK(n, vector<double>(dim_head, 0.0));
        vector<vector<double>> dV(n, vector<double>(dim_head, 0.0));

        for (int i = 0; i < n; i++) {
            vector<double> dP(i+1);
            for (int j = 0; j <= i; j++) {
                double p = s.P.at(i,j);
                double dp = 0;
                for (int d = 0; d < dim_head; d++) {
                    dV[j][d] += p * grad_out[i][d];
                    dp += grad_out[i][d] * s.V.at(j,d);
                }
                dP[j] = dp;
            }
            double weighted = 0;
            for (int j = 0; j <= i; j++) weighted += s.P.at(i,j) * dP[j];
            for (int j = 0; j <= i; j++) {
                double dscore = s.P.at(i,j) * (dP[j] - weighted) * scale;
                for (int d = 0; d < dim_head; d++) {
                    dQ[i][d] += dscore * s.K.at(j,d);
                    dK[j][d] += dscore * s.Q.at(i,d);
                }
            }
        }

        for (int i = 0; i < n; i++) {
            RoPE::rotate(dQ[i], i, -1.0);
            RoPE::rotate(dK[i], i, -1.0);
        }

        for (int i = 0; i < n; i++) {
            for (int r = 0; r < dim_head; r++) {
                for (int c = 0; c < dim_model; c++) {
                    s.dW_q.at(r,c) += dQ[i][r] * s.x[i][c];
                    s.dW_k.at(r,c) += dK[i][r] * s.x[i][c];
                    s.dW_v.at(r,c) += dV[i][r] * s.x[i][c];
                }
            }
        }

        vector<vector<double>> dx(n, vector<double>(dim_model, 0.0));
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < dim_head; r++) {
                for (int c = 0; c < dim_model; c++) {
                    dx[i][c] += W_q.at(r,c) * dQ[i][r] + W_k.at(r,c) * dK[i][r] + W_v.at(r,c) * dV[i][r];
                }
            }
        }
        return dx;
    }
};