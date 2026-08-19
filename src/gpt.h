#pragma once
#include <omp.h>
#include <cstring>
#include "matrix.h"
#include "attention_head.h"
#include "attention_layer.h"
#include "attention_scratch.h"
#include "adam_state.h"
#include "ffn.h"
#include "rmsnorm.h"
#include "gpt_scratch.h"
#include "tokenizer.h"
#include "chat_dataset.h"
#include "config.h"

struct GPT{
    int num_layers, num_heads, dim_head, dim_model, hidden_dim, vocab_size, max_seq_len;
    Matrix W_e;
    vector<AttentionLayer> attention_layers;
    vector<Matrix> W_o_layers;
    vector<RMSNorm> attn_norms;
    vector<RMSNorm> ffn_norms;
    vector<FFN> ffn_layers;
    RMSNorm final_norm;
    Tokenizer tokenizer;

    static constexpr uint32_t CHECKPOINT_MAGIC = 0xDEADBEEFu;

    GPT(int num_layers, int num_heads, int dim_head, int dim_model, int hidden_dim, int vocab_size, int max_seq_len) :
        num_layers(num_layers), num_heads(num_heads), dim_head(dim_head), dim_model(dim_model), hidden_dim(hidden_dim),
        vocab_size(vocab_size), max_seq_len(max_seq_len),
        W_e(vocab_size, dim_model),
        final_norm(dim_model) {
        W_e.randomize(INIT_STD_EMBED);
        RoPE::precompute(dim_head, max_seq_len);

        attention_layers.reserve(num_layers);
        W_o_layers.reserve(num_layers);
        ffn_layers.reserve(num_layers);
        attn_norms.reserve(num_layers);
        ffn_norms.reserve(num_layers);
        for (int l = 0; l < num_layers; l++) {
            attention_layers.emplace_back(num_heads, dim_head, dim_model, max_seq_len);

            W_o_layers.emplace_back(dim_model, num_heads * dim_head);
            float wo_std = (float)sqrt(1.0 / (num_heads * dim_head)) * init_residual_scale(num_layers);
            W_o_layers.back().randomize(wo_std);

            ffn_layers.emplace_back(dim_model, hidden_dim, num_layers);
            attn_norms.emplace_back(dim_model);
            ffn_norms.emplace_back(dim_model);
        }
    }

    vector<vector<float>> forward(const vector<int>& tokens) {
        int n = tokens.size();
        vector<vector<float>> x(n, vector<float>(dim_model));
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim_model; d++) {
                x[i][d] = W_e.at(tokens[i], d);
            }
        }

        for (int l = 0; l < num_layers; l++) {
            int num_heads = attention_layers[l].heads.size();

            Matrix normed(n, dim_model);
            for (int i = 0; i < n; i++) {
                vector<float> normed_i = attn_norms[l].forward(x[i]);
                for (int d = 0; d < dim_model; d++) normed.at(i, d) = normed_i[d];
            }

            vector<vector<float>> concat(n, vector<float>(num_heads * dim_head));
            for (int h = 0; h < num_heads; h++) {
                AttentionScratch scratch(n, dim_head, dim_model);
                Matrix& head_out = attention_layers[l].heads[h].forward(normed, scratch);
                for (int i = 0; i < n; i++) {
                    for (int d = 0; d < dim_head; d++) {
                        concat[i][h * dim_head + d] = head_out.at(i, d);
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                vector<float> proj = W_o_layers[l].times(concat[i]);
                for (int d = 0; d < dim_model; d++) x[i][d] += proj[d];
            }

            for (int i = 0; i < n; i++) {
                vector<float> normed_ffn = ffn_norms[l].forward(x[i]);
                vector<float> ffn_out = ffn_layers[l].forward(normed_ffn);
                for (int d = 0; d < dim_model; d++) x[i][d] += ffn_out[d];
            }
        }

        vector<vector<float>> logits(n, vector<float>(vocab_size));
        for (int i = 0; i < n; i++) {
            vector<float> normed_final = final_norm.forward(x[i]);
            logits[i] = W_e.times(normed_final);
        }
        return logits;
    }

    void forward_train(const vector<int>& tokens, GPTScratch& s) {
        int n = tokens.size();
        s.tokens = tokens;
        s.set_n(n);
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim_model; d++) {
                s.x_buf.at(i, d) = W_e.at(tokens[i], d);
            }
        }

        for (int l = 0; l < num_layers; l++) {
            int num_heads = attention_layers[l].heads.size();

            for (int i = 0; i < n; i++) {
                span<float> row = s.x_buf.row(i);
                copy(row.begin(), row.end(), s.x_pre_attn[l][i].begin());
            }

            for (int i = 0; i < n; i++) {
                span<float> x_row = s.x_buf.row(i);
                span<float> normed_row = s.normed_buf.row(i);
                attn_norms[l].forward(x_row, normed_row);
            }

            for (int h = 0; h < num_heads; h++) {
                Matrix& head_out = attention_layers[l].heads[h].forward(s.normed_buf, s.attn_scratch[l][h]);
                for (int i = 0; i < n; i++) {
                    for (int d = 0; d < dim_head; d++) {
                        s.concat[l].at(i, h * dim_head + d) = head_out.at(i, d);
                    }
                }
            }

            W_o_layers[l].times(s.concat[l], s.proj_buf, n);
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < dim_model; d++) s.x_buf.at(i, d) += s.proj_buf.at(i, d);
            }

            for (int i = 0; i < n; i++) {
                span<float> row = s.x_buf.row(i);
                copy(row.begin(), row.end(), s.x_pre_ffn[l][i].begin());
            }

            for (int i = 0; i < n; i++) {
                span<float> x_row = s.x_buf.row(i);
                span<float> normed_row = s.normed_buf.row(i);
                ffn_norms[l].forward(x_row, normed_row);
            }
            Matrix& ffn_out = ffn_layers[l].forward_train(s.normed_buf, n, s.ffn_cache[l]);
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < dim_model; d++) s.x_buf.at(i, d) += ffn_out.at(i, d);
            }
        }

        for (int i = 0; i < n; i++) {
            span<float> row = s.x_buf.row(i);
            copy(row.begin(), row.end(), s.x_pre_final[i].begin());
        }

        for (int i = 0; i < n; i++) {
            span<float> x_row = s.x_buf.row(i);
            span<float> normed_row = s.normed_final_buf.row(i);
            final_norm.forward(x_row, normed_row);
        }
    }

    void backward_body(GPTScratch& s) {
        int n = s.n;

        for (int i = 0; i < n; i++) {
            span<float> dnormed_row = s.dnormed_final_buf.row(i);
            span<float> dx_row = s.dx_buf.row(i);
            final_norm.backward(s.x_pre_final[i], dnormed_row, s.dg_final_norm, dx_row);
        }

        for (int l = num_layers - 1; l >= 0; l--) {
            int num_heads = attention_layers[l].heads.size();

            s.ffn_delta.clear();
            Matrix& d_ffn = ffn_layers[l].backward(s.dx_buf, n, s.ffn_cache[l], s.ffn_delta);
            Matrix::add(s.dW_ffn_gate[l], s.ffn_delta.W_gate_delta);
            Matrix::add(s.dW_ffn_up[l], s.ffn_delta.W_up_delta);
            Matrix::add(s.dW_ffn_down[l], s.ffn_delta.W_down_delta);

            for (int i = 0; i < n; i++) {
                span<float> d_ffn_row = d_ffn.row(i);
                ffn_norms[l].backward(s.x_pre_ffn[l][i], d_ffn_row, s.dg_ffn_norms[l], s.dffn_norm_scratch);
                for (int d = 0; d < dim_model; d++) s.dx_buf.at(i, d) += s.dffn_norm_scratch[d];
            }

            Matrix::accumulate_weight_grad(s.dx_buf, s.concat[l], s.dW_o[l], n);
            W_o_layers[l].backward_input(s.dx_buf, s.dconcat_buf, n, false);

            fill(s.dnormed_attn_buf.data.begin(), s.dnormed_attn_buf.data.begin() + (size_t)n * dim_model, 0.0f);
            for (int h = 0; h < num_heads; h++) {
                AttentionScratch& hs = s.attn_scratch[l][h];
                for (int i = 0; i < n; i++) {
                    for (int d = 0; d < dim_head; d++) {
                        hs.grad_out_buf.at(i,d) = s.dconcat_buf.at(i, h * dim_head + d);
                    }
                }
                Matrix& dnormed_h = attention_layers[l].heads[h].backward(hs.grad_out_buf, hs);
                for (int i = 0; i < n; i++) {
                    for (int d = 0; d < dim_model; d++) {
                        s.dnormed_attn_buf.at(i, d) += dnormed_h.at(i,d);
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                span<float> dnormed_attn_row = s.dnormed_attn_buf.row(i);
                attn_norms[l].backward(s.x_pre_attn[l][i], dnormed_attn_row, s.dg_attn_norms[l], s.dattn_norm_scratch);
                for (int d = 0; d < dim_model; d++) s.dx_buf.at(i, d) += s.dattn_norm_scratch[d];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim_model; d++) {
                s.dW_e.at(s.tokens[i], d) += s.dx_buf.at(i, d);
            }
        }
    }

    void reset_cache() {
        for (auto& layer : attention_layers) {
            for (auto& head : layer.heads) {
                head.K.clear_rows();
                head.V.clear_rows();
            }
        }
    }

    vector<float> generate_step(int token) {
        vector<float> x(dim_model);
        for (int d = 0; d < dim_model; d++) {
            x[d] = W_e.at(token, d);
        }

        for (int l = 0; l < num_layers; l++) {
            int num_heads = attention_layers[l].heads.size();
            vector<float> normed = attn_norms[l].forward(x);

            vector<float> concat(num_heads * dim_head);
            for (int h = 0; h < num_heads; h++) {
                auto head_out = attention_layers[l].heads[h].add_token(normed);
                for (int d = 0; d < dim_head; d++) {
                    concat[h * dim_head + d] = head_out[d];
                }
            }

            vector<float> proj = W_o_layers[l].times(concat);
            for (int d = 0; d < dim_model; d++) x[d] += proj[d];

            vector<float> normed_ffn = ffn_norms[l].forward(x);
            vector<float> ffn_out = ffn_layers[l].forward(normed_ffn);
            for (int d = 0; d < dim_model; d++) x[d] += ffn_out[d];
        }

        vector<float> normed_final = final_norm.forward(x);
        return W_e.times(normed_final);
    }

    void train_tokenizer(const string& corpus_path, int target_vocab_size, int min_frequency,
                          const string& vocab_out_path, const string& merges_out_path) {
        tokenizer.train(corpus_path, target_vocab_size, min_frequency, vocab_out_path, merges_out_path);
    }

    void load_tokenizer(const string& vocab_path, const string& merges_path) {
        tokenizer.load(vocab_path, merges_path);
    }

    vector<int> tokenize(const string& text) const {
        return tokenizer.encode(text);
    }

    string detokenize(const vector<int>& tokens) const {
        return tokenizer.decode(tokens);
    }

    void tokenize_corpus_to_file(const string& corpus_path, const string& out_path) const {
        ifstream f(corpus_path, ios::binary | ios::ate);
        streamsize size = f.tellg();
        f.seekg(0, ios::beg);
        string text(size, '\0');
        f.read(&text[0], size);
        Tokenizer::save_tokens(out_path, tokenizer.encode(text));
    }

    void apply_gradients(GPTScratch& s, AdamState& opt, double lr, double grad_scale = 1.0,
                          double beta1 = BETA1, double beta2 = BETA2, double eps = EPS, double weight_decay = WEIGHT_DECAY) {
        opt.t++;
        int t = opt.t;
        float lr_f = (float)lr, b1_f = (float)beta1, b2_f = (float)beta2, eps_f = (float)eps;
        float wd_f = (float)weight_decay, scale_f = (float)grad_scale;
        Matrix::adamw_update(W_e, s.dW_e, opt.m_W_e, opt.v_W_e, lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
        for (int l = 0; l < num_layers; l++) {
            Matrix::adamw_update(W_o_layers[l], s.dW_o[l], opt.m_W_o[l], opt.v_W_o[l], lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
            Matrix::adamw_update(ffn_layers[l].W_gate, s.dW_ffn_gate[l], opt.m_ffn_gate[l], opt.v_ffn_gate[l], lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
            Matrix::adamw_update(ffn_layers[l].W_up, s.dW_ffn_up[l], opt.m_ffn_up[l], opt.v_ffn_up[l], lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
            Matrix::adamw_update(ffn_layers[l].W_down, s.dW_ffn_down[l], opt.m_ffn_down[l], opt.v_ffn_down[l], lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
            Matrix::adamw_update(attn_norms[l].g, s.dg_attn_norms[l], opt.m_attn_norms[l], opt.v_attn_norms[l], lr_f, b1_f, b2_f, eps_f, 0.0f, t, scale_f);
            Matrix::adamw_update(ffn_norms[l].g, s.dg_ffn_norms[l], opt.m_ffn_norms[l], opt.v_ffn_norms[l], lr_f, b1_f, b2_f, eps_f, 0.0f, t, scale_f);
            for (int h = 0; h < num_heads; h++) {
                Matrix::adamw_update(attention_layers[l].heads[h].W_q, s.attn_scratch[l][h].dW_q, opt.m_W_q[l][h], opt.v_W_q[l][h], lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
                Matrix::adamw_update(attention_layers[l].heads[h].W_k, s.attn_scratch[l][h].dW_k, opt.m_W_k[l][h], opt.v_W_k[l][h], lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
                Matrix::adamw_update(attention_layers[l].heads[h].W_v, s.attn_scratch[l][h].dW_v, opt.m_W_v[l][h], opt.v_W_v[l][h], lr_f, b1_f, b2_f, eps_f, wd_f, t, scale_f);
            }
        }
        Matrix::adamw_update(final_norm.g, s.dg_final_norm, opt.m_final_norm, opt.v_final_norm, lr_f, b1_f, b2_f, eps_f, 0.0f, t, scale_f);
    }

    pair<double,int> accumulate_gradients(const ChatExample& example, GPTScratch& scratch) {
        int n = (int)example.tokens.size();
        forward_train(example.tokens, scratch);

        int m = 0;
        for (int i = 0; i + 1 < n; i++) {
            if (!example.mask[i + 1]) continue;
            scratch.scored_idx[m] = i;
            const float* src = &scratch.normed_final_buf.data[(size_t)i * dim_model];
            float* dst = &scratch.normed_scored.data[(size_t)m * dim_model];
            copy(src, src + dim_model, dst);
            m++;
        }
        if (m == 0) return {0.0, 0};

        W_e.times(scratch.normed_scored, scratch.logits_scored, m);

        double loss = 0.0;
        for (int r = 0; r < m; r++) {
            span<float> probs = scratch.logits_scored.row(r);
            FFN::output_activation(probs);
            int i = scratch.scored_idx[r];
            int target = example.tokens[i + 1];
            loss -= log(max(probs[target], 1e-12f));
            for (int v = 0; v < vocab_size; v++) {
                scratch.grad_logits_scored.at(r, v) = probs[v] - (v == target ? 1.0f : 0.0f);
            }
        }

        Matrix::accumulate_weight_grad(scratch.grad_logits_scored, scratch.normed_scored, scratch.dW_e, m);
        W_e.backward_input(scratch.grad_logits_scored, scratch.dnormed_scored, m, false);

        fill(scratch.dnormed_final_buf.data.begin(), scratch.dnormed_final_buf.data.begin() + (size_t)n * dim_model, 0.0f);
        for (int r = 0; r < m; r++) {
            int i = scratch.scored_idx[r];
            const float* src = &scratch.dnormed_scored.data[(size_t)r * dim_model];
            float* dst = &scratch.dnormed_final_buf.data[(size_t)i * dim_model];
            copy(src, src + dim_model, dst);
        }

        backward_body(scratch);
        return {loss, m};
    }

    double train_step(const ChatExample& example, GPTScratch& scratch, AdamState& opt, double lr) {
        scratch.clear();
        auto [sum_loss, target_count] = accumulate_gradients(example, scratch);
        if (target_count == 0) return 0.0;
        apply_gradients(scratch, opt, lr, 1.0 / target_count);
        return sum_loss / target_count;
    }

    static void add_scratch(GPTScratch& into, const GPTScratch& from, int num_layers, int num_heads) {
        Matrix::add(into.dW_e, from.dW_e);
        for (int l = 0; l < num_layers; l++) {
            Matrix::add(into.dW_o[l], from.dW_o[l]);
            Matrix::add(into.dW_ffn_gate[l], from.dW_ffn_gate[l]);
            Matrix::add(into.dW_ffn_up[l], from.dW_ffn_up[l]);
            Matrix::add(into.dW_ffn_down[l], from.dW_ffn_down[l]);
            Matrix::add(into.dg_attn_norms[l], from.dg_attn_norms[l]);
            Matrix::add(into.dg_ffn_norms[l], from.dg_ffn_norms[l]);
            for (int h = 0; h < num_heads; h++) {
                Matrix::add(into.attn_scratch[l][h].dW_q, from.attn_scratch[l][h].dW_q);
                Matrix::add(into.attn_scratch[l][h].dW_k, from.attn_scratch[l][h].dW_k);
                Matrix::add(into.attn_scratch[l][h].dW_v, from.attn_scratch[l][h].dW_v);
            }
        }
        Matrix::add(into.dg_final_norm, from.dg_final_norm);
    }

    static void grad_stats(const GPTScratch& s, int num_layers, int num_heads, double& sq_norm_out, bool& finite_out) {
        double sq_norm = 0.0;
        bool finite = true;
        auto visit = [&](const vector<float>& v) {
            for (float x : v) {
                uint32_t bits;
                memcpy(&bits, &x, sizeof(bits));
                if (((bits >> 23) & 0xFF) == 0xFF) finite = false;
                sq_norm += (double)x * x;
            }
        };
        visit(s.dW_e.data);
        for (int l = 0; l < num_layers; l++) {
            visit(s.dW_o[l].data);
            visit(s.dW_ffn_gate[l].data);
            visit(s.dW_ffn_up[l].data);
            visit(s.dW_ffn_down[l].data);
            visit(s.dg_attn_norms[l]);
            visit(s.dg_ffn_norms[l]);
            for (int h = 0; h < num_heads; h++) {
                visit(s.attn_scratch[l][h].dW_q.data);
                visit(s.attn_scratch[l][h].dW_k.data);
                visit(s.attn_scratch[l][h].dW_v.data);
            }
        }
        visit(s.dg_final_norm);
        sq_norm_out = sq_norm;
        finite_out = finite;
    }

    void train(vector<ChatExample>& examples, int epochs, int batch_size,
               const string& checkpoint_path = "", int checkpoint_every = 0, int log_every = 50,
               AdamState* resume_opt = nullptr) {
        omp_set_dynamic(0);
        int thread_num = omp_get_max_threads();
        cout << "Beginning GPT training on " << examples.size() << " examples (batch_size=" << batch_size
             << ", threads=" << thread_num << ")..." << endl;

        AdamState fresh_opt(num_layers, num_heads, dim_head, dim_model, hidden_dim, vocab_size);
        AdamState& opt = resume_opt ? *resume_opt : fresh_opt;

        vector<GPTScratch> thread_data;
        thread_data.reserve(thread_num);
        for (int t = 0; t < thread_num; t++) {
            thread_data.emplace_back(max_seq_len, num_layers, num_heads, dim_head, dim_model, hidden_dim, vocab_size);
        }

        mt19937 shuffle_rng(Matrix::rng()());

        int total_steps = max(1, ((int)examples.size() / batch_size) * epochs);
        int warmup = warmup_steps(total_steps);

        int step = 0, last_log_step = 0, last_checkpoint_step = 0, batch_num = 0;
        int epoch = 0;
        for (; epoch < epochs; epoch++) {
            vector<int> order(examples.size());
            iota(order.begin(), order.end(), 0);
            sort(order.begin(), order.end(), [&](int a, int b) {
                return examples[a].tokens.size() < examples[b].tokens.size();
            });
            vector<vector<int>> buckets;
            buckets.reserve((order.size() + batch_size - 1) / batch_size);
            for (size_t i = 0; i < order.size(); i += batch_size) {
                size_t end = min(i + (size_t)batch_size, order.size());
                vector<int> bucket(order.begin() + i, order.begin() + end);
                shuffle(bucket.begin(), bucket.end(), shuffle_rng);
                buckets.push_back(move(bucket));
            }
            shuffle(buckets.begin(), buckets.end(), shuffle_rng);
            vector<int> epoch_order;
            epoch_order.reserve(examples.size());
            for (auto& b : buckets) for (int idx : b) epoch_order.push_back(idx);

            double epoch_loss = 0.0;
            long long epoch_tokens = 0;

            for (int start = 0; start < (int)epoch_order.size(); start += batch_size) {
                int end = min(start + batch_size, (int)epoch_order.size());
                int this_batch = end - start;

                for (GPTScratch& td : thread_data) td.clear();

                vector<double> batch_losses(this_batch, 0.0);
                vector<int> batch_tokens(this_batch, 0);
                vector<char> batch_valid(this_batch, 0);

                #pragma omp parallel for schedule(dynamic, 1)
                for (int i = 0; i < this_batch; i++) {
                    auto& ex = examples[epoch_order[start + i]];
                    if (ex.tokens.size() < 2) continue;
                    GPTScratch& td = thread_data[omp_get_thread_num()];
                    auto [sum_loss, target_count] = accumulate_gradients(ex, td);
                    batch_losses[i] = sum_loss;
                    batch_tokens[i] = target_count;
                    batch_valid[i] = target_count > 0 ? 1 : 0;
                }

                for (int t = 1; t < thread_num; t++) {
                    add_scratch(thread_data[0], thread_data[t], num_layers, num_heads);
                }

                double batch_loss_sum = 0.0;
                long long batch_token_total = 0;
                for (int i = 0; i < this_batch; i++) {
                    if (batch_valid[i]) { batch_loss_sum += batch_losses[i]; batch_token_total += batch_tokens[i]; }
                }

                if (batch_token_total > 0) {
                    double sq_norm; bool finite;
                    grad_stats(thread_data[0], num_layers, num_heads, sq_norm, finite);

                    if (!finite) {
                        cerr << "warning: non-finite gradient at step " << step
                             << " (batch starting at example " << start << "), skipping this update" << endl;
                    } else {
                        double grad_norm = sqrt(sq_norm);
                        double clip_scale = grad_norm > GRAD_CLIP ? GRAD_CLIP / grad_norm : 1.0;
                        double token_scale = 1.0 / (double)batch_token_total;
                        double lr_t = lr_at(batch_num, warmup, total_steps);

                        apply_gradients(thread_data[0], opt, lr_t, token_scale * clip_scale);

                        epoch_loss += batch_loss_sum;
                        epoch_tokens += batch_token_total;
                        step += this_batch;
                        batch_num++;

                        double batch_avg_loss = batch_loss_sum / batch_token_total;
                        if (log_every > 0 && step - last_log_step >= log_every) {
                            cout << "epoch " << epoch + 1 << " step " << step << " lr " << lr_t
                                 << " loss " << batch_avg_loss << endl;
                            last_log_step = step;
                        }
                        if (!checkpoint_path.empty() && checkpoint_every > 0 && step - last_checkpoint_step >= checkpoint_every) {
                            save_checkpoint(checkpoint_path, opt, epoch, step);
                            cout << "checkpoint saved to " << checkpoint_path << endl;
                            last_checkpoint_step = step;
                        }
                    }
                }
            }
            if (epoch_tokens > 0) {
                cout << "epoch " << epoch + 1 << "/" << epochs << " avg loss (per token) " << epoch_loss / epoch_tokens << endl;
            }
        }

        if (!checkpoint_path.empty()) {
            save_checkpoint(checkpoint_path, opt, epoch, step);
            cout << "Final checkpoint saved to " << checkpoint_path << endl;
        }
    }

    void save_checkpoint(const string& path, const AdamState& opt, int epoch, int step) const {
        ofstream out(path, ios::binary);
        uint32_t magic = CHECKPOINT_MAGIC;
        out.write((const char*)&magic, sizeof(magic));
        auto wi = [&](int v) { out.write((const char*)&v, sizeof(v)); };
        wi(num_layers); wi(num_heads); wi(dim_head); wi(dim_model);
        wi(hidden_dim); wi(vocab_size); wi(max_seq_len);
        wi(epoch); wi(step);

        W_e.save_bin(out);
        for (int l = 0; l < num_layers; l++) {
            W_o_layers[l].save_bin(out);
            Matrix::save_vec_bin(out, attn_norms[l].g);
            Matrix::save_vec_bin(out, ffn_norms[l].g);
            ffn_layers[l].W_gate.save_bin(out);
            ffn_layers[l].W_up.save_bin(out);
            ffn_layers[l].W_down.save_bin(out);
            for (int h = 0; h < num_heads; h++) {
                attention_layers[l].heads[h].W_q.save_bin(out);
                attention_layers[l].heads[h].W_k.save_bin(out);
                attention_layers[l].heads[h].W_v.save_bin(out);
            }
        }
        Matrix::save_vec_bin(out, final_norm.g);

        opt.save(out, num_layers, num_heads);
    }

    static GPT load_checkpoint(const string& path, AdamState* opt_out = nullptr, int* epoch_out = nullptr, int* step_out = nullptr) {
        uint32_t magic = 0;
        {
            ifstream probe(path, ios::binary);
            probe.read((char*)&magic, sizeof(magic));
        }

        if (magic == CHECKPOINT_MAGIC) {
            ifstream in(path, ios::binary);
            uint32_t m; in.read((char*)&m, sizeof(m));
            int num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_, max_seq_len_, epoch_, step_;
            auto ri = [&](int& v) { in.read((char*)&v, sizeof(v)); };
            ri(num_layers_); ri(num_heads_); ri(dim_head_); ri(dim_model_);
            ri(hidden_dim_); ri(vocab_size_); ri(max_seq_len_);
            ri(epoch_); ri(step_);

            GPT gpt(num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_, max_seq_len_);
            gpt.W_e.load_bin(in);
            for (int l = 0; l < num_layers_; l++) {
                gpt.W_o_layers[l].load_bin(in);
                Matrix::load_vec_bin(in, gpt.attn_norms[l].g);
                Matrix::load_vec_bin(in, gpt.ffn_norms[l].g);
                gpt.ffn_layers[l].W_gate.load_bin(in);
                gpt.ffn_layers[l].W_up.load_bin(in);
                gpt.ffn_layers[l].W_down.load_bin(in);
                for (int h = 0; h < num_heads_; h++) {
                    gpt.attention_layers[l].heads[h].W_q.load_bin(in);
                    gpt.attention_layers[l].heads[h].W_k.load_bin(in);
                    gpt.attention_layers[l].heads[h].W_v.load_bin(in);
                }
            }
            Matrix::load_vec_bin(in, gpt.final_norm.g);

            if (epoch_out) *epoch_out = epoch_;
            if (step_out) *step_out = step_;
            if (opt_out) {
                *opt_out = AdamState(num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_);
                opt_out->load(in, num_layers_, num_heads_);
            }
            return gpt;
        }

        ifstream legacy(path);
        int num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_, max_seq_len_;
        legacy >> num_layers_ >> num_heads_ >> dim_head_ >> dim_model_ >> hidden_dim_ >> vocab_size_ >> max_seq_len_;

        GPT gpt(num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_, max_seq_len_);
        gpt.W_e.load(legacy);
        Matrix legacy_w_u(0, 0);
        legacy_w_u.load(legacy);
        for (int l = 0; l < num_layers_; l++) {
            gpt.W_o_layers[l].load(legacy);
            Matrix::load_vec(legacy, gpt.attn_norms[l].g);
            Matrix::load_vec(legacy, gpt.ffn_norms[l].g);
            gpt.ffn_layers[l].W_gate.load(legacy);
            gpt.ffn_layers[l].W_up.load(legacy);
            gpt.ffn_layers[l].W_down.load(legacy);
            for (int h = 0; h < num_heads_; h++) {
                gpt.attention_layers[l].heads[h].W_q.load(legacy);
                gpt.attention_layers[l].heads[h].W_k.load(legacy);
                gpt.attention_layers[l].heads[h].W_v.load(legacy);
            }
        }
        Matrix::load_vec(legacy, gpt.final_norm.g);

        cout << "Loaded legacy checkpoint (pre-tied-embedding format): kept W_e, discarded the "
                "separately-trained W_u; optimizer state reset to t=0 (this format has no saved "
                "Adam state). Subsequent checkpoints will be full-fidelity." << endl;

        if (epoch_out) *epoch_out = 0;
        if (step_out) *step_out = 0;
        if (opt_out) *opt_out = AdamState(num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_);
        return gpt;
    }
};
