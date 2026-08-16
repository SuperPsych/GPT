#pragma once
#include <omp.h>
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

struct GPT{
    int num_layers, num_heads, dim_head, dim_model, hidden_dim, vocab_size, max_seq_len;
    Matrix W_e;
    Matrix W_u;
    vector<AttentionLayer> attention_layers;
    vector<Matrix> W_o_layers;
    vector<RMSNorm> attn_norms;
    vector<RMSNorm> ffn_norms;
    vector<FFN> ffn_layers;
    RMSNorm final_norm;
    Tokenizer tokenizer;

    GPT(int num_layers, int num_heads, int dim_head, int dim_model, int hidden_dim, int vocab_size, int max_seq_len) :
        num_layers(num_layers), num_heads(num_heads), dim_head(dim_head), dim_model(dim_model), hidden_dim(hidden_dim),
        vocab_size(vocab_size), max_seq_len(max_seq_len),
        W_e(vocab_size, dim_model), W_u(vocab_size, dim_model),
        final_norm(dim_model) {
        W_e.randomize();
        W_u.randomize();

        attention_layers.reserve(num_layers);
        W_o_layers.reserve(num_layers);
        ffn_layers.reserve(num_layers);
        attn_norms.reserve(num_layers);
        ffn_norms.reserve(num_layers);
        for (int l = 0; l < num_layers; l++) {
            attention_layers.emplace_back(num_heads, dim_head, dim_model);

            W_o_layers.emplace_back(dim_model, num_heads * dim_head);
            W_o_layers.back().randomize();

            ffn_layers.emplace_back(dim_model, hidden_dim);
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
            logits[i] = W_u.times(normed_final);
        }
        return logits;
    }

    // Training hot path: buffers are owned by GPTScratch and reused across
    // examples/tokens/layers (no per-call allocation), and every projection
    // (Q/K/V, W_o, FFN gate/up/down, W_u) is one GEMM over the whole sequence
    // instead of a per-token times() loop. RMSNorm and RoPE stay per-token
    // since neither is a matmul.
    Matrix& forward_train(const vector<int>& tokens, GPTScratch& s) {
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
        W_u.times(s.normed_final_buf, s.logits_buf, n);
        return s.logits_buf;
    }

    void backward(const Matrix& grad_logits, GPTScratch& s) {
        int n = s.n;

        // dW_u += grad_logits^T @ normed_final (cached from forward_train);
        // dnormed_final = grad_logits @ W_u. Both are one GEMM instead of a
        // per-token double loop.
        Matrix::accumulate_weight_grad(grad_logits, s.normed_final_buf, s.dW_u, n);
        W_u.backward_input(grad_logits, s.dnormed_final_buf, n, /*accumulate=*/false);

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

            // dW_o += dx_buf^T @ concat; dconcat = dx_buf @ W_o. Was a
            // per-token double-nested loop for each; now one GEMM each.
            Matrix::accumulate_weight_grad(s.dx_buf, s.concat[l], s.dW_o[l], n);
            W_o_layers[l].backward_input(s.dx_buf, s.dconcat_buf, n, /*accumulate=*/false);

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
                head.K = Matrix(0, dim_head);
                head.V = Matrix(0, dim_head);
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
        return W_u.times(normed_final);
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
                          double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weight_decay = 0.01) {
        opt.t++;
        int t = opt.t;
        Matrix::adamw_update(W_e, s.dW_e, opt.m_W_e, opt.v_W_e, lr, beta1, beta2, eps, weight_decay, t, grad_scale);
        Matrix::adamw_update(W_u, s.dW_u, opt.m_W_u, opt.v_W_u, lr, beta1, beta2, eps, weight_decay, t, grad_scale);
        for (int l = 0; l < num_layers; l++) {
            Matrix::adamw_update(W_o_layers[l], s.dW_o[l], opt.m_W_o[l], opt.v_W_o[l], lr, beta1, beta2, eps, weight_decay, t, grad_scale);
            Matrix::adamw_update(ffn_layers[l].W_gate, s.dW_ffn_gate[l], opt.m_ffn_gate[l], opt.v_ffn_gate[l], lr, beta1, beta2, eps, weight_decay, t, grad_scale);
            Matrix::adamw_update(ffn_layers[l].W_up, s.dW_ffn_up[l], opt.m_ffn_up[l], opt.v_ffn_up[l], lr, beta1, beta2, eps, weight_decay, t, grad_scale);
            Matrix::adamw_update(ffn_layers[l].W_down, s.dW_ffn_down[l], opt.m_ffn_down[l], opt.v_ffn_down[l], lr, beta1, beta2, eps, weight_decay, t, grad_scale);
            Matrix::adamw_update(attn_norms[l].g, s.dg_attn_norms[l], opt.m_attn_norms[l], opt.v_attn_norms[l], lr, beta1, beta2, eps, 0.0, t, grad_scale);
            Matrix::adamw_update(ffn_norms[l].g, s.dg_ffn_norms[l], opt.m_ffn_norms[l], opt.v_ffn_norms[l], lr, beta1, beta2, eps, 0.0, t, grad_scale);
            for (int h = 0; h < num_heads; h++) {
                Matrix::adamw_update(attention_layers[l].heads[h].W_q, s.attn_scratch[l][h].dW_q, opt.m_W_q[l][h], opt.v_W_q[l][h], lr, beta1, beta2, eps, weight_decay, t, grad_scale);
                Matrix::adamw_update(attention_layers[l].heads[h].W_k, s.attn_scratch[l][h].dW_k, opt.m_W_k[l][h], opt.v_W_k[l][h], lr, beta1, beta2, eps, weight_decay, t, grad_scale);
                Matrix::adamw_update(attention_layers[l].heads[h].W_v, s.attn_scratch[l][h].dW_v, opt.m_W_v[l][h], opt.v_W_v[l][h], lr, beta1, beta2, eps, weight_decay, t, grad_scale);
            }
        }
        Matrix::adamw_update(final_norm.g, s.dg_final_norm, opt.m_final_norm, opt.v_final_norm, lr, beta1, beta2, eps, 0.0, t, grad_scale);
    }

    double accumulate_gradients(const ChatExample& example, GPTScratch& scratch) {
        int n = (int)example.tokens.size();
        Matrix& logits = forward_train(example.tokens, scratch);

        fill(scratch.grad_logits.data.begin(), scratch.grad_logits.data.begin() + (size_t)n * vocab_size, 0.0f);

        double loss = 0.0;
        int target_count = 0;
        for (int i = 0; i + 1 < n; i++) {
            if (!example.mask[i + 1]) continue;
            span<float> probs = logits.row(i);
            FFN::output_activation(probs);
            int target = example.tokens[i + 1];
            loss -= log(max(probs[target], 1e-12f));
            for (int v = 0; v < vocab_size; v++) {
                scratch.grad_logits.at(i, v) = probs[v] - (v == target ? 1.0f : 0.0f);
            }
            target_count++;
        }
        if (target_count > 0) {
            loss /= target_count;
            float scale = 1.0f / target_count;
            for (int idx = 0; idx < n * vocab_size; idx++) scratch.grad_logits.data[idx] *= scale;
        }

        backward(scratch.grad_logits, scratch);
        return loss;
    }

    double train_step(const ChatExample& example, GPTScratch& scratch, AdamState& opt, double lr) {
        scratch.clear();
        double loss = accumulate_gradients(example, scratch);
        apply_gradients(scratch, opt, lr);
        return loss;
    }

    static void add_scratch(GPTScratch& into, const GPTScratch& from, int num_layers, int num_heads) {
        Matrix::add(into.dW_e, from.dW_e);
        Matrix::add(into.dW_u, from.dW_u);
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

    void train(vector<ChatExample>& examples, int epochs, double lr, int batch_size,
               const string& checkpoint_path = "", int checkpoint_every = 0, int log_every = 50) {
        omp_set_dynamic(0);
        int thread_num = omp_get_max_threads();
        cout << "Beginning GPT training on " << examples.size() << " examples (batch_size=" << batch_size
             << ", threads=" << thread_num << ")..." << endl;

        AdamState opt(num_layers, num_heads, dim_head, dim_model, hidden_dim, vocab_size);

        vector<GPTScratch> thread_data;
        thread_data.reserve(thread_num);
        for (int t = 0; t < thread_num; t++) {
            thread_data.emplace_back(max_seq_len, num_layers, num_heads, dim_head, dim_model, hidden_dim, vocab_size);
        }

        // Reshuffled every epoch (not just once): with epochs > 1 the same
        // fixed batch composition would otherwise repeat on every pass.
        mt19937 shuffle_rng(random_device{}());

        int step = 0, last_log_step = 0, last_checkpoint_step = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(examples.begin(), examples.end(), shuffle_rng);

            double epoch_loss = 0.0;
            int counted = 0;

            for (int start = 0; start < (int)examples.size(); start += batch_size) {
                int end = min(start + batch_size, (int)examples.size());
                int this_batch = end - start;

                for (GPTScratch& td : thread_data) td.clear();

                vector<double> batch_losses(this_batch, 0.0);
                vector<char> batch_valid(this_batch, 0);

                #pragma omp parallel for
                for (int i = 0; i < this_batch; i++) {
                    auto& ex = examples[start + i];
                    if (ex.tokens.size() < 2) continue;
                    GPTScratch& td = thread_data[omp_get_thread_num()];
                    batch_losses[i] = accumulate_gradients(ex, td);
                    batch_valid[i] = 1;
                }

                for (int t = 1; t < thread_num; t++) {
                    add_scratch(thread_data[0], thread_data[t], num_layers, num_heads);
                }

                int valid_count = 0;
                double batch_loss_sum = 0.0;
                for (int i = 0; i < this_batch; i++) {
                    if (batch_valid[i]) { valid_count++; batch_loss_sum += batch_losses[i]; }
                }

                if (valid_count > 0) {
                    apply_gradients(thread_data[0], opt, lr, 1.0 / valid_count);
                    epoch_loss += batch_loss_sum;
                    counted += valid_count;
                    step += valid_count;

                    double batch_avg_loss = batch_loss_sum / valid_count;
                    if (log_every > 0 && step - last_log_step >= log_every) {
                        cout << "epoch " << epoch + 1 << " step " << step << " loss " << batch_avg_loss << endl;
                        last_log_step = step;
                    }
                    if (!checkpoint_path.empty() && checkpoint_every > 0 && step - last_checkpoint_step >= checkpoint_every) {
                        save_checkpoint(checkpoint_path);
                        cout << "checkpoint saved to " << checkpoint_path << endl;
                        last_checkpoint_step = step;
                    }
                }
            }
            if (counted > 0) {
                cout << "epoch " << epoch + 1 << "/" << epochs << " avg loss " << epoch_loss / counted << endl;
            }
        }
    }

    void save_checkpoint(const string& path) const {
        ofstream out(path);
        out << num_layers << " " << num_heads << " " << dim_head << " " << dim_model << " "
            << hidden_dim << " " << vocab_size << " " << max_seq_len << "\n";
        W_e.save(out);
        W_u.save(out);
        for (int l = 0; l < num_layers; l++) {
            W_o_layers[l].save(out);
            Matrix::save_vec(out, attn_norms[l].g);
            Matrix::save_vec(out, ffn_norms[l].g);
            ffn_layers[l].W_gate.save(out);
            ffn_layers[l].W_up.save(out);
            ffn_layers[l].W_down.save(out);
            for (int h = 0; h < num_heads; h++) {
                attention_layers[l].heads[h].W_q.save(out);
                attention_layers[l].heads[h].W_k.save(out);
                attention_layers[l].heads[h].W_v.save(out);
            }
        }
        Matrix::save_vec(out, final_norm.g);
    }

    static GPT load_checkpoint(const string& path) {
        ifstream in(path);
        int num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_, max_seq_len_;
        in >> num_layers_ >> num_heads_ >> dim_head_ >> dim_model_ >> hidden_dim_ >> vocab_size_ >> max_seq_len_;

        GPT gpt(num_layers_, num_heads_, dim_head_, dim_model_, hidden_dim_, vocab_size_, max_seq_len_);
        gpt.W_e.load(in);
        gpt.W_u.load(in);
        for (int l = 0; l < num_layers_; l++) {
            gpt.W_o_layers[l].load(in);
            Matrix::load_vec(in, gpt.attn_norms[l].g);
            Matrix::load_vec(in, gpt.ffn_norms[l].g);
            gpt.ffn_layers[l].W_gate.load(in);
            gpt.ffn_layers[l].W_up.load(in);
            gpt.ffn_layers[l].W_down.load(in);
            for (int h = 0; h < num_heads_; h++) {
                gpt.attention_layers[l].heads[h].W_q.load(in);
                gpt.attention_layers[l].heads[h].W_k.load(in);
                gpt.attention_layers[l].heads[h].W_v.load(in);
            }
        }
        Matrix::load_vec(in, gpt.final_norm.g);
        return gpt;
    }
};
