#pragma once
#include <bits/stdc++.h>
#include "json.h"
#include "tokenizer.h"
using namespace std;

struct ChatTurn {
    string role;
    string content;
};

struct ChatExample {
    vector<int> tokens;
    vector<bool> mask;
};

struct ChatDataset {
    static vector<vector<ChatTurn>> load_conversations(const string& pages_path) {
        vector<vector<ChatTurn>> conversations;
        ifstream f(pages_path);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            JsonValue page = parse_json(line);
            const JsonValue& rows = page["rows"];
            for (size_t i = 0; i < rows.size(); i++) {
                const JsonValue& messages = rows[i]["row"]["messages"];
                vector<ChatTurn> conv;
                for (size_t m = 0; m < messages.size(); m++) {
                    conv.push_back({messages[m]["role"].str, messages[m]["content"].str});
                }
                if (!conv.empty()) conversations.push_back(move(conv));
            }
        }
        return conversations;
    }

    static vector<int> encode_turn(const ChatTurn& turn, const Tokenizer& tok, int max_len,
                                    unordered_map<string, vector<int>>& role_start_cache,
                                    const vector<int>& im_end_tokens) {
        string role = turn.role.empty() ? "user" : turn.role;
        auto it = role_start_cache.find(role);
        if (it == role_start_cache.end()) {
            it = role_start_cache.emplace(role, tok.encode("<|im_start|>" + role + "\n")).first;
        }
        vector<int> encoded = it->second;
        vector<int> content_tokens = tok.encode(turn.content);
        encoded.insert(encoded.end(), content_tokens.begin(), content_tokens.end());
        encoded.insert(encoded.end(), im_end_tokens.begin(), im_end_tokens.end());
        return encoded;
    }

    static void append_turn(ChatExample& ex, const ChatTurn& turn, const vector<int>& encoded) {
        bool is_assistant = (turn.role == "assistant");
        for (int id : encoded) { ex.tokens.push_back(id); ex.mask.push_back(is_assistant); }
    }

    static ChatExample build_example(const vector<ChatTurn>& conv, const Tokenizer& tok, int max_len,
                                      unordered_map<string, vector<int>>& role_start_cache,
                                      const vector<int>& im_end_tokens) {
        ChatExample ex;

        size_t i = 0;
        while (i < conv.size() && conv[i].role != "user" && conv[i].role != "assistant") {
            vector<int> encoded = encode_turn(conv[i], tok, max_len, role_start_cache, im_end_tokens);
            if ((int)ex.tokens.size() + (int)encoded.size() > max_len) return ex;
            append_turn(ex, conv[i], encoded);
            i++;
        }

        while (i + 1 < conv.size()) {
            vector<int> user_encoded = encode_turn(conv[i], tok, max_len, role_start_cache, im_end_tokens);
            vector<int> assistant_encoded = encode_turn(conv[i + 1], tok, max_len, role_start_cache, im_end_tokens);
            int pair_len = (int)(user_encoded.size() + assistant_encoded.size());
            if ((int)ex.tokens.size() + pair_len > max_len) break;

            append_turn(ex, conv[i], user_encoded);
            append_turn(ex, conv[i + 1], assistant_encoded);
            i += 2;
        }

        return ex;
    }

    static bool has_valid_target(const ChatExample& ex) {
        for (size_t i = 1; i < ex.mask.size(); i++) {
            if (ex.mask[i]) return true;
        }
        return false;
    }

    static vector<ChatExample> build_examples(const vector<vector<ChatTurn>>& conversations,
                                               const Tokenizer& tok, int max_len, int num_examples) {
        unordered_map<string, vector<int>> role_start_cache;
        vector<int> im_end_tokens = tok.encode("<|im_end|>\n");

        int n = min((int)conversations.size(), num_examples);
        vector<ChatExample> examples;
        examples.reserve(n);
        for (int i = 0; i < n; i++) {
            ChatExample ex = build_example(conversations[i], tok, max_len, role_start_cache, im_end_tokens);
            if (has_valid_target(ex)) examples.push_back(move(ex));
        }
        return examples;
    }

    static void save_examples_cache(const string& path, const vector<ChatExample>& examples,
                                     int vocab_size, int max_seq_len, int num_examples) {
        ofstream out(path, ios::binary);
        uint32_t magic = 0x43484558;
        out.write((const char*)&magic, sizeof(magic));
        out.write((const char*)&vocab_size, sizeof(vocab_size));
        out.write((const char*)&max_seq_len, sizeof(max_seq_len));
        out.write((const char*)&num_examples, sizeof(num_examples));
        uint64_t count = examples.size();
        out.write((const char*)&count, sizeof(count));
        for (const auto& ex : examples) {
            uint32_t n = (uint32_t)ex.tokens.size();
            out.write((const char*)&n, sizeof(n));
            out.write((const char*)ex.tokens.data(), n * sizeof(int));
            for (uint32_t i = 0; i < n; i++) {
                char m = ex.mask[i] ? 1 : 0;
                out.write(&m, 1);
            }
        }
    }

    static bool load_examples_cache(const string& path, vector<ChatExample>& examples,
                                     int vocab_size, int max_seq_len, int num_examples) {
        ifstream in(path, ios::binary);
        if (!in.good()) return false;

        uint32_t magic = 0;
        in.read((char*)&magic, sizeof(magic));
        int cached_vocab_size = 0, cached_max_seq_len = 0, cached_num_examples = 0;
        in.read((char*)&cached_vocab_size, sizeof(cached_vocab_size));
        in.read((char*)&cached_max_seq_len, sizeof(cached_max_seq_len));
        in.read((char*)&cached_num_examples, sizeof(cached_num_examples));
        if (!in.good() || magic != 0x43484558 || cached_vocab_size != vocab_size ||
            cached_max_seq_len != max_seq_len || cached_num_examples != num_examples) {
            return false;
        }

        uint64_t count = 0;
        in.read((char*)&count, sizeof(count));
        if (!in.good()) return false;

        vector<ChatExample> loaded;
        loaded.reserve(count);
        for (uint64_t e = 0; e < count; e++) {
            uint32_t n = 0;
            in.read((char*)&n, sizeof(n));
            ChatExample ex;
            ex.tokens.resize(n);
            in.read((char*)ex.tokens.data(), n * sizeof(int));
            ex.mask.resize(n);
            for (uint32_t i = 0; i < n; i++) {
                char m = 0;
                in.read(&m, 1);
                ex.mask[i] = (m != 0);
            }
            if (!in.good()) return false;
            loaded.push_back(move(ex));
        }
        examples = move(loaded);
        return true;
    }
};
