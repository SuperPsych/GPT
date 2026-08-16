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

    // Role prefixes ("User: ", "Assistant: ", ...) and "<|endoftext|>" repeat
    // identically across every one of the hundreds of thousands of turns/examples,
    // so cache their token encodings once instead of re-running BPE on them every time.
    static ChatExample build_example(const vector<ChatTurn>& conv, const Tokenizer& tok, int max_len) {
        static unordered_map<string, vector<int>> prefix_cache;
        static const vector<int> eot_tokens = tok.encode("<|endoftext|>");

        ChatExample ex;
        for (const auto& turn : conv) {
            string role_label = turn.role.empty() ? "User" : turn.role;
            if (!role_label.empty()) role_label[0] = toupper((unsigned char)role_label[0]);

            auto it = prefix_cache.find(role_label);
            if (it == prefix_cache.end()) {
                it = prefix_cache.emplace(role_label, tok.encode(role_label + ": ")).first;
            }
            for (int id : it->second) {
                ex.tokens.push_back(id);
                ex.mask.push_back(false);
            }

            bool is_assistant = turn.role == "assistant";
            for (int id : tok.encode(turn.content)) {
                ex.tokens.push_back(id);
                ex.mask.push_back(is_assistant);
            }
            if ((int)ex.tokens.size() >= max_len) break;
        }
        for (int id : eot_tokens) {
            ex.tokens.push_back(id);
            ex.mask.push_back(true);
        }
        if ((int)ex.tokens.size() > max_len) {
            ex.tokens.resize(max_len);
            ex.mask.resize(max_len);
        }
        return ex;
    }

    static bool has_valid_target(const ChatExample& ex) {
        for (size_t i = 1; i < ex.mask.size(); i++) {
            if (ex.mask[i]) return true;
        }
        return false;
    }

    // Takes num_examples rather than a pre-sliced subrange so callers never need to
    // copy conversations into a separate vector just to bound how many are used.
    static vector<ChatExample> build_examples(const vector<vector<ChatTurn>>& conversations,
                                               const Tokenizer& tok, int max_len, int num_examples) {
        int n = min((int)conversations.size(), num_examples);
        vector<ChatExample> examples;
        examples.reserve(n);
        for (int i = 0; i < n; i++) {
            ChatExample ex = build_example(conversations[i], tok, max_len);
            if (has_valid_target(ex)) examples.push_back(move(ex));
        }
        return examples;
    }

    // Binary cache of tokenized examples, so a re-run with the same tokenizer/max_len
    // can skip re-loading the raw dataset and re-running BPE encode() on everything.
    static void save_examples_cache(const string& path, const vector<ChatExample>& examples,
                                     int vocab_size, int max_seq_len, int num_examples) {
        ofstream out(path, ios::binary);
        uint32_t magic = 0x43484558; // "CHEX"
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
