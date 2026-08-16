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

    static ChatExample build_example(const vector<ChatTurn>& conv, const Tokenizer& tok, int max_len) {
        ChatExample ex;
        for (const auto& turn : conv) {
            string role_label = turn.role.empty() ? "User" : turn.role;
            if (!role_label.empty()) role_label[0] = toupper((unsigned char)role_label[0]);
            string prefix = role_label + ": ";

            for (int id : tok.encode(prefix)) {
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
        for (int id : tok.encode("<|endoftext|>")) {
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

    static vector<ChatExample> build_examples(const vector<vector<ChatTurn>>& conversations,
                                               const Tokenizer& tok, int max_len) {
        vector<ChatExample> examples;
        examples.reserve(conversations.size());
        for (auto& conv : conversations) {
            ChatExample ex = build_example(conv, tok, max_len);
            if (has_valid_target(ex)) examples.push_back(move(ex));
        }
        return examples;
    }
};
