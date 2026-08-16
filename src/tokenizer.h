#pragma once
#include <bits/stdc++.h>
using namespace std;

struct VectorStringHash {
    size_t operator()(const vector<string>& v) const {
        size_t seed = v.size();
        for (auto& s : v) {
            seed ^= hash<string>{}(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct Tokenizer {
    vector<string> id_to_token;
    unordered_map<string, int> token_to_id;
    vector<pair<string,string>> merges;
    unordered_map<string, int> merge_rank;
    vector<string> special_tokens = {"<|endoftext|>", "<|unk|>", "<|pad|>", "<|mask|>"};
    bool lowercase = true;
    bool split_punctuation = true;
    bool normalize_whitespace = true;

    int vocab_size() const { return (int)id_to_token.size(); }

    void train(const string& corpus_path, int target_vocab_size, int min_frequency,
               const string& vocab_out_path, const string& merges_out_path) {
        ifstream file(corpus_path);
        unordered_map<vector<string>, long long, VectorStringHash> word_freqs;
        string line;
        while (getline(file, line)) {
            if (line.empty()) continue;
            string normalized = normalize(line, lowercase, split_punctuation, normalize_whitespace);
            istringstream iss(normalized);
            string word;
            while (iss >> word) {
                word_freqs[word_to_chars(word)]++;
            }
        }

        vector<vector<string>> words;
        vector<long long> freqs;
        words.reserve(word_freqs.size());
        freqs.reserve(word_freqs.size());
        for (auto& kv : word_freqs) {
            words.push_back(kv.first);
            freqs.push_back(kv.second);
        }

        unordered_set<string> vocabulary;
        for (auto& w : words) for (auto& sym : w) vocabulary.insert(sym);
        for (auto& special : special_tokens) vocabulary.insert(special);

        unordered_map<string, long long> pair_counts;
        unordered_map<string, unordered_set<int>> pair_words;
        for (size_t idx = 0; idx < words.size(); idx++) {
            auto& w = words[idx];
            for (size_t i = 0; i + 1 < w.size(); i++) {
                string key = w[i] + " " + w[i + 1];
                pair_counts[key] += freqs[idx];
                pair_words[key].insert((int)idx);
            }
        }

        using HeapEntry = pair<long long, string>;
        priority_queue<HeapEntry> heap;
        for (auto& kv : pair_counts) heap.push({kv.second, kv.first});

        vector<pair<string,string>> merges_out;

        while ((int)vocabulary.size() < target_vocab_size && !heap.empty()) {
            auto top = heap.top(); heap.pop();
            long long count = top.first;
            const string& key = top.second;
            auto it = pair_counts.find(key);
            if (it == pair_counts.end() || it->second != count) continue;
            if (count < min_frequency) break;

            size_t sp = key.find(' ');
            string first = key.substr(0, sp), second = key.substr(sp + 1);
            merges_out.push_back({first, second});
            vocabulary.insert(first + second);

            vector<int> affected(pair_words[key].begin(), pair_words[key].end());
            for (int idx : affected) {
                auto& w = words[idx];
                for (size_t i = 0; i + 1 < w.size(); i++) {
                    pair_counts[w[i] + " " + w[i + 1]] -= freqs[idx];
                }
                w = apply_merge(w, first, second);
                for (size_t i = 0; i + 1 < w.size(); i++) {
                    string k = w[i] + " " + w[i + 1];
                    pair_counts[k] += freqs[idx];
                    pair_words[k].insert(idx);
                    heap.push({pair_counts[k], k});
                }
            }
            pair_counts.erase(key);
        }

        ofstream vf(vocab_out_path);
        for (auto& s : special_tokens) vf << s << "\n";
        vector<string> rest;
        for (auto& t : vocabulary) {
            if (find(special_tokens.begin(), special_tokens.end(), t) == special_tokens.end()) {
                rest.push_back(t);
            }
        }
        sort(rest.begin(), rest.end());
        for (auto& t : rest) vf << t << "\n";
        vf.close();

        ofstream mf(merges_out_path);
        mf << "#version: 0.2\n";
        for (auto& m : merges_out) mf << m.first << " " << m.second << "\n";
        mf.close();

        load(vocab_out_path, merges_out_path);
    }

    void load(const string& vocab_path, const string& merges_path) {
        id_to_token.clear();
        token_to_id.clear();
        ifstream vf(vocab_path);
        string line;
        while (getline(vf, line)) {
            if (line.empty()) continue;
            token_to_id[line] = (int)id_to_token.size();
            id_to_token.push_back(line);
        }

        merges.clear();
        ifstream mf(merges_path);
        bool first = true;
        while (getline(mf, line)) {
            if (line.empty()) continue;
            if (first) {
                first = false;
                if (line[0] == '#') continue;
            }
            size_t sp = line.find(' ');
            if (sp == string::npos) continue;
            merges.push_back({line.substr(0, sp), line.substr(sp + 1)});
        }

        merge_rank.clear();
        for (int r = 0; r < (int)merges.size(); r++) {
            merge_rank[merges[r].first + " " + merges[r].second] = r;
        }
    }

    static string normalize(const string& text, bool lowercase, bool split_punct, bool norm_ws) {
        string result = text;
        if (norm_ws) {
            static const regex ws_regex("\\s+");
            result = regex_replace(result, ws_regex, " ");
        }
        if (split_punct) {
            static const regex punct_regex("([.,!?;:()\\[\\]{}\"'])");
            result = regex_replace(result, punct_regex, " $1 ");
        }
        if (lowercase) {
            transform(result.begin(), result.end(), result.begin(), ::tolower);
        }
        return result;
    }

    static vector<string> word_to_chars(const string& word) {
        vector<string> chars;
        for (size_t i = 0; i < word.size(); ) {
            unsigned char c = word[i];
            size_t char_len = 1;
            if ((c & 0x80) == 0) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            chars.push_back(word.substr(i, char_len));
            i += char_len;
        }
        if (!chars.empty()) chars.back() += "</w>";
        return chars;
    }

    static vector<string> apply_merge(const vector<string>& word, const string& first, const string& second) {
        vector<string> result;
        size_t i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                result.push_back(first + second);
                i += 2;
            } else {
                result.push_back(word[i]);
                i++;
            }
        }
        return result;
    }

    vector<string> tokenize_to_strings(const string& text) const {
        string normalized = normalize(text, lowercase, split_punctuation, normalize_whitespace);
        istringstream iss(normalized);
        string word;
        vector<string> result;
        while (iss >> word) {
            auto chars = word_to_chars(word);
            while (chars.size() > 1) {
                int best_rank = INT_MAX;
                size_t best_i = string::npos;
                for (size_t i = 0; i + 1 < chars.size(); i++) {
                    auto it = merge_rank.find(chars[i] + " " + chars[i + 1]);
                    if (it != merge_rank.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_i = i;
                    }
                }
                if (best_i == string::npos) break;
                chars = apply_merge(chars, chars[best_i], chars[best_i + 1]);
            }
            result.insert(result.end(), chars.begin(), chars.end());
        }
        return result;
    }

    vector<int> encode(const string& text) const {
        vector<int> ids;
        string buffer;
        auto flush_buffer = [&]() {
            if (buffer.empty()) return;
            for (auto& tok : tokenize_to_strings(buffer)) {
                auto it = token_to_id.find(tok);
                ids.push_back(it != token_to_id.end() ? it->second : token_to_id.at("<|unk|>"));
            }
            buffer.clear();
        };

        size_t i = 0;
        while (i < text.size()) {
            const string* matched = nullptr;
            for (auto& special : special_tokens) {
                if (!special.empty() && text.compare(i, special.size(), special) == 0) {
                    matched = &special;
                    break;
                }
            }
            if (matched && token_to_id.count(*matched)) {
                flush_buffer();
                ids.push_back(token_to_id.at(*matched));
                i += matched->size();
            } else {
                buffer += text[i];
                i++;
            }
        }
        flush_buffer();
        return ids;
    }

    string decode(const vector<int>& ids) const {
        static const string marker = "</w>";
        string out, word;
        for (int id : ids) {
            if (id < 0 || id >= (int)id_to_token.size()) continue;
            string tok = id_to_token[id];

            bool is_special = find(special_tokens.begin(), special_tokens.end(), tok) != special_tokens.end();
            if (is_special) {
                if (!word.empty()) { out += word; word.clear(); }
                if (!out.empty() && out.back() != ' ') out += ' ';
                out += tok + " ";
                continue;
            }

            bool end_of_word = tok.size() >= marker.size() &&
                tok.compare(tok.size() - marker.size(), marker.size(), marker) == 0;
            if (end_of_word) tok = tok.substr(0, tok.size() - marker.size());
            word += tok;
            if (end_of_word) {
                out += word + " ";
                word.clear();
            }
        }
        out += word;
        if (!out.empty() && out.back() == ' ') out.pop_back();
        return out;
    }

    static void save_tokens(const string& path, const vector<int>& ids) {
        ofstream f(path);
        for (size_t i = 0; i < ids.size(); i++) {
            f << ids[i];
            if (i + 1 < ids.size()) f << ' ';
        }
    }

    static vector<int> load_tokens(const string& path) {
        ifstream f(path);
        vector<int> ids;
        int v;
        while (f >> v) ids.push_back(v);
        return ids;
    }
};
