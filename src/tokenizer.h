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
    vector<string> special_tokens = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"};

    int vocab_size() const { return (int)id_to_token.size(); }

    static const array<int,256>& byte_to_unicode() {
        static array<int,256> table = [] {
            array<int,256> t{};
            array<bool,256> assigned{};
            vector<int> bs;
            for (int b = 33; b <= 126; b++) bs.push_back(b);
            for (int b = 161; b <= 172; b++) bs.push_back(b);
            for (int b = 174; b <= 255; b++) bs.push_back(b);
            for (int b : bs) assigned[b] = true;
            for (int b : bs) t[b] = b;
            int n = 0;
            for (int b = 0; b < 256; b++) {
                if (!assigned[b]) {
                    t[b] = 256 + n;
                    n++;
                }
            }
            return t;
        }();
        return table;
    }
    static const unordered_map<int,int>& unicode_to_byte() {
        static unordered_map<int,int> inv = [] {
            unordered_map<int,int> m;
            const auto& t = byte_to_unicode();
            for (int b = 0; b < 256; b++) m[t[b]] = b;
            return m;
        }();
        return inv;
    }

    static string utf8_encode_cp(int cp) {
        string out;
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
        return out;
    }
    static int utf8_decode_cp(const string& text, size_t& i) {
        unsigned char c0 = text[i];
        if ((c0 & 0x80) == 0) { i += 1; return c0; }
        if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
            int cp = ((c0 & 0x1F) << 6) | ((unsigned char)text[i+1] & 0x3F);
            i += 2; return cp;
        }
        if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
            int cp = ((c0 & 0x0F) << 12) | (((unsigned char)text[i+1] & 0x3F) << 6) | ((unsigned char)text[i+2] & 0x3F);
            i += 3; return cp;
        }
        if ((c0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
            int cp = ((c0 & 0x07) << 18) | (((unsigned char)text[i+1] & 0x3F) << 12)
                   | (((unsigned char)text[i+2] & 0x3F) << 6) | ((unsigned char)text[i+3] & 0x3F);
            i += 4; return cp;
        }
        i += 1; return c0;
    }

    static string bytes_to_symbol_string(const string& raw_bytes) {
        string out;
        out.reserve(raw_bytes.size() * 2);
        for (unsigned char b : raw_bytes) out += utf8_encode_cp(byte_to_unicode()[b]);
        return out;
    }
    static string symbol_string_to_bytes(const string& sym) {
        string out;
        size_t i = 0;
        while (i < sym.size()) {
            int cp = utf8_decode_cp(sym, i);
            auto it = unicode_to_byte().find(cp);
            out += (char)(it != unicode_to_byte().end() ? it->second : (cp & 0xFF));
        }
        return out;
    }
    static vector<string> symbol_string_to_chars(const string& sym) {
        vector<string> out;
        size_t i = 0;
        while (i < sym.size()) {
            size_t start = i;
            utf8_decode_cp(sym, i);
            out.push_back(sym.substr(start, i - start));
        }
        return out;
    }

    enum class CharClass { WS, DIGIT, PUNCT, OTHER };
    static CharClass classify(unsigned char c) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') return CharClass::WS;
        if (c < 0x80 && isdigit(c)) return CharClass::DIGIT;
        if (c < 0x80 && ispunct(c)) return CharClass::PUNCT;
        return CharClass::OTHER;
    }
    static vector<string> pretokenize_bytes(const string& text) {
        vector<string> chunks;
        size_t n = text.size(), i = 0;
        while (i < n) {
            CharClass cls = classify((unsigned char)text[i]);
            size_t j = i + 1;
            while (j < n && classify((unsigned char)text[j]) == cls) j++;

            if (cls == CharClass::WS && j < n) {
                if (j - i > 1) chunks.push_back(text.substr(i, j - i - 1));
                size_t fold_start = j - 1;
                CharClass next_cls = classify((unsigned char)text[j]);
                size_t k = j + 1;
                while (k < n && classify((unsigned char)text[k]) == next_cls) k++;
                chunks.push_back(text.substr(fold_start, k - fold_start));
                i = k;
            } else {
                chunks.push_back(text.substr(i, j - i));
                i = j;
            }
        }
        return chunks;
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

    void train(const string& corpus_path, int target_vocab_size, int min_frequency,
               const string& vocab_out_path, const string& merges_out_path) {
        ifstream file(corpus_path, ios::binary);
        string text((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

        unordered_map<vector<string>, long long, VectorStringHash> word_freqs;
        for (auto& chunk : pretokenize_bytes(text)) {
            if (chunk.empty()) continue;
            string sym = bytes_to_symbol_string(chunk);
            word_freqs[symbol_string_to_chars(sym)]++;
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
        for (int b = 0; b < 256; b++) vocabulary.insert(utf8_encode_cp(byte_to_unicode()[b]));
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

        ofstream vf(vocab_out_path, ios::binary);
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

        ofstream mf(merges_out_path, ios::binary);
        mf << "#version: 0.2\n";
        for (auto& m : merges_out) mf << m.first << " " << m.second << "\n";
        mf.close();

        load(vocab_out_path, merges_out_path);
    }

    void load(const string& vocab_path, const string& merges_path) {
        id_to_token.clear();
        token_to_id.clear();
        ifstream vf(vocab_path, ios::binary);
        string line;
        while (getline(vf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            token_to_id[line] = (int)id_to_token.size();
            id_to_token.push_back(line);
        }

        merges.clear();
        ifstream mf(merges_path, ios::binary);
        bool first = true;
        while (getline(mf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
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

    vector<string> tokenize_to_strings(const string& text) const {
        vector<string> result;
        for (auto& chunk : pretokenize_bytes(text)) {
            if (chunk.empty()) continue;
            string sym = bytes_to_symbol_string(chunk);
            vector<string> chars = symbol_string_to_chars(sym);
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
                ids.push_back(it != token_to_id.end() ? it->second : 0);
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
        string sym_concat, out;
        for (int id : ids) {
            if (id < 0 || id >= (int)id_to_token.size()) continue;
            const string& tok = id_to_token[id];
            bool is_special = find(special_tokens.begin(), special_tokens.end(), tok) != special_tokens.end();
            if (is_special) {
                if (!sym_concat.empty()) { out += symbol_string_to_bytes(sym_concat); sym_concat.clear(); }
                out += tok;
            } else {
                sym_concat += tok;
            }
        }
        if (!sym_concat.empty()) out += symbol_string_to_bytes(sym_concat);
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
