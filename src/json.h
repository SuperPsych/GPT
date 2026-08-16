#pragma once
#include <bits/stdc++.h>
using namespace std;

struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    string str;
    vector<JsonValue> arr;
    vector<pair<string, JsonValue>> obj;

    const JsonValue& operator[](const string& key) const {
        for (auto& kv : obj) if (kv.first == key) return kv.second;
        static JsonValue null_val;
        return null_val;
    }
    const JsonValue& operator[](size_t i) const { return arr[i]; }
    size_t size() const { return type == Array ? arr.size() : obj.size(); }
};

struct JsonParser {
    const string& s;
    size_t i = 0;
    JsonParser(const string& s) : s(s) {}

    void skip_ws() { while (i < s.size() && isspace((unsigned char)s[i])) i++; }

    JsonValue parse() {
        skip_ws();
        return parse_value();
    }

private:
    JsonValue parse_value() {
        skip_ws();
        if (i >= s.size()) return JsonValue();
        char c = s[i];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') { i += 4; return JsonValue(); }
        return parse_number();
    }

    JsonValue parse_object() {
        JsonValue v; v.type = JsonValue::Object;
        i++;
        skip_ws();
        if (i < s.size() && s[i] == '}') { i++; return v; }
        while (true) {
            skip_ws();
            string key = parse_string();
            skip_ws();
            if (i < s.size() && s[i] == ':') i++;
            JsonValue val = parse_value();
            v.obj.push_back({key, move(val)});
            skip_ws();
            if (i < s.size() && s[i] == ',') { i++; continue; }
            break;
        }
        skip_ws();
        if (i < s.size() && s[i] == '}') i++;
        return v;
    }

    JsonValue parse_array() {
        JsonValue v; v.type = JsonValue::Array;
        i++;
        skip_ws();
        if (i < s.size() && s[i] == ']') { i++; return v; }
        while (true) {
            JsonValue val = parse_value();
            v.arr.push_back(move(val));
            skip_ws();
            if (i < s.size() && s[i] == ',') { i++; continue; }
            break;
        }
        skip_ws();
        if (i < s.size() && s[i] == ']') i++;
        return v;
    }

    static void append_utf8(string& out, unsigned int cp) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    string parse_string() {
        i++;
        string out;
        while (i < s.size() && s[i] != '"') {
            char c = s[i];
            if (c == '\\' && i + 1 < s.size()) {
                char e = s[i + 1];
                switch (e) {
                    case '"': out += '"'; i += 2; break;
                    case '\\': out += '\\'; i += 2; break;
                    case '/': out += '/'; i += 2; break;
                    case 'n': out += '\n'; i += 2; break;
                    case 't': out += '\t'; i += 2; break;
                    case 'r': out += '\r'; i += 2; break;
                    case 'b': out += '\b'; i += 2; break;
                    case 'f': out += '\f'; i += 2; break;
                    case 'u': {
                        unsigned int cp = (unsigned int)stoul(s.substr(i + 2, 4), nullptr, 16);
                        i += 6;
                        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                            unsigned int lo = (unsigned int)stoul(s.substr(i + 2, 4), nullptr, 16);
                            i += 6;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: out += e; i += 2; break;
                }
            } else {
                out += c;
                i++;
            }
        }
        if (i < s.size()) i++;
        return out;
    }

    JsonValue parse_string_value() {
        JsonValue v; v.type = JsonValue::String;
        v.str = parse_string();
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v; v.type = JsonValue::Bool;
        if (s[i] == 't') { v.b = true; i += 4; }
        else { v.b = false; i += 5; }
        return v;
    }

    JsonValue parse_number() {
        size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) i++;
        while (i < s.size() && (isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) i++;
        JsonValue v; v.type = JsonValue::Number;
        v.num = stod(s.substr(start, i - start));
        return v;
    }
};

inline JsonValue parse_json(const string& s) {
    JsonParser p(s);
    return p.parse();
}
