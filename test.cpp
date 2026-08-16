#include "src/gpt.h"
#include <random>

static int sample_token(const vector<float>& logits, double temperature, int top_k, mt19937& rng) {
    int vocab_size = (int)logits.size();
    int k = min(top_k, vocab_size);

    vector<int> idx(vocab_size);
    iota(idx.begin(), idx.end(), 0);
    partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });
    idx.resize(k);

    vector<double> top_probs(k);
    for (int i = 0; i < k; i++) top_probs[i] = logits[idx[i]] / temperature;
    FFN::output_activation(top_probs);

    discrete_distribution<int> dist(top_probs.begin(), top_probs.end());
    return idx[dist(rng)];
}

int main() {
    string vocab_path = "data/vocab.txt";
    string merges_path = "data/merges.txt";
    string checkpoint_path = "data/checkpoint.txt";

    Tokenizer tok;
    tok.load(vocab_path, merges_path);
    cout << "Tokenizer vocab size: " << tok.vocab_size() << endl;

    cout << "Loading checkpoint from " << checkpoint_path << "..." << endl;
    GPT gpt = GPT::load_checkpoint(checkpoint_path);
    gpt.tokenizer = tok;
    if (gpt.vocab_size != tok.vocab_size()) {
        cerr << "Fatal: checkpoint vocab_size (" << gpt.vocab_size
             << ") does not match tokenizer vocab_size (" << tok.vocab_size()
             << "). The checkpoint was trained with a different vocab.txt/merges.txt." << endl;
        return 1;
    }

    int eot_id = tok.token_to_id.count("<|endoftext|>") ? tok.token_to_id.at("<|endoftext|>") : -1;
    int max_new_tokens = 200;
    double temperature = 0.8;
    int top_k = 10;
    mt19937 rng(random_device{}());

    cout << "\nInteractive mode. Type a message and press Enter (empty line, \"exit\", or \"quit\" to stop).\n" << endl;

    string line;
    while (true) {
        cout << "> ";
        if (!getline(cin, line)) break;
        if (line.empty() || line == "exit" || line == "quit") break;

        gpt.reset_cache();
        string prompt = "User: " + line + "\nAssistant:";
        auto prompt_ids = gpt.tokenize(prompt);

        vector<int> generated;
        vector<float> logits;
        for (int id : prompt_ids) logits = gpt.generate_step(id);
        for (int step = 0; step < max_new_tokens; step++) {
            int next = sample_token(logits, temperature, top_k, rng);
            if (next == eot_id) break;
            generated.push_back(next);
            logits = gpt.generate_step(next);
        }
        cout << gpt.detokenize(generated) << "\n" << endl;
    }
    return 0;
}
