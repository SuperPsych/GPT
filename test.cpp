#include "src/gpt.h"

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

    cout << "\nSample generation (greedy decode):" << endl;
    gpt.reset_cache();
    string prompt = "User: Explain how to do data analysis. \nAssistant:";
    auto prompt_ids = gpt.tokenize(prompt);
    vector<int> generated = prompt_ids;
    vector<double> logits;
    for (int id : prompt_ids) logits = gpt.generate_step(id);
    for (int step = 0; step < 40; step++) {
        int best = 0;
        double best_val = -1e18;
        for (int v = 0; v < gpt.vocab_size; v++) {
            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
        }
        generated.push_back(best);
        logits = gpt.generate_step(best);
    }
    cout << gpt.detokenize(generated) << endl;
}
