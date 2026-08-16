#include "src/gpt.h"

int main() {
    string dataset_path = "data/smoltalk_train_pages.jsonl";
    string vocab_path = "data/vocab.txt";
    string merges_path = "data/merges.txt";
    string tokenizer_corpus_path = "data/tokenizer_corpus.txt";
    string checkpoint_path = "data/checkpoint.txt";

    int num_layers = 2, num_heads = 2, dim_head = 32, dim_model = 64, hidden_dim = 192;
    int target_vocab = 2000, max_seq_len = 512;
    int num_examples = 500000, epochs = 1;
    int batch_size = 16;
    double lr = 3e-4;

    cout << "Loading conversations from " << dataset_path << "..." << endl;
    auto conversations = ChatDataset::load_conversations(dataset_path);
    cout << "Loaded " << conversations.size() << " conversations." << endl;

    if (num_examples > (int)conversations.size()) num_examples = (int)conversations.size();
    vector<vector<ChatTurn>> slice(conversations.begin(), conversations.begin() + num_examples);

    Tokenizer tok;
    ifstream vocab_check(vocab_path);
    if (vocab_check.good()) {
        cout << "Loading existing tokenizer from " << vocab_path << "..." << endl;
        tok.load(vocab_path, merges_path);
    } else {
        cout << "Training tokenizer on " << num_examples << " conversations..." << endl;
        ofstream corpus_out(tokenizer_corpus_path);
        for (auto& conv : slice) for (auto& turn : conv) corpus_out << turn.content << "\n";
        corpus_out.close();
        tok.train(tokenizer_corpus_path, target_vocab, 3, vocab_path, merges_path);
    }
    cout << "Tokenizer vocab size: " << tok.vocab_size() << endl;

    GPT gpt(num_layers, num_heads, dim_head, dim_model, hidden_dim, tok.vocab_size(), max_seq_len);
    gpt.tokenizer = tok;

    ifstream ckpt_check(checkpoint_path);
    if (ckpt_check.good()) {
        cout << "Loading checkpoint from " << checkpoint_path << "..." << endl;
        gpt = GPT::load_checkpoint(checkpoint_path);
        gpt.tokenizer = tok;
        if (gpt.vocab_size != tok.vocab_size()) {
            cerr << "Fatal: checkpoint vocab_size (" << gpt.vocab_size
                 << ") does not match tokenizer vocab_size (" << tok.vocab_size()
                 << "). The checkpoint was trained with a different vocab.txt/merges.txt." << endl;
            return 1;
        }
    }

    auto examples = ChatDataset::build_examples(slice, gpt.tokenizer, max_seq_len);
    cout << "Built " << examples.size() << " training examples." << endl;

    gpt.train(examples, epochs, lr, batch_size, checkpoint_path, /*checkpoint_every=*/200, /*log_every=*/64);

    gpt.save_checkpoint(checkpoint_path);
    cout << "Final checkpoint saved to " << checkpoint_path << endl;

    cout << "\nSample generation (greedy decode):" << endl;
    gpt.reset_cache();
    string prompt = "User: hello\nAssistant:";
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
