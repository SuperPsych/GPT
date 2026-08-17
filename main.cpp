#include "src/gpt.h"

int main() {
    string dataset_path = "data/smoltalk_train_pages.jsonl";
    string vocab_path = "data/vocab.txt";
    string merges_path = "data/merges.txt";
    string tokenizer_corpus_path = "data/tokenizer_corpus.txt";
    string checkpoint_path = "data/checkpoint.txt";
    string examples_cache_path = "data/examples_cache.bin";

    int num_layers = 2, num_heads = 2, dim_head = 32, dim_model = 64, hidden_dim = 192;
    int target_vocab = 2000, max_seq_len = 512;
    int num_examples = 500000, epochs = 4;
    int batch_size = 16;
    double lr = 3e-4;

    int requested_examples = num_examples;

    Tokenizer tok;
    vector<ChatExample> examples;
    {
        vector<vector<ChatTurn>> conversations;
        bool conversations_loaded = false;

        ifstream vocab_check(vocab_path);
        if (vocab_check.good()) {
            cout << "Loading existing tokenizer from " << vocab_path << "..." << endl;
            tok.load(vocab_path, merges_path);
        } else {
            cout << "Loading conversations from " << dataset_path << "..." << endl;
            conversations = ChatDataset::load_conversations(dataset_path);
            conversations_loaded = true;
            cout << "Loaded " << conversations.size() << " conversations." << endl;
            if (num_examples > (int)conversations.size()) num_examples = (int)conversations.size();

            cout << "Training tokenizer on " << num_examples << " conversations..." << endl;
            ofstream corpus_out(tokenizer_corpus_path);
            for (int i = 0; i < num_examples; i++) for (auto& turn : conversations[i]) corpus_out << turn.content << "\n";
            corpus_out.close();
            tok.train(tokenizer_corpus_path, target_vocab, 3, vocab_path, merges_path);
        }
        cout << "Tokenizer vocab size: " << tok.vocab_size() << endl;

        bool have_examples = ChatDataset::load_examples_cache(examples_cache_path, examples,
                                                                tok.vocab_size(), max_seq_len, requested_examples);
        if (have_examples) {
            cout << "Loaded " << examples.size() << " training examples from cache (" << examples_cache_path << ")." << endl;
        } else {
            if (!conversations_loaded) {
                cout << "Loading conversations from " << dataset_path << "..." << endl;
                conversations = ChatDataset::load_conversations(dataset_path);
                conversations_loaded = true;
                cout << "Loaded " << conversations.size() << " conversations." << endl;
                if (num_examples > (int)conversations.size()) num_examples = (int)conversations.size();
            }
            examples = ChatDataset::build_examples(conversations, tok, max_seq_len, num_examples);
            cout << "Built " << examples.size() << " training examples." << endl;
            ChatDataset::save_examples_cache(examples_cache_path, examples, tok.vocab_size(), max_seq_len, requested_examples);
            cout << "Cached training examples to " << examples_cache_path << endl;
        }
    }

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
        if (gpt.max_seq_len != max_seq_len) {
            cerr << "Fatal: checkpoint max_seq_len (" << gpt.max_seq_len
                 << ") does not match the configured max_seq_len (" << max_seq_len
                 << "). Training examples were built for a different sequence length than "
                    "this checkpoint's architecture." << endl;
            return 1;
        }
    }

    gpt.train(examples, epochs, lr, batch_size, checkpoint_path, 8192, 1024);

    gpt.save_checkpoint(checkpoint_path);
    cout << "Final checkpoint saved to " << checkpoint_path << endl;

    cout << "\nSample generation (greedy decode):" << endl;
    gpt.reset_cache();
    string prompt = "User: hello\nAssistant:";
    auto prompt_ids = gpt.tokenize(prompt);
    vector<int> generated = prompt_ids;
    vector<float> logits;
    for (int id : prompt_ids) logits = gpt.generate_step(id);
    for (int step = 0; step < 40; step++) {
        int best = 0;
        float best_val = -1e18f;
        for (int v = 0; v < gpt.vocab_size; v++) {
            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
        }
        generated.push_back(best);
        logits = gpt.generate_step(best);
    }
    cout << gpt.detokenize(generated) << endl;
}
