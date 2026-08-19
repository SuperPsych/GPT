#include "src/gpt.h"
#include "src/config.h"

int main() {
    string dataset_path = "data/smoltalk_train_pages.jsonl";
    string vocab_path = "data/vocab.txt";
    string merges_path = "data/merges.txt";
    string tokenizer_corpus_path = "data/tokenizer_corpus.txt";
    string checkpoint_path = "data/checkpoint.txt";
    string examples_cache_path = "data/examples_cache.bin";

    int num_layers = NUM_LAYERS, num_heads = NUM_HEADS, dim_head = DIM_HEAD;
    int dim_model = DIM_MODEL, hidden_dim = HIDDEN_DIM;
    int target_vocab = TARGET_VOCAB, max_seq_len = MAX_SEQ_LEN;
    int num_examples = NUM_EXAMPLES, epochs = EPOCHS;
    int batch_size = BATCH_SIZE;

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

    AdamState resume_opt(num_layers, num_heads, dim_head, dim_model, hidden_dim, tok.vocab_size());
    bool resumed = false;

    ifstream ckpt_check(checkpoint_path);
    if (ckpt_check.good()) {
        cout << "Loading checkpoint from " << checkpoint_path << "..." << endl;
        gpt = GPT::load_checkpoint(checkpoint_path, &resume_opt);
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
        if (gpt.num_layers != num_layers || gpt.num_heads != num_heads || gpt.dim_head != dim_head ||
            gpt.dim_model != dim_model || gpt.hidden_dim != hidden_dim) {
            cerr << "Fatal: checkpoint architecture (layers=" << gpt.num_layers << " heads=" << gpt.num_heads
                 << " dim_head=" << gpt.dim_head << " dim_model=" << gpt.dim_model << " hidden=" << gpt.hidden_dim
                 << ") does not match the configured architecture (layers=" << num_layers << " heads=" << num_heads
                 << " dim_head=" << dim_head << " dim_model=" << dim_model << " hidden=" << hidden_dim
                 << "). Move or delete the checkpoint to start a fresh run with this config." << endl;
            return 1;
        }
        resumed = true;
    }

    gpt.train(examples, epochs, batch_size, checkpoint_path, CHECKPOINT_EVERY, LOG_EVERY,
              resumed ? &resume_opt : nullptr);
}
