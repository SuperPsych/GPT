#include "matrix.h"

struct AttentionHead {
    Matrix W_q, W_k, W_v;
    Matrix Q, K, V;
    AttentionHead(int dim_head, int dim_model) : W_q(dim_head, dim_model), W_k(dim_head, dim_model), W_v(dim_head, dim_model) {
        W_q.randomize();
        W_k.randomize();
        W_v.randomize();
    }
    void add_token(){
        
    }
};