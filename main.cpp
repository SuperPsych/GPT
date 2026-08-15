#include "src/mlp.h"
#include "src/data_loader.h"

int main() {

    int input_dim = 784;
    int output_dim = 10;
    int hidden_dim = 128;
    int hidden_num = 1;

    auto mlp = MLP::initialize(input_dim, output_dim, hidden_dim, hidden_num);

    auto train_samples = load_samples("./mnist_train.txt", input_dim, output_dim);
    auto test_samples = load_samples("./mnist_test.txt", input_dim, output_dim);

    int epochs = 30;
    int batch_size = 32;
    double lr = 0.01;
    mlp.train(train_samples, test_samples, epochs, batch_size, lr);

    mlp.eval(test_samples);
}
