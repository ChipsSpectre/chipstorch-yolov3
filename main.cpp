#include <cmath>

#include <torch/torch.h>

// Define a new Module.
struct Net : torch::nn::Module {
    Net() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(3072, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 3072})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

/**
 * Samples f(x) for f(x) = x * x with x discrete, starting from 0 and distance 0.01 between samples.
 */
class DummyData : public torch::data::Dataset<DummyData> {
public:
    DummyData() : _data(torch::full({10000}, 50, torch::kFloat32)),
                  _target(torch::ones({10000}, torch::kLong)) {
    }

    torch::data::Example<> get(size_t index) override {
        return {_data[index], _target[index]};
    }

    c10::optional<size_t> size() const override {
        return 10000;
    }
private:
    torch::Tensor _data;
    torch::Tensor _target;
};

class Cifar10Data : public torch::data::Dataset<Cifar10Data> {
private:
    int count = 10000;
    int kImageRows = 32;
    int kImageColumns = 32;
    int kColorChannels = 3;
    int ENTRY_LENGTH = 3073;

    std::string join_paths(std::string head, const std::string& tail) {
        if (head.back() != '/') {
            head.push_back('/');
        }
        head += tail;
        return head;
    }

    std::vector<char> readoutFile(std::string full_path) {
        std::ifstream images(full_path, std::ios::binary);
        AT_CHECK(images, "Error opening images file at ", full_path);
        // get the starting position
        std::streampos start = images.tellg();

        // go to the end
        images.seekg(0, std::ios::end);

        // get the ending position
        std::streampos end = images.tellg();

        // go back to the start
        images.seekg(0, std::ios::beg);

        // create a vector to hold the data that
        // is resized to the total size of the file
        std::vector<char> contents;
        contents.resize(static_cast<size_t>(end - start));

        // read it in
        images.read(&contents[0], contents.size());

        return contents;
    }

    torch::Tensor readImages(std::string path, bool isTraining) {
        const auto full_path = join_paths(path, "cifar10-bin/data_batch_1.bin");

        auto content = readoutFile(full_path);

        auto tensor =
                torch::empty({count, ENTRY_LENGTH}, torch::kByte);
        std::ifstream images(full_path, std::ios::binary);
        AT_CHECK(images, "Error opening images file at ", full_path);
        images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());

        auto idx = torch::empty({ENTRY_LENGTH-1}, torch::kLong);
        for(int i = 0; i<ENTRY_LENGTH-1; i++) {
            idx[i] = i;
        }
        tensor = tensor.index_select(1, idx);
        return tensor.reshape({count, kColorChannels, kImageRows, kImageColumns}).to(torch::kFloat32).div_(255);
    }

    torch::Tensor readTargets(std::string path, bool isTraining) {
        const auto full_path = join_paths(path, "cifar10-bin/data_batch_1.bin");

        auto tensor =
                torch::empty({count, ENTRY_LENGTH}, torch::kByte);
        std::ifstream images(full_path, std::ios::binary);
        AT_CHECK(images, "Error opening images file at ", full_path);
        images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());

        auto idx = torch::full({1}, 0, torch::kLong);
        tensor = tensor.index_select(1, idx);
        return tensor.reshape({count}).to(torch::kLong);
    }
public:
    Cifar10Data(std::string path)
        : _images(readImages(path, true)),
          _targets(readTargets(path, true))
    {

    }

    torch::data::Example<> get(size_t index) override {
        return {_images[index], _targets[index]};
    }

    c10::optional<size_t> size() const override {
        return count;
    }
private:
    torch::Tensor _images, _targets;
};

int main() {
    // Create a new Net.
    auto net = std::make_shared<Net>();

    // Create a multi-threaded data loader for the MNIST dataset.
    std::string CIFAR10_PATH = "../data/";
    auto dataLoaderInt = Cifar10Data(CIFAR10_PATH).map(
            torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
            dataLoaderInt,
            /*batch_size=*/64
            );

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }
}
