#include <cmath>

#include <torch/torch.h>

/**
 * Simplistic net for cifar10.
 */
class NetImpl : public torch::nn::Module {
public:
    NetImpl() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(3072, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 3072}))); // reshape to [3,32,32], flattened
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
TORCH_MODULE(Net);

/**
 * Implementation of a cifar10 data loader.
 */
class Cifar10Data : public torch::data::Dataset<Cifar10Data> {
private:
    // number of training images
    int countTrain = 50000;
    // number of test images
    int countTest = 10000;
    int kImageRows = 32;
    int kImageColumns = 32;
    int kColorChannels = 3;
    int ENTRY_LENGTH = 3073;
    int countBatch = 10000;

    /**
     * Joins two paths. They are composed such that the return
     * value can be used for path specifications.
     * @param head - first part
     * @param tail - new, second part
     * @return combination of first and second part.
     */
    std::string join_paths(std::string head, const std::string& tail) {
        if (head.back() != '/') {
            head.push_back('/');
        }
        head += tail;
        return head;
    }

    /**
     * Loads a tensor from a binary cifar10 batch file.
     *
     * Contains all data, i.e. images and targets of the batch.
     *
     * @param full_path - full path to the filename of the batch file.
     * @return a tensor that contains count_batch x ENTRY_LENGTH items
     */
    torch::Tensor loadTensorFromBatch(const std::string& full_path) {
        auto tensor =
                torch::empty({countBatch, ENTRY_LENGTH}, torch::kByte);
        std::ifstream images(full_path, std::ios::binary);
        AT_CHECK(images, "Error opening images file at ", full_path);
        images.read(reinterpret_cast<char *>(tensor.data_ptr()), tensor.numel());
        return tensor;
    }

    torch::Tensor readImages(const std::string& path, bool isTraining) {
        if(isTraining) {
            std::vector<std::string> paths = {"cifar10-bin/data_batch_1.bin", "cifar10-bin/data_batch_2.bin",
                                              "cifar10-bin/data_batch_3.bin", "cifar10-bin/data_batch_4.bin",
                                              "cifar10-bin/data_batch_5.bin"};
            auto trainTensor = torch::empty({5, countBatch, ENTRY_LENGTH - 1});
            for(int i = 0; i<paths.size(); i++) {
                auto currPath = paths[i];
                auto currTensor = loadTensorFromBatch(join_paths(path, currPath));

                auto currIdx = torch::empty({ENTRY_LENGTH - 1}, torch::kLong);
                for (int j = 0; j < ENTRY_LENGTH - 1; j++) {
                    currIdx[j] = j + 1;
                }
                currTensor = currTensor.index_select(1, currIdx);

                trainTensor[i] = currTensor;
            }
            trainTensor = trainTensor.reshape({countTrain, ENTRY_LENGTH - 1});
            return trainTensor.reshape({countTrain, kColorChannels, kImageRows, kImageColumns}).to(torch::kFloat32).div_(255);
        } else {
            auto tensor = loadTensorFromBatch(join_paths(path, "cifar10-bin/test_batch.bin"));

            auto idx = torch::empty({ENTRY_LENGTH - 1}, torch::kLong);
            for (int i = 0; i < ENTRY_LENGTH - 1; i++) {
                idx[i] = i + 1;
            }
            tensor = tensor.index_select(1, idx);
            return tensor.reshape({countTest, kColorChannels, kImageRows, kImageColumns}).to(torch::kFloat32).div_(255);
        }
    }

    /**
     * Reads the targets.
     * @param path - path to the cifar10 dataset
     * @param isTraining - specifies if training should be activated or not
     * @return tensor containing all required target labels, 1-dimensional
     */
    torch::Tensor readTargets(const std::string& path, bool isTraining) {
        if(isTraining) {
            std::vector<std::string> paths = {"cifar10-bin/data_batch_1.bin", "cifar10-bin/data_batch_2.bin",
                                              "cifar10-bin/data_batch_3.bin", "cifar10-bin/data_batch_4.bin",
                                              "cifar10-bin/data_batch_5.bin"};
            auto trainTensor = torch::empty({5, countBatch});
            for(int i = 0; i<paths.size(); i++) {
                auto currPath = paths[i];
                auto currTensor = loadTensorFromBatch(join_paths(path, currPath));

                auto idx = torch::full({1}, 0, torch::kLong);
                trainTensor[i]  = currTensor.index_select(1, idx).reshape({countBatch});
            }

            return trainTensor.reshape({countTrain}).to(torch::kLong);
        } else {
            auto tensor = loadTensorFromBatch(join_paths(path, "cifar10-bin/test_batch.bin"));

            auto idx = torch::full({1}, 0, torch::kLong);
            tensor = tensor.index_select(1, idx);
            return tensor.reshape({countTest}).to(torch::kLong);
        }
    }
public:
    Cifar10Data(const std::string& path, bool isTraining)
        : _isTraining(isTraining),
          _images(readImages(path, isTraining)),
          _targets(readTargets(path, isTraining))
    {

    }

    torch::data::Example<> get(size_t index) override {
        return {_images[index], _targets[index]};
    }

    c10::optional<size_t> size() const override {
        if(_isTraining) {
            return countTrain;
        }
        return countTest;
    }
private:
    bool _isTraining;
    torch::Tensor _images, _targets;
};

int main() {
    // Create a new Net.
    auto net = Net();
    // enable if you have pretrained model.
    // torch::load(net, "net.pt");

    // Create a multi-threaded data loader for the Cifar10 dataset.
    std::string CIFAR10_PATH = "../data/";
    auto dataLoaderInt = Cifar10Data(CIFAR10_PATH, true).map(
            torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
            dataLoaderInt,
            /*batch_size=*/64
            );

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    // train
    for (size_t epoch = 1; epoch <= 100; ++epoch) {
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

    // test
    auto testDataLoader = torch::data::make_data_loader(
            Cifar10Data(CIFAR10_PATH, false).map(torch::data::transforms::Stack<>()),
            /*batch_size=*/64
    );
    for (auto& batch : *data_loader) {
        torch::Tensor prediction = net->forward(batch.data);
        for(int pos = 0; pos < 9; pos++) {
            torch::save(batch.data[pos], torch::str("image", pos, ".pt"));
            torch::save(batch.target[pos], torch::str("target", pos, ".pt"));
            torch::save(prediction[pos], torch::str("prediction", pos, ".pt"));
        }
        break;
    }
}
