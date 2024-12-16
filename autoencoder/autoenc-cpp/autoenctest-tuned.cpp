/*
The skeleton was based on the PyTorch documentation, but right now it is almost completely new code

This is the tuned version - it works significantly faster than the version that follows the practices from documentation.

Tadeusz Puźniakowski, 2024
*/

#include <chrono>
#include <numeric>
#include <iomanip>
#include <tuple>


#include <torch/torch.h>

#include "thirdparty/lodepng.h"

struct Net : torch::nn::Module {
    Net() {
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Conv2d(1, 32, torch::ExpandingArray < 2 > {3, 3}),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(2),
            torch::nn::Flatten(),
            torch::nn::Linear(13 * 13 * 32, 256),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.5),
            torch::nn::Linear(256, 32),
            torch::nn::ReLU(),
            torch::nn::Linear(32, 2),
            torch::nn::Tanh()
            ));
        decoder = register_module("decoder", torch::nn::Sequential(
            torch::nn::Linear(2, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 26*26*32),
            torch::nn::ReLU(),
            torch::nn::Unflatten(torch::nn::UnflattenOptions(1, {32, 26,26})),
            torch::nn::ConvTranspose2d(32,1,torch::ExpandingArray < 2 > {3,3})
            ));            
    }

    torch::Tensor forward(torch::Tensor x) {
        x = encoder->forward(x);
        x = decoder->forward(x);
        return x;
    }
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Sequential decoder{nullptr};
};

/**
 * @brief Splits a 1D vector into a 2D vector, where each inner vector has a specified length.
 *
 * This function takes a 1D vector and splits it into a 2D vector, with each inner vector having
 * a length of `l`. If the input vector's size is not divisible by `l`, an exception is thrown.
 *
 * @param v The input 1D vector to be split.
 * @param l The length of each inner vector in the resulting 2D vector.
 * @return A 2D vector, where each inner vector has a length of `l`.
 *
 * @throws std::invalid_argument If the size of the input vector is not divisible by `l`.
 *
 * @note This function is designed to work in the context of PyTorch workflows, where splitting
 *       data into fixed-size chunks is often required.
 *
 * @example
 * // Example usage:
 * std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
 * int chunk_size = 2;
 * auto result = split_vector(data, chunk_size);
 * // result: {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}
 */
std::vector<std::vector<float>> split_vector(std::vector<float> v, int l) {
    std::vector<std::vector<float>> ret;
    std::vector<float> inner;
    for (auto e : v) {
        inner.push_back(e);
        if (inner.size() == l) {
            ret.push_back(inner);
            inner.clear();
        }
    }
    if (inner.size() != 0) throw std::invalid_argument("The input vector cannot be divided by " + std::to_string(l));
    return ret;
}

auto argmax = [](auto v) -> std::size_t{
    return std::max_element(v.begin(), v.end()) - v.begin();
};

template<typename T>
std::vector<T> to_vector(torch::Tensor result) {
    auto sizes = result.sizes(); // Get tensor dimensions
    int size = [&](){int s = 1; for (auto x:sizes) s*=x;return s;}();
    auto ptr = result.to(torch::kCPU).contiguous().data_ptr<T>();
    return std::vector<T>(ptr, ptr+size);
}

auto train = [](auto epoch, auto &data_loader, auto &model, auto &loss_fn, auto &optimizer, auto &device) {
    size_t batch_index = 0;
    model->train();
    for (auto &batch: *data_loader) {
        auto &[x_batch, y_batch] = batch;
        torch::Tensor prediction = model->forward(x_batch);
        torch::Tensor loss = loss_fn(prediction, y_batch);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        std::cout << "[" << epoch << "] | Batch: " << (batch_index++)
                    << " | Loss: " << loss.item<float>() << "        \r";
        std::cout.flush();
    }
    std::cout << std::endl;
};

auto save_images = [](auto &net, auto &data_loader,auto &device){
    torch::InferenceMode guard(true);
    net->eval();
    for (auto &batch: *data_loader) {
        auto x_batch = batch.data.data().to(device);
        //net->to(device);
        //auto y_result = net->forward(x_batch);
        net->encoder->to(device);
        net->decoder->to(device);
        auto latent = net->encoder->forward(x_batch);
        auto y_result = net->decoder->forward(latent);
        std::vector<std::vector<float>> images_float = split_vector(to_vector<float>(x_batch),28*28);
        std::vector<std::vector<float>> y_result_vec = split_vector(to_vector<float>(y_result),28*28);
        std::vector<std::vector<float>> latent_vec = split_vector(to_vector<float>(latent),2);

        for (int i = 0; i < std::min((int)images_float.size(), (int)20); i++) {
            std::vector<unsigned char> image;
            for (int xx = 0; xx < images_float[i].size(); xx++) {
                image.push_back(std::max(0.0,images_float[i][xx]*128.0));
                image.push_back(std::max(0.0,images_float[i][xx]*255.0));
                image.push_back(std::max(0.0,images_float[i][xx]*128.0));
                image.push_back(255);
            }
            lodepng::encode("results/__ret_" + std::to_string(i) + + "_" + std::to_string(latent_vec[i][0]) + "x" + std::to_string(latent_vec[i][1]) +"_x.png", image, 28,28);
            image.clear();
            for (int xx = 0; xx < y_result_vec[i].size(); xx++) {
                image.push_back(std::max(0.0, y_result_vec[i][xx]*128.0));
                image.push_back(std::max(0.0, y_result_vec[i][xx]*255.0));
                image.push_back(std::max(0.0, y_result_vec[i][xx]*128.0));
                image.push_back(255);
            }
            lodepng::encode("results/__ret_" + std::to_string(i) + + "_" + std::to_string(latent_vec[i][0]) + "x" + std::to_string(latent_vec[i][1]) +"_y.png", image, 28,28);
        }
        break;
    }
};



int main() {
    std::map<std::string,std::function<torch::Tensor(const torch::Tensor&,const torch::Tensor&)>> loss_functions = {
        {"mse",[](const torch::Tensor &a,const torch::Tensor &b){return torch::mse_loss(a,b); }}

    };
    std::cout << "PyTorch: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "."<< TORCH_VERSION_PATCH << std::endl;
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available. Moving model to GPU." << std::endl;
    }
    // Create a new Net.
    auto net = std::make_shared<Net>();

    // Create a multi-threaded data loader for the MNIST dataset.
    auto mnist_training = torch::data::datasets::MNIST("./data/FashionMNIST/raw");
    int batch_size=10000;
    auto data_loader = torch::data::make_data_loader(
            mnist_training.map(
                    torch::data::transforms::Stack<>()),
            batch_size);
    auto batches = std::make_shared<std::vector<std::pair<torch::Tensor,torch::Tensor>>>() ;
    for (auto &batch: *data_loader) {
        torch::Tensor x_batch = batch.data.data().to(device);
        torch::Tensor y_batch = batch.data.data().to(device); //batch.target.data().to(device);
        batches->push_back(std::make_pair(x_batch, y_batch));
    }
    try {
        torch::load(net, "autoencoder.pt");
    } catch (...) {
        std::cout << "some error lading model" << std::endl;
    }
    net->to(device);
    double lr = 0.001;
    size_t epochs = 30;
    const auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        torch::optim::Adam optimizer(net->parameters(), lr);
        train(epoch, batches, net, loss_functions["mse"] , optimizer, device);
        lr = lr * 0.9;
        torch::save(net, "autoencoder.pt");
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end_time - start_time;
    std::cout << std::fixed << std::setprecision(9) << std::left;
    std::cout << "Training time[s]: " << diff.count() << '\n';

    save_images(net, data_loader ,device);
    return 0;
}
