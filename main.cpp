#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>

using namespace Eigen;

// Activation function and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Layer class
class Layer {
public:
    int num_inputs;
    int num_outputs;
    MatrixXd weights;
    VectorXd biases;

    Layer(int num_inputs, int num_outputs) 
        : num_inputs(num_inputs), num_outputs(num_outputs) {
        weights = MatrixXd::Random(num_outputs, num_inputs);
        biases = VectorXd::Random(num_outputs);
    }

    VectorXd forward(const VectorXd& inputs) {
        VectorXd outputs = (weights * inputs) + biases;
        for (int i = 0; i < outputs.size(); ++i) {
            outputs[i] = sigmoid(outputs[i]);
        }
        return outputs;
    }
};

// Neural Network class
class NeuralNetwork {
public:
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<int>& layer_sizes) {
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
        }
    }

    VectorXd predict(const VectorXd& inputs) {
        VectorXd output = inputs;
        for (Layer& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    void train(const std::vector<VectorXd>& X, const std::vector<VectorXd>& Y, int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < X.size(); ++i) {
                // Forward pass
                std::vector<VectorXd> activations = { X[i] };
                for (Layer& layer : layers) {
                    activations.push_back(layer.forward(activations.back()));
                }

                // Compute error
                VectorXd error = Y[i] - activations.back();

                // Backward pass
                for (int l = layers.size() - 1; l >= 0; --l) {
                    VectorXd delta = error.cwiseProduct(activations[l + 1].unaryExpr(&sigmoid_derivative));

                    VectorXd prev_activation = activations[l];
                    MatrixXd weight_gradient = delta * prev_activation.transpose();

                    layers[l].weights += learning_rate * weight_gradient;
                    layers[l].biases += learning_rate * delta;

                    if (l > 0) {
                        error = layers[l].weights.transpose() * delta;
                    }
                }
            }

            if (epoch % 1000 == 0) {
                std::cout << "Epoch " << epoch << " complete.\n";
            }
        }
    }
};

// Main function
int main() {
    // XOR dataset
    std::vector<VectorXd> X = {
        (VectorXd(2) << 0, 0).finished(),
        (VectorXd(2) << 0, 1).finished(),
        (VectorXd(2) << 1, 0).finished(),
        (VectorXd(2) << 1, 1).finished()
    };

    std::vector<VectorXd> Y = {
        (VectorXd(1) << 0).finished(),
        (VectorXd(1) << 1).finished(),
        (VectorXd(1) << 1).finished(),
        (VectorXd(1) << 0).finished()
    };

    // Create neural network with 2 inputs, one hidden layer with 2 neurons, and 1 output
    NeuralNetwork nn({ 2, 2, 1 });

    // Train the network
    nn.train(X, Y, 10000, 0.1);

    // Test the network
    for (const auto& x : X) {
        VectorXd prediction = nn.predict(x);
        std::cout << x.transpose() << " -> " << prediction[0] << std::endl;
    }

    return 0;
}

