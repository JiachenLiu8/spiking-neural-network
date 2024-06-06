#include <vector>
#include <cmath>
#include <random>
#include <iostream>

// use use surrogate gradient descent to solve the sound localization problem.

class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;

   // Default constructor
    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}

    // Constructor with random initialization option
    Matrix(int r, int c, bool randomize) : rows(r), cols(c), data(r, std::vector<double>(c)) {
        if (randomize) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0); // Uniform distribution between -1.0 and 1.0

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    data[i][j] = dis(gen); // Randomly initialize the matrix elements
                }
            }
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    data[i][j] = 0.0; // Initialize the matrix elements to zero
                }
            }
        }
    }

    static Matrix random(int r, int c) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);

        Matrix m(r, c);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                m.data[i][j] = d(gen) / std::sqrt(c);  // Xavier initialization
            }
        }
        return m;
    }

    Matrix applyFunction(std::function<double(double)> func) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix trans(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                trans.data[j][i] = data[i][j];
        return trans;
    }

    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows);
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // Method to multiply matrix by a scalar
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    // Apply a function to each element of the matrix
    Matrix applyFunction(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = func(data[i][j]);
        return result;
    }

    // Add two matrices
    Matrix operator+(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    // Subtraction of two matrices
    Matrix operator-(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }
};

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double surrogateGradient(double x) {
    const double beta = 2.0;  // steepness of the surrogate gradient
    return beta * exp(-beta * abs(x)) / (pow(1.0 + exp(-beta * abs(x)), 2));
}

class SpikingNeuralNetwork {
    Matrix W1, W2; // Weights for two layers
    double learningRate;

public:
    SpikingNeuralNetwork(int inputSize, int hiddenSize, int outputSize, double lr = 0.01)
        : W1(Matrix::random(inputSize, hiddenSize)),
          W2(Matrix::random(hiddenSize, outputSize)),
          learningRate(lr) {}

    // Forward propagation with surrogate gradient
    std::vector<Matrix> forward(const Matrix& inputs) {
        Matrix h = inputs * W1;
        Matrix hiddenSpikes = h.applyFunction([](double x) { return x > 0 ? 1 : 0; }); // Spiking function
        Matrix outputPotentials = hiddenSpikes * W2;
        Matrix outputSpikes = outputPotentials.applyFunction([](double x) { return x > 0 ? 1 : 0; }); // Output spikes
        return {hiddenSpikes, outputSpikes}; // Return all layer results for backpropagation
    }

    // Implement backpropagation using the recorded spikes and potentials
    void backpropagate(const Matrix& input, const Matrix& hiddenSpikes, const Matrix& outputSpikes, const Matrix& target) {
        Matrix outputError = target - outputSpikes;
        Matrix hiddenError = outputError * W2.transpose();

        // Gradient calculation with surrogate function applied
        Matrix hiddenGradient = hiddenError.applyFunction(surrogateGradient);
        Matrix inputGradient = hiddenGradient * W1.transpose();

        // Update weights
        W1 = W1 + (input.transpose() * hiddenGradient) * learningRate;
        W2 = W2 + (hiddenSpikes.transpose() * outputError) * learningRate;
    }
};

int main() {
    SpikingNeuralNetwork snn(100, 50, 10); // Example sizes: 100 input neurons, 50 hidden neurons, 10 output classes

    // Generate some fake data for demonstration purposes
    Matrix inputs(1, 100, true); // one sample, 100 features
    Matrix targets(1, 10);       // one sample, 10 target classes
    targets.data[0][5] = 1;      // example: class 5 is the correct class

    // Training loop
    for (int i = 0; i < 1000; ++i) {
        auto results = snn.forward(inputs);
        snn.backpropagate(inputs, results[0], results[1], targets);
    }

    // Check outputs
    auto outputs = snn.forward(inputs);
    std::cout << "Output spikes from the network: ";
    for (auto& val : outputs[1].data[0]) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
