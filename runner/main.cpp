#include <iostream>
#include <random>
#include "ManagedMatrix.h"
#include "TrainSettings.h"
#include "CudaFeedforwardNeuralNetwork.h"
#include "data-util.h"
#include "int-util.h"

using dc::ManagedMatrix;
using std::cout;

int main(int argc, const char* argv[])
{
    if (argc != 4) {
      cout << "Usage: ./Neural_networks threads iterations inner_steps\n";
      return -1;
    }

    int threads = atoi(argv[1]),
		iterations = atoi(argv[2]),
		inner_steps = atoi(argv[3]);

    std::random_device rd;
    std::mt19937 generator(rd());

    CudaFeedforwardNeuralNetwork network({2, 8, 1});
    //CudaFeedforwardNeuralNetwork network({3, 8, 1});

    /*MatrixXd weights(2, 3);
    weights << -30, 20, 20, 10, -20, -20;
    MatrixXd weights2(1, 3);
    weights2 << -10, 20, 20;
    network.set_weights(0, weights);
    network.set_weights(1, weights2);*/

    ManagedMatrix<double> sample_x(4, 2);
    sample_x.set_all_row_wise({
    	0, 0,
    	0, 1,
    	1, 0,
    	1, 1
    });

    sample_x = sample_x.get_transposed();


    ManagedMatrix<double> sample_y(4, 1);
    sample_y.set_all_row_wise({
    	0,
    	1,
    	1,
    	0
    });

    sample_y = sample_y.get_transposed();

    /*auto mnist_data = read_iris_database("iris.data");
    ManagedMatrix<double>& sample_x = mnist_data.first;
    ManagedMatrix<double>& sample_y = mnist_data.second;*/

    ManagedMatrix<double> test_sample_x = sample_x;
    ManagedMatrix<double> test_sample_y = sample_y;

    /*auto training_data = generate_data(1000, generator);
    auto test_data = generate_data(100, generator);
    ManagedMatrix<double> sample_x = training_data.first.get_transposed();
    ManagedMatrix<double> sample_y = training_data.second.get_transposed();
    ManagedMatrix<double> test_sample_x = test_data.first.get_transposed();
    ManagedMatrix<double> test_sample_y = test_data.second.get_transposed();*/


    TrainSettings train_settings;
    train_settings.threads = threads;
	train_settings.blocks = 10;
	train_settings.generator = &generator;
    train_settings.inner_steps = inner_steps;
    train_settings.iterations = iterations;
    train_settings.regularization_term = 0.0;
    train_settings.momentum = 0.9;
    train_settings.step_factor = 1.0;
    train_settings.random_epsilon = 10;
    train_settings.target_error = 0.00000001;

    /*train_settings.regularization_term = 0.1;
	train_settings.momentum = 0.6;
	train_settings.step_factor = 0.06;
	train_settings.random_epsilon = 10;
	train_settings.target_error = 0.001;*/

    auto result = network.train(sample_x, sample_y, train_settings);

    for(int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::vector<double> input = {(double)i, (double)j};
            auto result = network.predict(input);
            cout << "x: \n" << i << " " << j << "\n =>\n" << result[0] << '\n';
        }
    }

    int error_blocks = 1;
    int error_threads = int_division_round_up(sample_x.get_columns(), error_blocks);

    cout << "Actual iterations: " << result.iterations << '\n';

    cout << "Training error: " << network.compute_error_gpu(sample_x,
                                                        sample_y,
                                                        train_settings.regularization_term, error_blocks, error_threads) << "\n";
    cout << "Test error: " << network.compute_error_gpu(test_sample_x,
                                                    test_sample_y,
                                                    train_settings.regularization_term, error_blocks, error_threads) << "\n";
}
