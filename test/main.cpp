#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "CudaFeedforwardNeuralNetwork.h"
#include "ManagedMatrix.h"
#include "ApproxMatrixMatcher.h"
#include "TrainSettings.h"
#include "TrainResult.h"
#include <vector>
#include <cmath>
#include <random>
#include <sstream>

using dc::ManagedMatrix;
using std::vector;

constexpr double sigmoid(double t) {
	return 1.0 / (1 + std::exp(-t));
}

template <class TElem>
std::string to_string(const vector<TElem>& x) {
	std::ostringstream os;

	os << "[";
	for(int i = 0; i < x.size(); ++i) {
		if (i >= 1) {
			os << ' ';
		}

		os << x[i];
	}

	os << ']';

	return os.str();
}

TEST_CASE("can set up weights manually") {
	CudaFeedforwardNeuralNetwork network({2, 4, 1});

	ManagedMatrix<double> weights(4, 3);

	for(int i = 0; i < weights.get_rows(); ++i) {
		for (int j = 0; j < weights.get_columns(); ++j) {
			weights.set(i, j, 1.0);
		}
	}

	network.set_weights(0, weights);

	REQUIRE(network.get_weights(0) == weights);
}

void test_single_output_neural_network(const CudaFeedforwardNeuralNetwork& network,
		const vector<vector<double> >& inputs, const vector<double>& expected_outputs) {
	REQUIRE(inputs.size() == expected_outputs.size());

	for(int i = 0; i < inputs.size(); ++i) {
		CAPTURE(i);
		CAPTURE(to_string(inputs[i]));
		auto output = network.predict(inputs[i]);

		REQUIRE(output.size() == 1);
		REQUIRE(output[0] == Approx(expected_outputs[i]));
	}
}

void test_single_output_neural_network_approx(const CudaFeedforwardNeuralNetwork& network,
		const vector<vector<double> >& inputs, const vector<bool>& expected_outputs, double threshold = 0.8) {
	REQUIRE(inputs.size() == expected_outputs.size());

	for(int i = 0; i < inputs.size(); ++i) {
		CAPTURE(i);
		CAPTURE(to_string(inputs[i]));
		auto output = network.predict(inputs[i]);

		REQUIRE(output.size() == 1);
		if (expected_outputs[i]) {
			REQUIRE(output[0] >= threshold);
		} else {
			REQUIRE(output[0] < threshold);
		}

	}
}

TEST_CASE("performs forward-propagation correctly", "[forwardpropagation]") {
	SECTION("AND gate neural-network") {
		CudaFeedforwardNeuralNetwork network({2, 1});
		ManagedMatrix<double> weights(1, 3);
		weights.set_all_row_wise({
			-30, 20, 20
		});

		network.set_weights(0, weights);

		test_single_output_neural_network(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{sigmoid(-30), sigmoid(-10), sigmoid(-10), sigmoid(10)});
	}

	SECTION("OR gate neural-network") {
		CudaFeedforwardNeuralNetwork network({2, 1});
		ManagedMatrix<double> weights(1, 3);
		weights.set_all_row_wise({
			-10, 20, 20
		});

		network.set_weights(0, weights);

		test_single_output_neural_network(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{sigmoid(-10), sigmoid(10), sigmoid(10), sigmoid(30)});
	}

	SECTION("NOR gate neural-network") {
		CudaFeedforwardNeuralNetwork network({2, 1});
		ManagedMatrix<double> weights(1, 3);
		weights.set_all_row_wise({
			10, -20, -20
		});

		network.set_weights(0, weights);

		test_single_output_neural_network(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{sigmoid(10), sigmoid(-10), sigmoid(-10), sigmoid(-30)});
	}

	SECTION("XOR gate neural-network") {
		CudaFeedforwardNeuralNetwork network({2, 2, 1});

		ManagedMatrix<double> weights_0(2, 3);
		weights_0.set_all_row_wise({
			30, -20, -20,
			-10, 20, 20
		});

		ManagedMatrix<double> weights_1(1, 3);
		weights_1.set_all_row_wise({
			-30, 20, 20
		});

		network.set_weights(0, weights_0);
		network.set_weights(1, weights_1);

		test_single_output_neural_network_approx(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{false, true, true, false});
	}
}

TEST_CASE("performs back-propagation correctly", "[backpropagation]") {
	SECTION("simple neural network") {
		CudaFeedforwardNeuralNetwork network({2, 2, 1});

		ManagedMatrix<double> weights_0(2, 3);
		weights_0.set_all_row_wise({
			1, -5, -5,
			-5, 10, -10
		});

		ManagedMatrix<double> weights_1(1, 3);
		weights_1.set_all_row_wise({
			-5, 1, 1
		});

		network.set_weights(0, weights_0);
		network.set_weights(1, weights_1);

		vector<ManagedMatrix<double> > expected_deltas;
		expected_deltas.push_back(ManagedMatrix<double>(2, 3));
		expected_deltas[0].set_all_row_wise({
			-1.75423615e-02,  -0.00000000e+00,  -1.75423615e-02,
			-3.03817871e-07,  -0.00000000e+00,  -3.03817871e-07
		});

		expected_deltas.push_back(ManagedMatrix<double>(1, 3));
		expected_deltas[1].set_all_row_wise({
			-9.93186507e-01,  -1.78636610e-02,  -3.03817964e-07
		});

		vector<ManagedMatrix<double> > result = network.compute_weights_error({0, 1}, {1});

		REQUIRE(result.size() == expected_deltas.size());
		for(int i = 0; i < result.size(); ++i) {
			CAPTURE(i);
			REQUIRE_THAT(result[i], ApproxMatrixMatcher(expected_deltas[i]));
		}
	}
}

TEST_CASE("calculates the error correctly", "[error]") {
	SECTION("simple neural network") {
		CudaFeedforwardNeuralNetwork network({2, 2, 1});

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

		ManagedMatrix<double> weights_0(2, 3);
		weights_0.set_all_row_wise({
			1, -5, -5,
			-5, 10, -10
		});

		ManagedMatrix<double> weights_1(1, 3);
		weights_1.set_all_row_wise({
			-5, 1, 1
		});

		network.set_weights(0, weights_0);
		network.set_weights(1, weights_1);

		const double expected_error = 8.06265;

		SECTION("CPU error") {
			double actual_error = network.compute_error(sample_x, sample_y, 0.2);

			REQUIRE(actual_error == Approx(expected_error));
		}

		SECTION("GPU error") {
			int blocks = 2;
			int threads_per_block = 2;
			double actual_error = network.compute_error_gpu(sample_x, sample_y, 0.2, blocks, threads_per_block);

			REQUIRE(actual_error == Approx(expected_error));
		}
	}
}

TEST_CASE("trains correctly", "[train]") {
	SECTION("XOR neural network") {
		CudaFeedforwardNeuralNetwork network({2, 8, 1});

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

		std::random_device rd;
		std::mt19937 generator(rd());

		TrainSettings train_settings;
		train_settings.initialize_weights_randomly = true;
		train_settings.inner_steps = 16;
		train_settings.iterations = 2000;
		train_settings.regularization_term = 0.0;
		train_settings.momentum = 0.9;
		train_settings.step_factor = 1.0;
		train_settings.threads = 1000;
		train_settings.blocks = 15;
		train_settings.generator = &generator;
		train_settings.random_epsilon = 10.0;
		train_settings.target_error = 0.001;

		TrainResult result = network.train(sample_x, sample_y, train_settings);

		CAPTURE(result.error);

		test_single_output_neural_network_approx(network, { {0, 0}, {0, 1}, {1, 0}, {1, 1} },
				{0, 1, 1, 0});
	}
}
