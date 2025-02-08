#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "solver.h"

double solver_progress = 0.0f;

// perform a relaxation step
double patial_relax(cv::Mat &output, cv::Mat &input, int width, int height, double omega) {
    cv::Mat neighbor_sum = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat neighbor_count = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat delta = cv::Mat::zeros(height, width, CV_64F);

    // Compute neighbor sums and counts
    cv::add(output(cv::Range(0, height), cv::Range(1, width)), output(cv::Range(0, height), cv::Range(0, width - 1)), neighbor_sum(cv::Range(0, height), cv::Range(1, width - 1)), cv::noArray(), CV_64F);
    cv::add(output(cv::Range(1, height), cv::Range(0, width)), output(cv::Range(0, height - 1), cv::Range(0, width)), neighbor_sum(cv::Range(1, height - 1), cv::Range(0, width)), cv::noArray(), CV_64F);

    cv::add(neighbor_count(cv::Range(0, height), cv::Range(1, width)), 1.0, neighbor_count(cv::Range(0, height), cv::Range(1, width)), cv::noArray(), CV_64F);
    cv::add(neighbor_count(cv::Range(0, height), cv::Range(0, width - 1)), 1.0, neighbor_count(cv::Range(0, height), cv::Range(0, width - 1)), cv::noArray(), CV_64F);
    cv::add(neighbor_count(cv::Range(1, height), cv::Range(0, width)), 1.0, neighbor_count(cv::Range(1, height), cv::Range(0, width)), cv::noArray(), CV_64F);
    cv::add(neighbor_count(cv::Range(0, height - 1), cv::Range(0, width)), 1.0, neighbor_count(cv::Range(0, height - 1), cv::Range(0, width)), cv::noArray(), CV_64F);

    // Avoid division by zero
    cv::Mat valid_neighbors = neighbor_count > 0;
    cv::divide(neighbor_sum, neighbor_count, neighbor_sum, 1, CV_64F);

    // Compute delta
    delta = omega * (neighbor_sum - output - input);

    // Update output
    output.setTo(output + delta, valid_neighbors);

    // Return max delta for convergence checking
    double max_delta;
    cv::minMaxLoc(cv::abs(delta), nullptr, &max_delta);
    return max_delta;
}

void calculate_progress(int value, int minValue, int maxValue) {
    const int barWidth = 50;

    // Calculate the percentage completion
    solver_progress = static_cast<double>(value - minValue) / (maxValue - minValue);
}

void poisson_solver(cv::Mat &input, cv::Mat &output, int width, int height, int max_iterations, double convergence_threshold) {
    double omega = 2.0 / (1.0 + 3.14159265 / width);

    double initial_max_update = 0.0f;

    for (int i = 0; i < max_iterations; i++) {
        double max_update = 0.0;

        // Perform a relaxation step
        max_update = patial_relax(output, input, width, height, omega);

        if (i == 0) {
            initial_max_update = max_update;
        }

        // Print progress to terminal
        if (i % 100 == 0) {
            printf("\33[2K\r");
            printf("Solver max_update: %f, convergence at %.2e\r", log(1.0f / max_update), convergence_threshold);

            calculate_progress(log(1.0f / max_update), log(1.0f / initial_max_update), log(1.0f / convergence_threshold));
        }

        // Check for convergence
        if (max_update < convergence_threshold) {
            printf("\nConverged at iteration %d\n", i);
            break;
        }
    }
}
