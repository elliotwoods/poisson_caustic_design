#ifndef SOLVER_H
#define SOLVER_H

#include <thread>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>

void poisson_solver(cv::Mat &D, cv::Mat &phi, int width, int height, int max_iterations, double convergence_threshold, int max_threads);

extern double solver_progress;

#endif // SOLVER_H