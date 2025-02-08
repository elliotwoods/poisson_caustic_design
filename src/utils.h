#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "polygon_utils.h"

void subtractAverage(cv::Mat& raster);
cv::Mat scale_matrix_proportional(const cv::Mat& matrix, double min_value, double max_value);
std::vector<double> scale_array_proportional(const std::vector<double>& arr, double min_value, double max_value);

void calculate_errors(std::vector<double> &source_areas, std::vector<double> &target_areas, const std::vector<std::vector<cv::Point2d>>& cells, std::vector<double> &errors);

// Vector math
std::vector<cv::Mat> calculate_gradient(const cv::Mat& grid);
cv::Mat calculate_divergence(const cv::Mat& Nx, const cv::Mat& Ny);

// SVG export
void export_cells_as_svg(const std::vector<std::vector<cv::Point2d>>& cells, const std::vector<double>& intensities, const std::string& filename);
void export_grid_to_svg(const std::vector<cv::Point2d>& points, double width, double height, int res_x, int res_y, const std::string& filename, double stroke_width);
void export_triangles_to_svg(const std::vector<cv::Point2d>& points, const std::vector<std::vector<int>>& triangles, double width, double height, int res_x, int res_y, const std::string& filename, double stroke_width);

// 3d export
void save_solid_obj(const std::vector<cv::Point3d>& front_points, const std::vector<cv::Point3d>& back_points, const std::vector<std::vector<int>>& triangles, double thickness, double width, double height, int res_x, int res_y, const std::string& filename);

// linear algebra
std::vector<double> vector_subtract(const std::vector<double>& p1, const std::vector<double>& p2);
std::vector<double> cross_product(const std::vector<double>& p1, const std::vector<double>& p2);
double dot_product(const std::vector<double>& p1, const std::vector<double>& p2);

std::vector<double> normalize(std::vector<double> p1);

void calculate_angle_and_normal_from_triangle(std::vector<double> &p1, std::vector<double> &p2, std::vector<double> &p3, std::vector<double> &normal_out, double &angle_out);

#endif