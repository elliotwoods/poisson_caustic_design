#include "stdio.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "src/caustic_design.h"
#include <thread>

void resize_image(const cv::Mat& input_image, cv::Mat& output_image, int new_width, int new_height) {
    cv::resize(input_image, output_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
}

cv::Mat image_to_grid(const std::string& filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("Failed to open image file.");
    }

    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);
    } else {
        gray_image = image;
    }

    gray_image.convertTo(gray_image, CV_64F, 1.0 / 255.0);
    return gray_image;
}

void save_heightmap_as_json(const cv::Mat& heightmap, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open output file.");
    }

    outfile << "[\n";
    for (int i = 0; i < heightmap.rows; ++i) {
        outfile << "  [";
        for (int j = 0; j < heightmap.cols; ++j) {
            outfile << heightmap.at<double>(i, j);
            if (j < heightmap.cols - 1) {
                outfile << ", ";
            }
        }
        outfile << "]";
        if (i < heightmap.rows - 1) {
            outfile << ",\n";
        } else {
            outfile << "\n";
        }
    }
    outfile << "]\n";
    outfile.close();
}

std::unordered_map<std::string, std::string> parse_arguments(int argc, char const *argv[]) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key, value;
        if (arg.substr(0, 2) == "--") {
            size_t pos = arg.find('=');
            if (pos != std::string::npos) {
                key = arg.substr(2, pos - 2);
                value = arg.substr(pos + 1);
            } else {
                key = arg.substr(2);
                value = "";
            }
        } else if (arg[0] == '-') {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                key = arg.substr(1);
                value = argv[++i];
            } else {
                key = arg.substr(1);
                value = "";
            }
        }
        args[key] = value;
    }
    return args;
}

int main(int argc, char const *argv[]) {
    std::unordered_map<std::string, std::string> args = parse_arguments(argc, argv);

    cv::Mat pixels = image_to_grid(args["input_png"]);
    double aspect_ratio = static_cast<double>(pixels.cols) / static_cast<double>(pixels.rows);

    cv::Mat resized_pixels;
    resize_image(pixels, resized_pixels, 4 * std::stoi(args["res_w"]), 4 * std::stoi(args["res_w"]) / aspect_ratio);

    Caustic_design caustic_design;

    int mesh_resolution_x = std::stoi(args["res_w"]);
    double mesh_width = std::stod(args["width"]);

    caustic_design.set_mesh_resolution(mesh_resolution_x, mesh_resolution_x / aspect_ratio);
    caustic_design.set_domain_resolution(4 * mesh_resolution_x, 4 * mesh_resolution_x / aspect_ratio);

    double mesh_height = std::floor(mesh_resolution_x / aspect_ratio) * (mesh_width / mesh_resolution_x);

    caustic_design.set_mesh_size(mesh_width, mesh_height);

    caustic_design.set_lens_focal_length(std::stod(args["focal_l"]));
    caustic_design.set_lens_thickness(std::stod(args["thickness"]));
    caustic_design.set_solver_max_threads(std::stoi(args["max_threads"]));

    caustic_design.initialize_solvers(resized_pixels);

    caustic_design.export_paramererization_to_svg("../parameterization_0.svg", 0.5f);

    for (int itr = 0; itr < 1; itr++) {
        printf("starting iteration %i\r\n", itr);

        auto start = std::chrono::high_resolution_clock::now();
        double step_size = caustic_design.perform_transport_iteration();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("Transport iteration %d took %f seconds\n", itr, elapsed.count());

        caustic_design.export_paramererization_to_svg("../parameterization_" + std::to_string(itr + 1) + ".svg", 1.0f);
        caustic_design.export_inverted_transport_map("../inverted.svg", 1.0f);

        printf("step_size = %f\r\n", step_size);

        if (step_size < 0.01) break;
    }

    printf("\033[0;32mTransport map solver done! Starting height solver.\033[0m\r\n");

    for (int itr = 0; itr < 1; itr++) {
        auto start = std::chrono::high_resolution_clock::now();
        caustic_design.perform_height_map_iteration(itr);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("Height map iteration %d took %f seconds\n", itr, elapsed.count());
    }

    printf("Height solver done! Exporting as solidified obj\r\n");

    caustic_design.save_solid_obj_source("../output.obj");

    try {
        save_heightmap_as_json(caustic_design.h, "../heightmap.json");
    } catch (std::exception& e) {
        printf("Failed to save heightmap as JSON: %s\r\n", e.what());
    } catch (...) {
        printf("Failed to save heightmap as JSON.\r\n");
    }

    return 0;
}