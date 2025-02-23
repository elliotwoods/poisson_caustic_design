#include "stdio.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <png.h>

#include "src/caustic_design.h"

#include <thread>

void resize_image(const std::vector<std::vector<double>>& input_image, std::vector<std::vector<double>>& output_image, int new_width, int new_height) {
    int old_height = input_image.size();
    int old_width = input_image[0].size();

    output_image.resize(new_height, std::vector<double>(new_width));

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            int src_x = x * old_width / new_width;
            int src_y = y * old_height / new_height;

            output_image[y][x] = input_image[src_y][src_x];
        }
    }
}

void image_to_grid(const std::string& filename, std::vector<std::vector<double>>& image_grid) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Failed to open PNG file.");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("Failed to create PNG read struct.");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        throw std::runtime_error("Failed to create PNG info struct.");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        throw std::runtime_error("Error during PNG read initialization.");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    png_read_update_info(png, info);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers.data());

    fclose(fp);

    for (int i = 0; i < height; ++i) {
        std::vector<double> row;
        for (int j = 0; j < width; ++j) {
            png_bytep px = &row_pointers[i][j * 4];
            double r = px[0] / 255.0;
            double g = px[1] / 255.0;
            double b = px[2] / 255.0;
            double gray = (0.299 * r) + (0.587 * g) + (0.114 * b);
            row.push_back(gray);
        }
        image_grid.push_back(row);
    }

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    png_destroy_read_struct(&png, &info, NULL);
}

void save_heightmap_as_json(const std::vector<std::vector<double>>& heightmap, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open output file.");
    }

    outfile << "[\n";
    for (size_t i = 0; i < heightmap.size(); ++i) {
        outfile << "  [";
        for (size_t j = 0; j < heightmap[i].size(); ++j) {
            outfile << heightmap[i][j];
            if (j < heightmap[i].size() - 1) {
                outfile << ", ";
            }
        }
        outfile << "]";
        if (i < heightmap.size() - 1) {
            outfile << ",\n";
        } else {
            outfile << "\n";
        }
    }
    outfile << "]\n";
    outfile.close();
}

std::unordered_map<std::string, std::string> parse_arguments(int argc, char const *argv[]) {
    // Define a map to store the parsed arguments
    std::unordered_map<std::string, std::string> args;

    // Iterate through command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key, value;

        // Check if argument starts with '--'
        if (arg.substr(0, 2) == "--") {
            // Split argument by '=' to separate key and value
            size_t pos = arg.find('=');
            if (pos != std::string::npos) {
                key = arg.substr(2, pos - 2);
                value = arg.substr(pos + 1);
            }
            else {
                key = arg.substr(2);
                value = ""; // No value provided
            }
        }
        // Check if argument starts with '-'
        else if (arg[0] == '-') {
            // The next argument is the value
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                key = arg.substr(1);
                value = argv[++i];
            }
            else {
                key = arg.substr(1);
                value = ""; // No value provided
            }
        }

        // Store key-value pair in the map
        args[key] = value;
    }

    return args;
}

int main(int argc, char const *argv[]) {
    // Parse user arguments
    std::unordered_map<std::string, std::string> args = parse_arguments(argc, argv);

    // Print parsed arguments
    /*for (const auto& pair : args) {
        //std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        printf("Key: %s, Value: %s\r\n", pair.first.c_str(), pair.second.c_str());
    }*/

    // Load image to grid
    std::vector<std::vector<double>> pixels;
    image_to_grid(args["input_png"], pixels);
    double aspect_ratio = (double)pixels[0].size() / (double)pixels.size();

    std::vector<std::vector<double>> resized_pixels;
    resize_image(pixels, resized_pixels, 4 * atoi(args["res_w"].c_str()), 4 * atoi(args["res_w"].c_str()) / aspect_ratio);

    Caustic_design caustic_design;

    int mesh_resolution_x = atoi(args["res_w"].c_str());
    double mesh_width = std::stod(args["width"]);

    caustic_design.set_mesh_resolution(mesh_resolution_x, mesh_resolution_x / aspect_ratio);
    caustic_design.set_domain_resolution(4 * mesh_resolution_x, 4 * mesh_resolution_x / aspect_ratio);

    double mesh_height = floor((mesh_resolution_x) / aspect_ratio) * (mesh_width / (mesh_resolution_x));

    caustic_design.set_mesh_size(mesh_width, mesh_height);

    caustic_design.set_lens_focal_length(std::stod(args["focal_l"]));
    caustic_design.set_lens_thickness(std::stod(args["thickness"]));
    caustic_design.set_solver_max_threads(atoi(args["max_threads"].c_str()));

    caustic_design.initialize_solvers(resized_pixels);

    caustic_design.export_paramererization_to_svg("../parameterization_0.svg", 0.5f);

    for (int itr = 0; itr < 30; itr++) {
        printf("starting iteration %i\r\n", itr);

        double step_size = caustic_design.perform_transport_iteration();

        //export_cells_as_svg(caustic_design.source_cells, scale_array_proportional(caustic_design.vertex_gradient[0], 0.0f, 1.0f), "../x_grad.svg");

        //save_grid_as_image(scale_matrix_proportional(caustic_design.gradient[0], 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "../grad_x_" + std::to_string(itr) + ".png");
        //save_grid_as_image(scale_matrix_proportional(caustic_design.gradient[1], 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "../grad_y_" + std::to_string(itr) + ".png");
        //save_grid_as_image(scale_matrix_proportional(caustic_design.raster, 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "../raster_" 

        caustic_design.export_paramererization_to_svg("../parameterization_" + std::to_string(itr + 1) + ".svg", 1.0f);
        caustic_design.export_inverted_transport_map("../inverted.svg", 1.0f);

        printf("step_size = %f\r\n", step_size);

        if (step_size < 0.01) break;
    }

    printf("\033[0;32mTransport map solver done! Starting height solver.\033[0m\r\n");

    for (int itr = 0; itr < 3; itr++) {
        caustic_design.perform_height_map_iteration(itr);
        //save_grid_as_image(scale_matrix_proportional(caustic_design.h, 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "../h" + std::to_string(itr) + ".png");
        //save_grid_as_image(scale_matrix_proportional(caustic_design.divergence, 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "../div" + std::to_string(itr) + ".png");
        //save_grid_as_image(scale_matrix_proportional(caustic_design.norm_x, 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "norm_x" + std::to_string(itr) + ".png");
        //save_grid_as_image(scale_matrix_proportional(caustic_design.norm_y, 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "norm_y" + std
    }

    /*std::vector<std::vector<double>> normals;
    std::vector<std::vector<double>> normals_trg;
    for (int i=0; i<caustic_design.mesh->source_points.size(); i++) {
        std::vector<double> normal = caustic_design.calculate_vertex_normal(caustic_design.mesh->source_points, i);
        normal[2] *= (mesh_resolution_x * 4) / mesh_width;
        normal = normalize(normal);

        normals_trg.push_back(normalize({
            caustic_design.normals[0][i],
            caustic_design.normals[1][i],
            caustic_design.normals[2][i]
        }));

        normals.push_back(normal);
    }

    std::vector<double> E_int;
    for (int i=0; i<caustic_design.mesh->source_points.size(); i++) {
        std::vector<double> diff = vector_subtract(normals[i], normals_trg[i]);
        double energy = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
        E_int.push_back(energy);
    }

    /*bool miss = false;
    std::vector<std::vector<double>> norm_x = caustic_design.mesh->interpolate_raster_source(x_normals, 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, miss);
    std::vector<std::vector<double>> norm_y = caustic_design.mesh->interpolate_raster_source(y_normals, 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, miss);
    save_grid_as_image(scale_matrix_proportional(norm_x, 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "../x_normals.png");
    save_grid_as_image(scale_matrix_proportional(norm_y, 0.0f, 1.0f), 4*mesh_resolution_x, 4*mesh_resolution_x / aspect_ratio, "../y_normals.png");
    //*/

    //export_cells_as_svg(caustic_design.source_cells, scale_array_proportional(E_int, 0.0f, 1.0f), "../integration_energy.svg");
    //export_cells_as_svg(caustic_design.source_cells, scale_array_proportional(y_normals_trg, 0.0f, 1.0f), "../y_normals_trg.svg");
    //export_cells_as_svg(caustic_design.source_cells, scale_array_proportional(x_normals, 0.0f, 1.0f), "../x_normals.svg");
    //export_cells_as_svg(caustic_design.source_cells, scale_array_proportional(y_normals, 0.0f, 1.0f), "../y_normals.svg");

    printf("Height solver done! Exporting as solidified obj\r\n");

    caustic_design.save_solid_obj_source("../output.obj");

    // Save the heightmap to JSON
    save_heightmap_as_json(caustic_design.h, "../heightmap.json");

    return 0;
}
