#include "caustic_design.h"

Caustic_design::Caustic_design(/* args */)
{
    this->mesh_res_x = 0;
    this->mesh_res_y = 0;
    this->resolution_x = 0;
    this->resolution_y = 0;
    this->width = 0.0f;
    this->height = 0.0f;
    this->focal_l = 0.0f;
    this->thickness = 0.0f;
    this->nthreads = 0;
}

Caustic_design::~Caustic_design()
{
}

void Caustic_design::export_inverted_transport_map(std::string filename, double stroke_width) {
    mesh->calculate_and_export_inverted_transport_map(filename, stroke_width);
}

void Caustic_design::export_paramererization_to_svg(const std::string& filename, double line_width) {
    mesh->export_paramererization_to_svg(filename, line_width);
}

void Caustic_design::set_mesh_resolution(int width, int height) {
    this->mesh_res_x = width;
    this->mesh_res_y = height;
}

void Caustic_design::set_domain_resolution(int width, int height) {
    this->resolution_x = width;
    this->resolution_y = height;
}

void Caustic_design::set_mesh_size(double width, double height) {
    this->width = width;
    this->height = height;
}

void Caustic_design::set_lens_focal_length(double focal_length) {
    this->focal_l = focal_length;
}

void Caustic_design::set_lens_thickness(double thickness) {
    this->thickness = thickness;
}

void Caustic_design::set_solver_max_threads(int n_threads) {
    this->nthreads = n_threads;
}

void Caustic_design::save_solid_obj_target(const std::string& filename) {
    this->mesh->save_solid_obj_target(thickness, filename);
}

void Caustic_design::save_solid_obj_source(const std::string& filename) {
    this->mesh->save_solid_obj_source(thickness, filename);
}

// Function to calculate the approximate vertex normal
std::vector<double> Caustic_design::calculate_vertex_normal(cv::Mat &points, int vertex_index) {
    std::vector<double> avg_normal = {0.0, 0.0, 0.0}; // Initialize normal to zero vector
    
    int left_vtx = 0;
    int right_vtx = 0;
    int top_vtx = 0;
    int bot_vtx = 0;

    mesh->get_vertex_neighbor_ids(vertex_index, left_vtx, right_vtx, top_vtx, bot_vtx);

    if (left_vtx != -1 && top_vtx != -1) {
        std::vector<double> normal;
        double angle_out;

        calculate_angle_and_normal_from_triangle(points.row(vertex_index), points.row(left_vtx), points.row(top_vtx), normal, angle_out);

        avg_normal[0] += normal[0] * angle_out;
        avg_normal[1] += normal[1] * angle_out;
        avg_normal[2] += normal[2] * angle_out;
    }

    if (left_vtx != -1 && bot_vtx != -1) {
        std::vector<double> normal;
        double angle_out;

        calculate_angle_and_normal_from_triangle(points.row(vertex_index), points.row(bot_vtx), points.row(left_vtx), normal, angle_out);

        avg_normal[0] += normal[0] * angle_out;
        avg_normal[1] += normal[1] * angle_out;
        avg_normal[2] += normal[2] * angle_out;
    }

    if (right_vtx != -1 && bot_vtx != -1) {
        std::vector<double> normal;
        double angle_out;

        calculate_angle_and_normal_from_triangle(points.row(vertex_index), points.row(right_vtx), points.row(bot_vtx), normal, angle_out);
        
        avg_normal[0] += normal[0] * angle_out;
        avg_normal[1] += normal[1] * angle_out;
        avg_normal[2] += normal[2] * angle_out;
    }

    if (right_vtx != -1 && top_vtx != -1) {
        std::vector<double> normal;
        double angle_out;

        calculate_angle_and_normal_from_triangle(points.row(vertex_index), points.row(top_vtx), points.row(right_vtx), normal, angle_out);
        
        avg_normal[0] += normal[0] * angle_out;
        avg_normal[1] += normal[1] * angle_out;
        avg_normal[2] += normal[2] * angle_out;
    }

    double magnitude = sqrt(avg_normal[0] * avg_normal[0] + avg_normal[1] * avg_normal[1] + avg_normal[2] * avg_normal[2]);

    if (magnitude > 1e-12) {
        avg_normal[0] /= -magnitude;
        avg_normal[1] /= -magnitude;
        avg_normal[2] /= magnitude;
    }

    return avg_normal;
}

void clamp(int &value, int min, int max) {
    value = std::max(std::min(value, max), min);
}

// Bilinear interpolation function
double bilinearInterpolation(const cv::Mat& image, double x, double y) {
    int x0 = floor(x);
    int y0 = floor(y);
    int x1 = ceil(x);
    int y1 = ceil(y);

    clamp(x0, 0, image.cols - 1);
    clamp(x1, 0, image.cols - 1);
    clamp(y0, 0, image.rows - 1);
    clamp(y1, 0, image.rows - 1);

    if (x0 < 0 || y0 < 0 || x1 >= image.cols || y1 >= image.rows) {
        printf("interpolation out of range: x: %f, y: %f\r\n", x, y);

        return 0.0;
    }

    double fx1 = x - x0;
    double fx0 = 1.0 - fx1;

    double fy1 = y - y0;
    double fy0 = 1.0 - fy1;

    double top = fx0 * image.at<double>(y0, x0) + fx1 * image.at<double>(y0, x1);
    double bottom = fx0 * image.at<double>(y1, x0) + fx1 * image.at<double>(y1, x1);
    return fy0 * top + fy1 * bottom;
}

double Caustic_design::perform_transport_iteration() {
    double min_step;

    target_cells.clear();
    mesh->build_target_dual_cells(target_cells);

    std::vector<double> source_areas = get_source_areas(target_cells);
    calculate_errors(source_areas, target_areas, target_cells, errors);

    bool triangle_miss = false;
    raster = mesh->interpolate_raster_target(errors, resolution_x, resolution_y, triangle_miss);
    
    if (triangle_miss) {
        mesh->laplacian_smoothing(mesh->target_points, 0.1f);
        return NAN;
    }

    subtractAverage(raster);
    poisson_solver(raster, phi, resolution_x, resolution_y, 100000, 0.0000001, nthreads);

    gradient = calculate_gradient(phi);

    double epsilon = 1e-8;
    
    cv::Mat vertex_gradient_x(mesh->target_points.size(), 1, CV_64F);
    cv::Mat vertex_gradient_y(mesh->target_points.size(), 1, CV_64F);

    for (int i=0; i<mesh->target_points.size(); i++) {
        vertex_gradient_x.at<double>(i) = bilinearInterpolation(gradient[0], 
            (mesh->target_points[i][0] / mesh->width) * (resolution_x) - 0.5, 
            (mesh->target_points[i][1] / mesh->height) * (resolution_y) - 0.5
        );

        vertex_gradient_y.at<double>(i) = bilinearInterpolation(gradient[1], 
            (mesh->target_points[i][0] / mesh->width) * (resolution_x) - 0.5, 
            (mesh->target_points[i][1] / mesh->height) * (resolution_y) - 0.5
        );
    }

    vertex_gradient.clear();
    vertex_gradient.push_back(vertex_gradient_x);
    vertex_gradient.push_back(vertex_gradient_y);
    
    std::vector<std::vector<double>> old_points;

    std::copy(mesh->target_points.begin(), mesh->target_points.end(), back_inserter(old_points));

    mesh->step_grid(vertex_gradient_x, vertex_gradient_y, 0.05f);

    min_step = 0.0f;

    for (int i=0; i<old_points.size(); i++) {
        double dx = (old_points[i][0] - mesh->target_points[i][0]);
        double dy = (old_points[i][1] - mesh->target_points[i][1]);
        double dz = (old_points[i][2] - mesh->target_points[i][2]);

        double dist = sqrt(dx*dx + dy*dy + dz*dz);

        if (min_step < dist) {
            min_step = dist;
        }
    }

    return min_step / width;
}

void Caustic_design::perform_height_map_iteration(int itr) {
    normals = mesh->calculate_refractive_normals_uniform(resolution_x / width * focal_l, 1.49);

    mesh->build_source_bvh(5, 30);
    bool triangle_miss = false;
    norm_x = mesh->interpolate_raster_source(normals[0], resolution_x, resolution_y, triangle_miss);
    norm_y = mesh->interpolate_raster_source(normals[1], resolution_x, resolution_y, triangle_miss);

    if (triangle_miss) {
        return;
    }

    divergence = calculate_divergence(norm_x, norm_y, resolution_x, resolution_y);
    subtractAverage(divergence);

    poisson_solver(divergence, h, resolution_x, resolution_y, 100000, 0.00000001, nthreads);

    double epsilon = 1e-8;

    cv::Mat interpolated_h(mesh->source_points.size(), 1, CV_64F);

    for (int i=0; i<mesh->source_points.size(); i++) {
        interpolated_h.at<double>(i) = bilinearInterpolation(h, 
            (mesh->source_points[i][0] / mesh->width) * (resolution_x) - 0.5, 
            (mesh->source_points[i][1] / mesh->height) * (resolution_y) - 0.5
        );
    }
    double max_update = mesh->set_source_heights(interpolated_h);
    printf("height max update %.5e\r\n", max_update);
}

void Caustic_design::initialize_solvers(cv::Mat image) {
    pixels = scale_matrix_proportional(image, 0, 1.0f);

    printf("scaled\r\n");

    mesh = new Mesh(width, height, mesh_res_x, mesh_res_y);

    printf("generated mesh\r\n");

    mesh->export_to_svg("../mesh.svg", 1);

    mesh->build_target_dual_cells(target_cells);
    mesh->build_source_dual_cells(source_cells);

    target_areas = get_target_areas(pixels, target_cells, resolution_x, resolution_y, width, height);

    export_cells_as_svg(target_cells, scale_array_proportional(target_areas, 0.0f, 1.0f), "../cells.svg");

    phi = cv::Mat::zeros(resolution_y, resolution_x, CV_64F);
    h = cv::Mat::zeros(resolution_y, resolution_x, CV_64F);
}
