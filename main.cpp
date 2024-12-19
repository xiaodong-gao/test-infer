#include <opencv2/opencv.hpp>
#include <iostream>
#include "infer.hpp"
#include "yolo.hpp"
#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>

struct bbox_t {
    unsigned int x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;        // class of object - from range [0, classes-1]
    //unsigned int track_id;        // tracking id for video (0 - untracked, 1 - inf - tracked object)
    //unsigned int frames_counter;// counter of frames on which the object was detected
};

yolo::Image cvimg(const cv::Mat& image) { return yolo::Image(image.data, image.cols, image.rows); }

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names) {
    for (auto& i : result_vec) {
        cv::Scalar color(60, 160, 260);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
        if (obj_names.size() > i.obj_id)
            putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
        //if (i.track_id > 0)
        //    putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x + 5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
    }
    cv::namedWindow("window name", cv::WINDOW_NORMAL);
    cv::imshow("window name", mat_img);
    cv::waitKey(0);
}


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
    for (auto& i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; file >> line;) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

int main()
{
    cv::Mat image = cv::imread("inference/car.jpg");

    auto yolo = yolo::load("yolov8n-seg.b1.transd.engine", yolo::Type::V8Seg);
    if (yolo == nullptr)
        return -1;

    auto objs = yolo->forward(cvimg(image));


    auto obj_names = objects_names_from_file("voc.names");

    std::vector<bbox_t> result_vec;
    for (auto& obj : objs) {
        bbox_t box;
        box.x = obj.left;
        box.y = obj.top;
        box.w = obj.right - obj.left;
        box.h = obj.bottom - obj.top;
        result_vec.push_back(box);
    }

    draw_boxes(image, result_vec, obj_names);

    printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("Result.jpg", image);
    return 0;
}
