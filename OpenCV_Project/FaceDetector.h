#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class FaceDetector
{
public:
	FaceDetector(const std::string& image_path);
	void create_face_classifier(const std::string& face_classifier_path);
	void create_eye_classifier(const std::string& eye_classifier_path);
	void detect_faces(double scale_factor = 1.1, int min_neighbours = 3);
	void detect_eyes(double scale_factor = 1.1, int min_neighbours = 3);
	void show_image();

private:
	cv::CascadeClassifier face_classifier;
	cv::CascadeClassifier eye_classifier;

	std::string image_path;

	cv::Mat image;
	cv::Mat gray_image;
};