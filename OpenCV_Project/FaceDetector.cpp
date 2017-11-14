#include "FaceDetector.h"

FaceDetector::FaceDetector(const std::string& image_path)
	: face_classifier{}, eye_classifier{}, image_path{ image_path }
{
	this->image = cv::imread(image_path);
	cv::cvtColor(this->image, this->gray_image, cv::COLOR_BGR2GRAY);
}

void FaceDetector::create_face_classifier(const std::string & face_classifier_path)
{
	this->face_classifier.load(face_classifier_path);
}

void FaceDetector::create_eye_classifier(const std::string & eye_classifier_path)
{
	this->eye_classifier.load(eye_classifier_path);
}

void FaceDetector::detect_faces(double scale_factor, int min_neighbours)
{
	std::vector<cv::Rect> detected_faces{};
	face_classifier.detectMultiScale(this->gray_image, detected_faces, scale_factor, min_neighbours, CV_HAAR_SCALE_IMAGE, cv::Size{ 30, 30 });

	for (const auto& face : detected_faces)
		cv::rectangle(this->image, face, cv::Scalar{255, 0, 0}, 2);
}

void FaceDetector::detect_eyes(double scale_factor, int min_neighbours)
{
	std::vector<cv::Rect> detected_eyes{};
	eye_classifier.detectMultiScale(this->gray_image, detected_eyes, scale_factor, min_neighbours, CV_HAAR_SCALE_IMAGE);

	for (const auto& eyes : detected_eyes)
		cv::rectangle(this->image, eyes, cv::Scalar{0, 0, 255}, 2);
}

void FaceDetector::show_image()
{
	cv::imshow(this->image_path, this->image);
	cv::waitKey();
}