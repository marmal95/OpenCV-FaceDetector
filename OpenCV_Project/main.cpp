#include <cstdlib>
#include <iostream>

#include "FaceDetector.h"

int main(int argc, char* argv[])
{
	const std::string frontal_face_cascade_path = "haarcascade_frontalface_default.xml";
	const std::string eye_cascade_path = "haarcascade_eye.xml";

	if (argc < 1)
	{
		std::cerr << "Please specify the entry image" << std::endl;
		return EXIT_FAILURE;
	}

	try
	{
		FaceDetector face_detector{ argv[1] };
		face_detector.create_face_classifier(frontal_face_cascade_path);
		face_detector.create_eye_classifier(eye_cascade_path);

		// TODO: Experiment with the following values
		face_detector.detect_faces(1.3, 5);
		face_detector.detect_eyes(1.1, 2);

		face_detector.show_image();
	}
	catch (cv::Exception& exception)
	{
		std::cerr << exception.what();
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}