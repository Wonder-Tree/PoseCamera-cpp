#include <torch/script.h>
#include <iostream>
#include <memory>
#include <string>
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "human_pose_estimator.h"
#include "render_human_pose.h"


using namespace std;
using namespace human_pose_estimation;
using namespace cv;



int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: app <path-to-exported-script-module>\n";
    return -1;
  }


  HumanPoseEstimator estimator(argv[1]);
  std::cout << "Intialized";
  string v_f = "../data/sample.mp4";
  if (argc > 2) {
    v_f = argv[2];
  }

  if (v_f.find("mp4") != string::npos) {
	std::cout << "inference on video file: " << std::endl;
	cv::VideoCapture cap(v_f);
	if (!cap.isOpened()) { std::cout << "failed to open file.";}

	while (true) {
	  cv::Mat frame;
	  bool ok = cap.read(frame);

	  if (!ok) {
		return 1;
	  }

	  double t1 = static_cast<double>(cv::getTickCount());
	  std::vector<HumanPose> poses = estimator.estimate(frame);
	  double delta_s = (static_cast<double>(cv::getTickCount()) - t1)/cv::getTickFrequency();

	  for (HumanPose const& pose : poses) {
		std::stringstream rawPose;
		rawPose << std::fixed << std::setprecision(0);
		rawPose << pose.score;
	  }
	  human_pose_estimation::renderHumanPose(poses, frame);

	  cv::Mat fpsPane(35, 155, CV_8UC3);
	  fpsPane.setTo(cv::Scalar(153, 119, 76));
	  cv::Mat srcRegion = frame(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
	  cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
	  cv::putText(frame, to_string(1/delta_s), cv::Point(16, 32),
				  cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));

	  cv::imshow("Pose Camera - HumanPose Estimation", frame);
	  cv::waitKey(1);
	}
  } else {
	  std::cout << "Specify video file.";
  }

  return 1;
}