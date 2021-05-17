#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "peak.h"

#include <torch/torch.h>
#include <torch/cuda.h>

namespace human_pose_estimation {

class FindPeaksBody : public cv::ParallelLoopBody {
 public:
  FindPeaksBody(const std::vector<cv::Mat> &heatMaps, float minPeaksDistance,
				std::vector<std::vector<Peak> > &peaksFromHeatMap)
	  : heatMaps(heatMaps),
		minPeaksDistance(minPeaksDistance),
		peaksFromHeatMap(peaksFromHeatMap) {}

  virtual void operator()(const cv::Range &range) const {
	for (int i = range.start; i < range.end; i++) {
	  findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
	}
  }

 private:
  const std::vector<cv::Mat> &heatMaps;
  float minPeaksDistance;
  std::vector<std::vector<Peak> > &peaksFromHeatMap;
};

class HumanPoseEstimator {
 private:
//  std::shared_ptr<torch::jit::script::Module> module;
  std::shared_ptr<torch::jit::script::Module> module;
  torch::jit::script::Module _module_not_ptr;

  int minJointsNumber;
  int stride;
  cv::Vec4i pad;
  cv::Vec3f meanPixel;
  float minPeaksDistance;
  float midPointsScoreThreshold;
  float foundMidPointsRatioThreshold;
  float minSubsetScore;
  cv::Size inputLayerSize;
  int upsampleRatio;

  std::string modelPath;
  torch::DeviceType device_type = torch::kCUDA;

 public:

  HumanPoseEstimator(const std::string &modelPath);
  ~HumanPoseEstimator();
  //estimate
  std::vector<HumanPose> estimate(const cv::Mat &image);

  void preprocess(const cv::Mat &image, cv::Mat &input_image);

	std::vector<HumanPose> postprocess(
	  const torch::Tensor &heatMapsTensor,
	  const torch::Tensor &pafsTensor,
	  const cv::Size &imageSize);

  std::vector<HumanPose> extractPoses(
	  const std::vector<cv::Mat> &heatMaps,
	  const std::vector<cv::Mat> &pafs);

  void resizeFeatureMaps(std::vector<cv::Mat> &featureMaps);

  void correctCoordinates(std::vector<HumanPose> &poses,
						  const cv::Size &featureMapsSize,
						  const cv::Size &imageSize);

  bool inputWidthIsChanged(const cv::Size &imageSize);


  const size_t keypointsNumber = 18;
};

}