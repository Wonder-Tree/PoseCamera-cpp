#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace human_pose_estimation {
struct HumanPose {
    HumanPose(const std::vector<cv::Point2f>& keypoints = std::vector<cv::Point2f>(),
              const float& score = 0);

    std::vector<cv::Point2f> keypoints;
    float score;
};
} 