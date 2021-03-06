#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

#include "human_pose.h"

namespace human_pose_estimation {
    void renderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image);
}