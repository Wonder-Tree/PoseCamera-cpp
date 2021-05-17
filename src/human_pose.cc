#include <vector>

#include "human_pose.h"

namespace human_pose_estimation {
HumanPose::HumanPose(const std::vector<cv::Point2f>& keypoints,
                     const float& score)
    : keypoints(keypoints),
      score(score) {}
} 