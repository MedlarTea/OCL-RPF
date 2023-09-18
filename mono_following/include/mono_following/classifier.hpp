#ifndef MONO_FOLLOWING_CLASSIFIER_HPP
#define MONO_FOLLOWING_CLASSIFIER_HPP

#include <random>
#include <unordered_map>
#include <queue>
#include <boost/optional.hpp>
#include <boost/circular_buffer.hpp>

#include <ros/ros.h>
#include <mono_following/tracklet.hpp>

#include <mono_following/Samples.h>
#include <mono_following/classifyTarget.h>


namespace mono_following {

class Classifier {
public:
    Classifier(ros::NodeHandle& nh);
    ~Classifier();

public:
    bool update_classifier(double label, const Tracklet::Ptr& track);
    bool predict(std::unordered_map<long, Tracklet::Ptr>& tracks);

private:
    // std::mt19937 mt;  // original author uses this to insert element randomly

    // for extraction of descriptors
    ros::ServiceClient classifyTarget_client;
    mono_following::classifyTarget classifyTarget_srv;
    // for extractor update
    ros::Publisher samples_pub;
    

    // positive image patches & descriptors
    boost::circular_buffer<cv::Mat> pos_patches_bank;
    boost::circular_buffer<cv::Mat> pos_feature_bank;
    // negative image patches & descriptors
    boost::circular_buffer<cv::Mat> neg_patches_bank;
    boost::circular_buffer<cv::Mat> neg_feature_bank;

};

}

#endif
