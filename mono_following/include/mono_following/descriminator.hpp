#ifndef MONO_FOLLOWING_DESCRIMINATOR_HPP
#define MONO_FOLLOWING_DESCRIMINATOR_HPP

#include <random>
#include <unordered_map>
#include <queue>
#include <boost/optional.hpp>
#include <boost/circular_buffer.hpp>

#include <ros/ros.h>
#include <mono_following/tracklet.hpp>

#include <mono_following/Box.h>
#include <mono_following/Samples.h>
#include <mono_following/Descriptor.h>
#include <mono_following/extractDescriptors.h>
#include <mono_following/updateExtractorAction.h>
#include <mono_following/updateDescriptors.h>
#include <mono_following/classifyTarget.h>


namespace mono_following {

class Descriminator {
public:
    Descriminator(ros::NodeHandle& nh);
    ~Descriminator();

public:
    static std::vector<bool> isGoodBbox(vector<vector<u_int16_t>> boxes);
    bool update_features(const std::unordered_map<long, Tracklet::Ptr>& tracks, const long &target_id);

    cv::Mat convertVector2Mat(std::vector<double> v, int channels, int rows);
    void extract_features(std::unordered_map<long, Tracklet::Ptr>& tracks);

    boost::optional<double> predict(const Tracklet::Ptr& track);
    boost::optional<vector<double>>predict(const std::unordered_map<long, Tracklet::Ptr>& tracks);

    std::vector<std::string> decriminator_names() const;

private:
    // std::mt19937 mt;  // original author uses this to insert element randomly

    // for extraction of descriptors
    ros::ServiceClient updateDescriptors_client, classifyTarget_client;
    mono_following::updateDescriptors updateDescriptors_srv;
    mono_following::classifyTarget classifyTarget_srv;

    // for extraction of descriptors
    ros::ServiceClient extractDescriptor_client;
    mono_following::extractDescriptors extractDescriptor_srv;

};

}

#endif
