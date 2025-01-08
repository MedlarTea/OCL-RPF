#ifndef MONO_FOLLOWING_STATE_HPP
#define MONO_FOLLOWING_STATE_HPP

#include <mono_following/descriminator.hpp>
#include <ros/ros.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mono_following/tracklet.hpp>

namespace mono_following {

class State {
public:
    State() {}
    virtual ~State() {}

    virtual long target() const { return -1; }

    virtual std::string state_name() const = 0;

    virtual State* update(ros::NodeHandle& nh, Descriminator& descriminator, const std::unordered_map<long, Tracklet::Ptr>& tracks) = 0;
private:

};

}

#endif
