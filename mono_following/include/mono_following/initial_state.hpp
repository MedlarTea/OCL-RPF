#ifndef MONO_FOLLOWING_INITIAL_STATE_HPP
#define MONO_FOLLOWING_INITIAL_STATE_HPP

// #include <mono_following/context.hpp>
#include <mono_following/state.hpp>

namespace mono_following {

class InitialState: public State {
public:
    InitialState() {}
    virtual ~InitialState() override {}

    virtual std::string state_name() const override{
        return "initial"; 
    }

    virtual State* update(ros::NodeHandle& nh, Descriminator& descriminator, const std::unordered_map<long, Tracklet::Ptr>& tracks) override;
private:
    long select_target(ros::NodeHandle& nh, const std::unordered_map<long, Tracklet::Ptr>& tracks);

};

}

#endif
