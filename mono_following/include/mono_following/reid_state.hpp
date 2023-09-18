#ifndef REID_STATE_HPP
#define REID_STATE_HPP

#include <mono_following/state.hpp>

namespace mono_following {

class ReidState : public State {
public:
    ReidState() {}
    virtual ~ReidState() override {}

    virtual std::string state_name() const override {
        return "re-identification";
    }

    virtual State* update(ros::NodeHandle& nh, Descriminator& descriminator, const std::unordered_map<long, Tracklet::Ptr>& tracks) override;

private:
    std::unordered_map<long, int> positive_count;
};

}

#endif // REID_STATE_HPP
