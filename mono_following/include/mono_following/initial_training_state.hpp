#ifndef MONO_FOLLOWING_INITIAL_TRAINING_STATE_HPP
#define MONO_FOLLOWING_INITIAL_TRAINING_STATE_HPP

// #include <mono_following/context.hpp>
#include <mono_following/state.hpp>

namespace mono_following {

class InitialTrainingState: public State {
public:
    InitialTrainingState(long target_id)
    : target_id(target_id),
      num_pos_samples(0)
    {}
    
    virtual ~InitialTrainingState() override {}

    virtual long target() const override {
        return target_id;
    }

    virtual std::string state_name() const override{
        return "initial_training"; 
    }

    virtual State* update(ros::NodeHandle& nh, Descriminator& descriminator, const std::unordered_map<long, Tracklet::Ptr>& tracks) override;
private:
    long target_id;
    std::vector<long> neg_ids;
    long num_pos_samples;
    long num_neg_samples;

};

}

#endif
