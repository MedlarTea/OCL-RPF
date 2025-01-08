#ifndef MONO_TRACKING_PERSON_TRACKER_HPP
#define MONO_TRACKING_PERSON_TRACKER_HPP

#include <memory>
#include <fstream>
#include <Eigen/Dense>

#include <kkl/math/gaussian.hpp>
#include <mono_tracking/observation.hpp>

namespace kkl {
namespace alg {
  template<typename T, typename System>
  class UnscentedKalmanFilterX;
}
}

namespace mono_tracking {

class TrackSystem;

class PersonTracker {
public:
  using Ptr = std::shared_ptr<PersonTracker>;
  using UnscentedKalmanFilter = kkl::alg::UnscentedKalmanFilterX<float, TrackSystem>;

  PersonTracker(ros::NodeHandle& nh, const std::shared_ptr<TrackSystem>& track_system, const ros::Time& stamp, long id, const Observation::Ptr& observation);
  ~PersonTracker();

  void predict(const ros::Time& stamp);
  void correct(const ros::Time& stamp, const Observation::Ptr& observation);

  Observation::Ptr get_last_associated() const { return last_associated; }

  Eigen::Vector2f expected_measurement_distribution(const float &real_width) const;

  long id() const {
    return id_;
  }

  double trace() const;
  Eigen::Vector2f pos() const;
  Eigen::Vector3f heights() const;
  Eigen::Vector2f vel() const;
  Eigen::MatrixXf cov() const;
  float real_width_value() const;

  long correction_count() const {
    return correction_count_;
  }

  long prediction_count() const {
    return prediction_count_;
  }

  bool is_valid() const {
    return correction_count() > validation_correction_count;

  }

private:
  Eigen::Matrix<float,5,1> estimate_init_state(const std::shared_ptr<TrackSystem>& track_system,  const std::pair<std::bitset<4>, Eigen::MatrixXf> &measurement);
  Eigen::Vector2f estimate_init_state_by_single(const std::shared_ptr<TrackSystem>& track_system,  const std::pair<std::bitset<4>, Eigen::MatrixXf> &measurement);

private:
  float h_neck, h_waist, h_kneel;
  float real_width;
  long id_;
  long correction_count_, prediction_count_;
  long validation_correction_count;
  ros::Time prev_stamp;

  Observation::Ptr last_associated;
  std::unique_ptr<UnscentedKalmanFilter> ukf;
  mutable boost::optional<Eigen::Vector2f> expected_measurement_dist;

  // float neck_height, waist_height, kneel_height;
  // Kalman filter parameters
  
};

}

#endif // PERSON_TRACKER_HPP
