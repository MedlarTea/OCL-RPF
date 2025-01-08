#ifndef MONO_TRACKING_ASSOCIATION_DISTANCE_HPP
#define MONO_TRACKING_ASSOCIATION_DISTANCE_HPP

#include <memory>
#include <vector>
#include <boost/optional.hpp>

#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <mono_tracking/person_tracker.hpp>
using namespace std;

namespace mono_tracking {

class AssociationDistance {
public:
    AssociationDistance(ros::NodeHandle& private_nh)
        : maha_sq_thresh(private_nh.param<double>("association_maha_sq_thresh", 9.0)),
        max_dist(private_nh.param<int>("max_dist", 100))
  {
  }

    boost::optional<double> operator() (const PersonTracker::Ptr& tracker, const Observation::Ptr& observation) const {

        const auto &expected_measurement = tracker->expected_measurement_distribution(tracker->real_width_value());
        // cout << "Now state: " << tracker->pos().x() << ", " << tracker->pos().y() << ", " << tracker->vel().x() << ", " << tracker->vel().y() << endl;

        const auto &measurement = Eigen::Vector2f(((*observation->bbox)(0)+(*observation->bbox)(2))/2.0f, (*observation->bbox)(2)-(*observation->bbox)(0));
        double distance = (expected_measurement - measurement).norm();
        // cout << "Expected measurements: " << expected_measurement << endl;
        // cout << "Measurements: " << measurement << endl;
        // cout << "Distance: " << distance << endl;

        

        if(!observation->min_distance)
            observation->min_distance = distance;
        else
            observation->min_distance = std::min(distance, *observation->min_distance);

        if(distance > max_dist)
            return boost::none;
        return distance;
  }

  bool is_good_expected_measurement(const Eigen::Vector2f &point, int left_board, int right_board, int up_board, int down_board) const{
      if (point.x() < left_board || point.x() > right_board || point.y() < up_board || point.y() > down_board)
        return false;
      return true;
  }
private:
  double maha_sq_thresh;
  // int neck_ankle_max_dist;
  int max_dist;
};

}

#endif
