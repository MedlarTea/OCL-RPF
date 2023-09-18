#ifndef MONO_TRACKING_TRACKSYSTEM_HPP
#define MONO_TRACKING_TRACKSYSTEM_HPP

#include <Eigen/Dense>
#include <ros/node_handle.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/CameraInfo.h>
#include <bitset>
#include<math.h>
using namespace std;
namespace mono_tracking {

class TrackSystem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrackSystem(ros::NodeHandle& private_nh, const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
    : camera_frame_id(camera_frame_id),
      tf_listener(tf_listener)
  {
    dt = 0.1;

    measurement_noise_scalar = private_nh.param<double>("measurement_noise_pix_cov", 100);
    process_noise_std = private_nh.param<double>("process_noise_std", 0.5);
    // [x, y, h_neck, h_waist, h_kneel, v_x, v_y]
    process_noise.setIdentity();
    process_noise.middleRows(0, 2) *= private_nh.param<double>("process_noise_pos_cov", 0.1);
    process_noise.middleRows(2, 2) *= private_nh.param<double>("process_noise_vel_cov", 0.1);
    // process_noise(4, 4) = 1e-10f;

    gt_normal.x() = private_nh.param<float>("gt_normal_x", 0.0);
    gt_normal.y() = private_nh.param<float>("gt_normal_y", -0.954);
    gt_normal.z() = private_nh.param<float>("gt_normal_z", 0.300);
    gt_distance = private_nh.param<float>("gt_dist", 0.19);

    joints_heights.reserve(4);
    joints_heights[0] = private_nh.param<float>("init_neck_height", 1.4);
    joints_heights[1] = private_nh.param<float>("init_waist_height", 0.94);
    joints_heights[2] = private_nh.param<float>("init_kneel_height", 0.53);
    joints_heights[3] = 0.0;
    // real_width = private_nh.param<float>("init_real_width", 0.56);

    camera_matrix = Eigen::Map<const Eigen::Matrix3d>(camera_info_msg->K.data()).transpose().cast<float>();
    update_matrices(ros::Time(0));
  }

  void update_matrices(const ros::Time& stamp) {
    camera2odom = lookup_eigen("base_link", camera_frame_id, stamp);  // my
    odom2camera = lookup_eigen(camera_frame_id, "base_link", stamp);
    odom2footprint = lookup_eigen("base_link", "base_link", stamp);
    footprint2base = lookup_eigen("base_link", "base_link", stamp);
    footprint2camera = lookup_eigen(camera_frame_id, "base_link", stamp);
  }

  Eigen::Isometry3f lookup_eigen(const std::string& to, const std::string& from, const ros::Time& stamp) {
    tf::StampedTransform transform;
    try{
      tf_listener->waitForTransform(to, from, stamp, ros::Duration(1.0));
      tf_listener->lookupTransform(to, from, stamp, transform);
    } catch (tf::ExtrapolationException& e) {
      tf_listener->waitForTransform(to, from, ros::Time(0), ros::Duration(5.0));
      tf_listener->lookupTransform(to, from, ros::Time(0), transform);
    }

    Eigen::Isometry3d iso;
    tf::transformTFToEigen(transform, iso);
    return iso.cast<float>();
  }

  Eigen::Vector3f transform_odom2camera(const Eigen::Vector3f& pos_in_odom) const {
    return (odom2camera * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
  }

  Eigen::Vector3f transform_odom2footprint(const Eigen::Vector3f& pos_in_odom) const {
    return (odom2footprint * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
  }

  Eigen::Vector3f transform_footprint2odom(const Eigen::Vector3f& pos_in_footprint) const {
    return (odom2footprint.inverse() * Eigen::Vector4f(pos_in_footprint.x(), pos_in_footprint.y(), pos_in_footprint.z(), 1.0f)).head<3>();
  }

  void set_dt(double d) {
    dt = std::max(d, 1e-9);
    float p_p = 0.25*pow(dt, 4)*process_noise_std;
    float v_v = pow(dt, 2)*process_noise_std;
    float p_v = 0.5*pow(dt, 3)*process_noise_std;
    process_noise(0,0) = process_noise(1,1) = p_p;
    process_noise(2,2) = process_noise(3,3) = v_v;
    process_noise(0,1) = process_noise(0,1) = p_v;
    process_noise(2,3) = process_noise(2,3) = p_v;
  }


  // interface for UKF
  Eigen::VectorXf f(const Eigen::VectorXf& state, const Eigen::VectorXf& control) const {
    Eigen::VectorXf next_state = state;
    // [x, y, v_x, v_y]
    
    next_state.middleRows(0, 2) += dt * state.middleRows(2, 2);
    return next_state;
  }

  Eigen::MatrixXf processNoiseCov() const {
    return process_noise;
  }

  // template<typename Measurement>
  Eigen::VectorXf h(const Eigen::VectorXf& state, const std::bitset<4> &visible_states) const;
  Eigen::Vector2f expected_measurement(const Eigen::VectorXf& state, const float &real_width) const ;

  // template<typename Measurement>
  Eigen::MatrixXf measurementNoiseCov(const std::bitset<4> &visible_states) const;

public:
  // Target model parameters
  std::vector<float> joints_heights;
  // float real_width;

  Eigen::Vector3f gt_normal;
  float gt_distance;

  double dt;
  Eigen::Isometry3f camera2odom;
  Eigen::Isometry3f odom2camera;
  Eigen::Isometry3f odom2footprint;
  Eigen::Isometry3f footprint2base;
  Eigen::Isometry3f footprint2camera;
  Eigen::Matrix3f camera_matrix;

  std::string camera_frame_id;
  std::shared_ptr<tf::TransformListener> tf_listener;

  // filter noise
  float measurement_noise_scalar, process_noise_std;  // neck, waist, kneel, ankle
  Eigen::Matrix<float, 4, 4> process_noise;
};

}

#endif // TRACKSYSTEM_CPP
