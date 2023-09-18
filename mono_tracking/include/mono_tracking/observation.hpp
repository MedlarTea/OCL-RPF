#ifndef MONOCULAR_PEOPLE_TRACKING_OBSERVATION_HPP
#define MONOCULAR_PEOPLE_TRACKING_OBSERVATION_HPP

#include <memory>
#include <Eigen/Dense>
#include <boost/optional.hpp>
#include <bitset>

#include <ros/node_handle.h>
#include <mono_tracking/Person.h>
#include <mono_tracking/BodyPartElm.h>
#include <sensor_msgs/CameraInfo.h>
using namespace std;

namespace mono_tracking {

struct Joint {
public:
    Joint()
    : confidence(0.0),
      x(0.0),
      y(0.0)
    {}

    Joint(float confidence, float x, float y)
      : confidence(confidence),
        x(x),
        y(y)
    {}

    float confidence;
    float x;
    float y;
};

struct Observation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Observation>;

  /** 
   * joints: [neck, lhip, rhip, lkneel, rkneel, lankle, rankle]
  */
    Observation(ros::NodeHandle& private_nh, const std::vector<Joint> &joints, const sensor_msgs::CameraInfoConstPtr& camera_info_msg, const mono_tracking::Person& person_msg)
      : person_msg(person_msg)
    {
        // cout << "Obs 1" << endl;

        const double confidence_thresh = private_nh.param<double>("detection_confidence_thresh", 0.3);
        border_thresh_w = private_nh.param<int>("detection_border_thresh_w", 100);
        border_thresh_h = private_nh.param<int>("detection_border_thresh_h", 25);
        image_w = camera_info_msg->width;
        image_h = camera_info_msg->height;

        // init visible states
        std::bitset<4> visible_states;  // []
        visible_states.reset();

        // measurement matrix
        Eigen::Matrix<float, 8, 1> observation_buffer;

        // joints [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
        body_parts = person_msg.body_part;

        //bbox
        bbox = Eigen::Vector4i(person_msg.box[0],person_msg.box[1],person_msg.box[2],person_msg.box[3]);
        confidence = person_msg.score;
        float obs_x = (person_msg.box[0]+person_msg.box[2])/2;

        // cout << "Obs neck" << endl;
        // Neck
        if(joints[0].confidence > confidence_thresh) {
            neck = Eigen::Vector2f(obs_x, joints[0].y);
        }
        if (neck && is_near_border((*neck)(0), (*neck)(1), image_h, image_w, border_thresh_h, border_thresh_w))
            neck.reset();
        else if (neck){
            visible_states.set(0);
            // (*measurement).first[0] = true;
            observation_buffer.block<2,1>(0,0) = *neck;
        }
        
        // cout << "Obs waist" << endl;
        // waist
        if(joints[1].confidence > confidence_thresh && joints[2].confidence > confidence_thresh) {
            waist = Eigen::Vector2f(obs_x, (joints[1].y + joints[2].y)/2.0f);
        } else if (joints[1].confidence > confidence_thresh) {
            waist = Eigen::Vector2f(obs_x, joints[1].y);
        } else if (joints[2].confidence > confidence_thresh) {
            waist = Eigen::Vector2f(obs_x, joints[2].y);
        }
        if (waist && is_near_border((*waist)(0), (*waist)(1), image_h, image_w, border_thresh_h, border_thresh_w))
            waist.reset();
        else if (waist){
            visible_states.set(1);
            // (*measurement).first[1] = true;
            observation_buffer.block<2,1>(2,0) = *waist;
        }

        // cout << "Obs kneel" << endl;
        // kneel
        if(joints[3].confidence > confidence_thresh && joints[4].confidence > confidence_thresh) {
            kneel = Eigen::Vector2f(obs_x, (joints[3].y + joints[4].y)/2.0f);
        } else if (joints[3].confidence > confidence_thresh) {
            kneel = Eigen::Vector2f(obs_x, joints[3].y);
        } else if (joints[4].confidence > confidence_thresh) {
            kneel = Eigen::Vector2f(obs_x, joints[4].y);
        }
        if (kneel && is_near_border((*kneel)(0), (*kneel)(1), image_h, image_w, border_thresh_h, border_thresh_w))
            kneel.reset();
        else if (kneel){
            visible_states.set(2);
            // (*measurement).first[2] = true;
            observation_buffer.block<2,1>(4,0) = *kneel;
        }

        // cout << "Obs ankle" << endl;
        // ankle, use bbox coord as ankle point
        // ankle
        if(joints[5].confidence > confidence_thresh && joints[6].confidence > confidence_thresh) {
            ankle = Eigen::Vector2f(obs_x, (joints[5].y + joints[6].y)/2.0f);
        } else if (joints[5].confidence > confidence_thresh) {
            ankle = Eigen::Vector2f(obs_x, joints[5].y);
        } else if (joints[6].confidence > confidence_thresh) {
            ankle = Eigen::Vector2f(obs_x, joints[6].y);
        }
        if (ankle && is_near_border((*ankle)(0), (*ankle)(1), image_h, image_w, border_thresh_h, border_thresh_w))
            ankle.reset();
        else if (ankle){
            visible_states.set(3);
            observation_buffer.block<2,1>(6,0) = *ankle;
        }

        Eigen::MatrixXf observation(visible_states.count()*2, 1);
        int indicator = 0;
        for (int i=0; i < visible_states.size(); i++){
            if (visible_states[i]==false)
                continue;
            observation.block<2,1>(indicator*2,0) = observation_buffer.block<2,1>(i*2,0);
            indicator++;
        }

        if (visible_states.count() != 0){
            measurement = std::make_pair(visible_states, observation);
            valid = true;
        }
        else
            valid = false;
    }

  bool is_valid() const {
    return  valid;
  }

  bool is_near_border(float &x, float &y, int &img_height, int &img_width, int &border_thresh_h, int &border_thresh_w) const {
    if(x < border_thresh_w || x > img_width - border_thresh_w || y < border_thresh_h || y > img_height - border_thresh_h )
      return true;
    return false;
  }

  Eigen::Vector4f neck_ankle_vector() const {
    Eigen::Vector4f x;
    x.head<2>() = *neck;
    x.tail<2>() = *ankle;
    return  x;
  }

  Eigen::Vector2f neck_vector() const {
    return  *neck;
  }
  Eigen::Vector2f waist_vector() const {
    return  *waist;
  }
  Eigen::Vector2f kneel_vector() const {
    return  *kneel;
  }
  Eigen::Vector2f ankle_vector() const {
    return  *ankle;
  }
  Eigen::Vector4i bbox_vector() const {
    return *bbox;
  }
  float bbox_confidence() const {
    return *confidence;
  }

  std::pair<std::bitset<4>, Eigen::MatrixXf> measurement_pair() const {
    return *measurement;
  }

  const mono_tracking::Person& person_msg;

  bool valid;
  int border_thresh_w, border_thresh_h, image_w, image_h;
  boost::optional<Eigen::Vector4i> bbox;
  boost::optional<Eigen::Vector2f> neck;
  boost::optional<Eigen::Vector2f> waist;
  boost::optional<Eigen::Vector2f> kneel;
  boost::optional<Eigen::Vector2f> ankle;
  boost::optional<std::pair<std::bitset<4>, Eigen::MatrixXf>> measurement;

  boost::optional<double> min_distance;
  boost::optional<float> confidence;
  std::vector<mono_tracking::BodyPartElm> body_parts;
  // use a hex to store the situation of current visible joints
  
};

}

#endif // OBSERVATION_HPP
