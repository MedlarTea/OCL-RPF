#include <memory>
#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/MarkerArray.h>
#include <mono_tracking/Box.h>
#include <mono_tracking/BoxArray.h>
#include <mono_tracking/Track.h>
#include <mono_tracking/TrackArray.h>
#include <mono_tracking/Person.h>
#include <mono_tracking/Persons.h>

#include <kkl/cvk/cvutils.hpp>
#include <kkl/math/gaussian.hpp>

#include <mono_tracking/observation.hpp>
#include <mono_tracking/people_tracker.hpp>

#include <spencer_tracking_msgs/TrackedPerson.h>
#include <spencer_tracking_msgs/TrackedPersons.h>
#include <spencer_tracking_msgs/Box.h>
#include <spencer_tracking_msgs/BodyPartElm.h>


using namespace std::chrono;
using namespace std;

namespace mono_tracking
{
class MonoTrackingNode
{
public:
    MonoTrackingNode()
    : nh(),
      private_nh("~"),
      poses_sub(nh.subscribe("/mono_detection/detections", 2, &MonoTrackingNode::observation_callback, this)),
      camera_info_sub(nh.subscribe("/camera/color/camera_info", 60, &MonoTrackingNode::camera_info_callback, this)),
      tracks_pub(private_nh.advertise<spencer_tracking_msgs::TrackedPersons>("/mono_tracking/tracks", 1)),
    //   markers_pub(private_nh.advertise<visualization_msgs::MarkerArray>("markers", 10)),
      image_trans(private_nh),
      image_pub(image_trans.advertise("/mono_tracking/image", 5))
    {
        ROS_INFO("Start Mono Tracking Node");
        color_palette = cvk::create_color_palette(16);
        tf_listener.reset(new tf::TransformListener());
        result_fname.open("/home/jing/Data/Projects/HumanFollowing/codes/mono_followingv2_ws/src/mono_tracking/src/debug/result.txt");
    }



private:
    void camera_info_callback(const sensor_msgs::CameraInfoConstPtr& _camera_info_msg) 
    {   
        if(camera_info_msg == nullptr)
        {
            ROS_INFO("camera_info");
            this->camera_info_msg = _camera_info_msg;
        }
    }

    bool check_IoU(vector<u_int16_t> box1_xywh, vector<u_int16_t> box2_xywh, const sensor_msgs::CameraInfoConstPtr& camera_info_msg){
        float IOU_THRESHOLD = 0.5;
        vector<u_int16_t> box1, box2;  // x1,y1,x2,y2
        box1.push_back(max(int(box1_xywh[0]-box1_xywh[2]/2),0));
        box1.push_back(max(int(box1_xywh[1]-box1_xywh[3]/2),0));
        box1.push_back(min(int(box1_xywh[0]+box1_xywh[2]/2),int(camera_info_msg->width-1)));
        box1.push_back(min(int(box1_xywh[1]+box1_xywh[3]/2),int(camera_info_msg->height-1)));

        box2.push_back(max(int(box2_xywh[0]-box2_xywh[2]/2),0));
        box2.push_back(max(int(box2_xywh[1]-box2_xywh[3]/2),0));
        box2.push_back(min(int(box2_xywh[0]+box2_xywh[2]/2),int(camera_info_msg->width-1)));
        box2.push_back(min(int(box2_xywh[1]+box2_xywh[3]/2),int(camera_info_msg->height-1)));

        u_int16_t xA = std::max(box1[0], box2[0]);
        u_int16_t yA = std::max(box1[1], box2[1]);
        u_int16_t xB = std::min(box1[2], box2[2]);
        u_int16_t yB = std::min(box1[3], box2[3]);

        u_int32_t interArea = max(0, xB-xA+1) * max(0, yB-yA+1);
        u_int32_t boxAArea = (box1[2]-box1[0]+1) * (box1[3]-box1[1]+1);
        u_int32_t boxBArea = (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1);

        float iou = interArea / float(boxAArea+boxBArea-interArea);
        return iou > IOU_THRESHOLD;
    }

    vector<bool> boxesCheck(const mono_tracking::BoxArrayConstPtr& boxes_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg){
        vector<bool> flags(boxes_msg->boxes.size(), true);
        // cout << "boxes: " << boxes_msg->boxes.size() << endl;
        // cout << "flags: " << flags.size() << endl;
        if(flags.size()<=1){
            return flags;
        }
        for(int i=0; i<(boxes_msg->boxes.size()-1); i++){
            for(int j=i+1; j<(boxes_msg->boxes.size()); j++){
                if(check_IoU(boxes_msg->boxes[i].box, boxes_msg->boxes[j].box, camera_info_msg)){
                    flags[i] = flags[j] = false;
                }
            }
        }
        // cout << "flags: " << flags.size() << endl;
        return flags;
    }

    void observation_callback(const mono_tracking::PersonsConstPtr& persons_msg)
    {
        auto start = high_resolution_clock::now();
        // ROS_INFO("Boxes_Callback");
        if(camera_info_msg == nullptr)
        {
            ROS_INFO("waiting for the camera info msg...");
            return;
        }
        // cout << "Observation" << endl;
        std::vector<Observation::Ptr> observations;
        observations.reserve(persons_msg->persons.size());
        for(const auto& person : persons_msg->persons) {
            // Joint: neck, lhip, rhip, lkneel, rkneel, lankle, rankle;
            vector<Joint> joints(7);
            // [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck]
            for(const auto& joint: person.body_part) {
                switch (joint.part_id) {
                case 7:
                    joints[1] = Joint(joint.confidence, joint.x, joint.y);  // LHip
                    break;
                case 8:
                    joints[2] = Joint(joint.confidence, joint.x, joint.y);  // RHip
                    break;
                case 9:
                    joints[3] = Joint(joint.confidence, joint.x, joint.y);  // LKnee
                    break;
                case 10:
                    joints[4] = Joint(joint.confidence, joint.x, joint.y);  // RKnee
                    break;
                case 11:
                    joints[5] = Joint(joint.confidence, joint.x, joint.y);  // LAnkle
                    break;
                case 12:
                    joints[6] = Joint(joint.confidence, joint.x, joint.y);  // RAnkle
                    break;
                case 13:
                    joints[0] = Joint(joint.confidence, joint.x, joint.y);  // Neck
                    break;
                }
            }

            auto observation = std::make_shared<Observation>(private_nh, joints, camera_info_msg, person);
            if(observation->is_valid()) {
                observations.push_back(observation);
            }
        }
        // result_fname << "observation nums: " << observations.size() << endl;
        // cout << "observation nums: " << observations.size() << endl;
        // cout << "visible: " << observations[0]->measurement_pair().first << endl;
        // cout << "measurement: " << observations[0]->measurement_pair().second << endl;
        if(!people_tracker) {
            // cout << "People tracker Init" << endl;
            people_tracker.reset(new PeopleTracker(private_nh, tf_listener, persons_msg->header.frame_id, camera_info_msg));
        }

        // update the tracker
        // cout << "People tracker Predict and Update" << endl;
        const auto& stamp = persons_msg->header.stamp;
        people_tracker->predict(private_nh, stamp);
        people_tracker->correct(private_nh, stamp, observations);
        
        const auto& people = people_tracker->get_people();
        result_fname << "Total people: " << people.size() << endl;
        // cout << "Total people: " << people.size() << endl;
        for(const auto& person : people) {
            result_fname << "Position: " << (*person).pos() << endl;
            // cout << "Position: " << (*person).pos() << endl;
            // cout << "Heights: " << (*person).heights() << endl;
            result_fname << "Velocity: " << (*person).vel() << endl;
            // cout << "Velocity: " << (*person).vel() << endl;
            // cout << "Trace: " << (*person).trace() << endl;
            if (person->get_last_associated()){
                if (person->get_last_associated()->min_distance)
                    result_fname << "Min distance: " << *(person->get_last_associated()->min_distance) << endl;
            }
        }
        
        // publish visualization msgs
        if(image_pub.getNumSubscribers()) {
                cv::Mat frame = cv::Mat(camera_info_msg->height, camera_info_msg->width, CV_8UC3, cv::Scalar::all(255));
                cv_bridge::CvImage cv_image(persons_msg->header, "bgr8");
                cv_image.image = visualize(frame, observations);
                image_pub.publish(cv_image.toImageMsg());
        }

        // publish tracks
        if(tracks_pub.getNumSubscribers())
            tracks_pub.publish(create_people_msgs(persons_msg->header.stamp));
        
        auto duration_time = duration_cast<milliseconds>(high_resolution_clock::now() - start);
        result_fname << "time: " << duration_time.count() << "ms" << endl; 
    }

    spencer_tracking_msgs::TrackedPersonsPtr create_people_msgs(const ros::Time& stamp){
        spencer_tracking_msgs::TrackedPersonsPtr tracked_msgs(new spencer_tracking_msgs::TrackedPersons());
        if(!people_tracker)
            return tracked_msgs;
        tracked_msgs->header.stamp = stamp;
        tracked_msgs->header.frame_id = "base_link";
        // cout << "peopleNums: " << people_tracker->get_people().size() << endl;
        std::cout << "--------------Tracks--------------" << std::endl;
        for(const auto& person : people_tracker->get_people())
        {
            if(!person->is_valid())
                continue;
            spencer_tracking_msgs::TrackedPersonPtr tracked_person(new spencer_tracking_msgs::TrackedPerson());
            tracked_person->track_id = person->id();
            tracked_person->is_occluded = false;
            tracked_person->is_matched = true;
            tracked_person->detection_id = person->id();
            Eigen::Vector2f pos = person->pos();
            Eigen::Vector2f vel = person->vel();

            tracked_person->pose.pose.position.x = pos.x();
            tracked_person->pose.pose.position.y = pos.y();
            tracked_person->pose.pose.position.z = 0.0;
            tracked_person->pose.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(vel(1),vel(0)));
            tracked_person->pose.covariance.fill(0);
            tracked_person->twist.twist.linear.x = vel(0);
            tracked_person->twist.twist.linear.y = vel(1);
            tracked_person->twist.twist.linear.z = 0.0;
            tracked_person->twist.covariance.fill(0);
            auto associated = person->get_last_associated();
            if (associated){
                spencer_tracking_msgs::BoxPtr box_ptr(new spencer_tracking_msgs::Box());
                Eigen::Vector4i box_in_observation = associated->bbox_vector();
                box_ptr->x = box_in_observation(0);
                box_ptr->y = box_in_observation(1);
                box_ptr->w = box_in_observation(2)-box_in_observation(0);
                box_ptr->h = box_in_observation(3)-box_in_observation(1);
                box_ptr->confidence = associated->bbox_confidence();
                tracked_person->bounding_box = *box_ptr;
                for (auto body_part: associated->body_parts){
                    spencer_tracking_msgs::BodyPartElm spencer_body_part;
                    spencer_body_part.part_id = body_part.part_id;
                    spencer_body_part.confidence = body_part.confidence;
                    spencer_body_part.x = body_part.x;
                    spencer_body_part.y = body_part.y;
                    tracked_person->body_parts.push_back(spencer_body_part);
                }
            }
            // tracked_person->box
            std::cout << "[" << tracked_person->track_id << "] "<< "(x y): " << tracked_person->pose.pose.position.x << " " << tracked_person->pose.pose.position.y << std::endl;
            tracked_msgs->tracks.push_back(*tracked_person);
        }
        return tracked_msgs;
    }

    cv::Mat visualize(const cv::Mat& frame, const std::vector<Observation::Ptr>& observations) const {
        cv::Mat canvas = frame.clone();

        cv::Mat layer(canvas.size(), CV_8UC3, cv::Scalar(0, 32, 0));
        cv::rectangle(layer, cv::Rect(100, 25, layer.cols - 200, layer.rows - 50), cv::Scalar(0, 0, 0), -1);
        canvas += layer;

        for(const auto& observation: observations) {
        if(observation->neck) {
            cv::circle(canvas, cv::Point(observation->neck->x(), observation->neck->y()), 5, cv::Scalar(0, 0, 255), -1);
        }
        if(observation->waist) {
            cv::circle(canvas, cv::Point(observation->waist->x(), observation->waist->y()), 5, cv::Scalar(0, 0, 255), -1);
        }
        if(observation->kneel) {
            cv::circle(canvas, cv::Point(observation->kneel->x(), observation->kneel->y()), 5, cv::Scalar(0, 0, 255), -1);
        }
        if(observation->ankle) {
            cv::circle(canvas, cv::Point(observation->ankle->x(), observation->ankle->y()), 5, cv::Scalar(0, 0, 255), -1);
        }
        }
        return canvas;
    }    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh;

    std::shared_ptr<tf::TransformListener> tf_listener;

    ros::Subscriber poses_sub;
    ros::Subscriber camera_info_sub;

    ros::Publisher tracks_pub;
    ros::Publisher markers_pub;

    image_transport::ImageTransport image_trans;
    image_transport::Publisher image_pub;

    boost::circular_buffer<cv::Scalar> color_palette;

    sensor_msgs::CameraInfoConstPtr camera_info_msg;

    std::shared_ptr<TrackSystem> track_system;
    std::unique_ptr<PeopleTracker> people_tracker;

    ofstream result_fname;

    int messageNums = 0;
};
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "mono_tracking");
  std::unique_ptr<mono_tracking::MonoTrackingNode> node(new mono_tracking::MonoTrackingNode());
  ros::spin();

  return 0;
}