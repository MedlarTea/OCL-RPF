#ifndef MONO_FOLLOWING_TRACKLET_HPP
#define MONO_FOLLOWING_TRACKLET_HPP

#include <vector>
#include <queue>
#include <Eigen/Dense>
#include <boost/optional.hpp>
#include <opencv2/opencv.hpp>

#include <tf/transform_listener.h>
#include <mono_tracking/Track.h>
// #include <mono_following/Box.h>
// #include <ccf_person_identification/online_classifier.hpp>
using namespace std;
namespace mono_following {

struct Tracklet {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Ptr = std::shared_ptr<Tracklet>;

    Tracklet(tf::TransformListener& tf_listener, const std_msgs::Header& header, const mono_tracking::Track& track_msg, const cv::Mat &image)
        : confidence(boost::none),
          track_msg(&track_msg)
    {
        geometry_msgs::PointStamped point;
        point.header = header;
        point.point = track_msg.pos;
        // cout << "I1" << endl;
        geometry_msgs::PointStamped transformed;
        tf_listener.transformPoint("base_link", point, transformed);

        pos_in_baselink = Eigen::Vector2f(transformed.point.x, transformed.point.y);
        // cout << "I2" << endl;

        if(!track_msg.box.box.empty()){
            for (auto x:track_msg.box.box){
                region.push_back(x);
            }   
            // cout << "I3" << endl;

            image_patch = _resize(image, region);
            // cout << "COLS: " << image_patch->cols << " ROWS: " << image_patch->rows << endl;
            // isTarget = (track_msg.id == traget_id);
        }
    }

public:
    // features
    // bool isTarget=false;
    std::vector<u_int16_t> region;  // x1, y1, x2, y2
    boost::optional<cv::Mat> image_patch;
    boost::optional<cv::Mat> descriptor;

    boost::optional<double> confidence;  // cos similarity with target features
    std::vector<double> classifier_confidences;

    Eigen::Vector2f pos_in_baselink;
    const mono_tracking::Track* track_msg;
private:
    int down_width = 64;
    int down_height = 128;
    cv::Mat _resize(const cv::Mat &_image, const std::vector<u_int16_t> &box){
        cv::Mat image = _image.clone();
        // cout << "COLS: " << image.cols << " ROWS: " << image.rows << endl;
        // cout << region.size() << endl;
        // cout << box[0] << " " << box[1] << " " << box[2] << " " <<box[3]<<endl;
        cv::Mat cropped_image = image(cv::Range(box[1],box[3]), cv::Range(box[0],box[2]));  // (y1,y2) (x1,x2)
        // cout << "COLS: " << cropped_image.cols << " ROWS: " << cropped_image.rows << endl;
        cv::Mat resized_image;
        cv::resize(cropped_image, resized_image, cv::Size(down_width, down_height), cv::INTER_LINEAR);
        // cout << "COLS: " << resized_image.cols << " ROWS: " << resized_image.rows << endl;
        return resized_image;
    }
    
};

}

#endif // TRACKLET_HPP
