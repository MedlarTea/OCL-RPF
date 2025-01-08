#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <image_transport/image_transport.h>

#include <velodyne_pcl/point_types.h>
#include <velodyne_pointcloud/point_types.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/flann.h>
#include <pcl/common/transforms.h>

#include <mono_tracking/Box.h>
#include <mono_tracking/BoxArray.h>
#include <mono_tracking/Track.h>
#include <mono_tracking/TrackArray.h>


using namespace std;
typedef message_filters::sync_policies::ApproximateTime<mono_tracking::TrackArray, sensor_msgs::PointCloud2> slamSyncPolicy;
class EstimatorNode{
public:
    EstimatorNode(){
        // Syncronise camera(30Hz) and lidar(10Hz)
        // 木桶效应, 将以Lidar为下限进行消息接受
        tracks_sub_ = new message_filters::Subscriber<mono_tracking::TrackArray>(node_, _tracks_topic, 1);
        lidar_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, _lidar_topic, 1);
        sync_ = new  message_filters::Synchronizer<slamSyncPolicy>(slamSyncPolicy(5), *tracks_sub_, *lidar_sub_);
        sync_->registerCallback(boost::bind(&EstimatorNode::tracksLiDARCallback, this, _1, _2));

        ROS_INFO("SAVE IMAGES AND DISTANCES");
    }

    void tracksLiDARCallback(const mono_tracking::TrackArray::ConstPtr& tracks_msg, const sensor_msgs::PointCloud2::ConstPtr& pointcloud_msg){
        ROS_INFO("CALLBACK");
        // Get time
        ros::Time t = pointcloud_msg->header.stamp;
        string t_str = std::to_string(t.toSec());

        cv_bridge::CvImagePtr cv_image;
        try{
            cv_image = cv_bridge::toCvCopy(tracks_msg->image, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_xyzi_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_organized(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*pointcloud_msg, *pcl_cloud_xyzi_ptr);
        organized_pointcloud(pcl_cloud_xyzi_ptr, pcl_organized);

        // Calculate the distance between person and LiDAR
        double gt_x = 0.0;
        double gt_y = 0.0;
        int nums = 0;
        for (pcl::PointCloud<pcl::PointXYZI>::const_iterator it = pcl_organized->points.begin(); it != pcl_organized->points.end(); it++) {
            if (it->x < 0 || it->x > 9 || abs(it->y) > 0.5 || it->z < 0){
                continue;
            }
            gt_x += abs(it->x);
            gt_y += abs(it->y);
            nums++;
        }
        gt_x /= nums;
        gt_y /= nums;
        cv::Mat gt(1,2, CV_64F);
        gt.at<double>(0,0) = gt_x;
        gt.at<double>(0,1) = gt_y;

        // estimated distance
        cout << "es" << endl;
        double es_x = 0.0;
        double es_y = 0.0;
        cv::Mat es(1,2, CV_64F);
        int num=0;
        if(tracks_msg->tracks.size()==0){
            return;
        }
        for(const auto& track : tracks_msg->tracks) {
            if(num>0){
                continue;
            }
            if(track.box.box.empty()){
                return;
            }
            es_x = 917.8394165039062*0.60/abs(int(track.box.box[0])-int(track.box.box[2]));
            es_y = 0.0;
            num++;
            if(es_x>8){
                return;
            }
        }
        if(num==0){
            return;
        }

        es.at<double>(0,0) = es_x;
        es.at<double>(0,1) = es_y;

        // kalman filter distance
        cout << "kf" << endl;
        double kf_x = 0.0;
        double kf_y = 0.0;
        cv::Mat kf(1,2, CV_64F);
        num=0;
        for(const auto& track : tracks_msg->tracks) {
            if(num>0){
                continue;
            }
            kf_x = track.pos.x;
            kf_y = track.pos.y;
            num++;
        }
        if(num==0){
            return;
        }
        kf.at<double>(0,0) = kf_x;
        kf.at<double>(0,1) = kf_y;

        // save results
        cout << t_str << endl;
        
        cv::FileStorage fs(store_dir+t_str+".txt", cv::FileStorage::WRITE);
        fs << "gt" << gt;
        fs << "es" << es;
        fs << "kf" << kf;
        fs.release();
        cv::imwrite(store_dir+t_str+".jpg", cv_image->image);  // save original image
    }

    void organized_pointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr input_pointcloud, pcl::PointCloud<pcl::PointXYZI>::Ptr organized_pc){
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;

        // Kdtree to sort the point cloud
        kdtree.setInputCloud(input_pointcloud);

        pcl::PointXYZI searchPoint;// camera position as target
        searchPoint.x = 0.0f;
        searchPoint.y = 0.0f;
        searchPoint.z = 0.0f;

        int K = input_pointcloud->points.size();
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);

        // Sort the point cloud based on distance to the camera
        if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i) {
                pcl::PointXYZI point;
                point.x = input_pointcloud->points[pointIdxNKNSearch[i]].x;
                point.y = input_pointcloud->points[pointIdxNKNSearch[i]].y;
                point.z = input_pointcloud->points[pointIdxNKNSearch[i]].z;
                point.intensity = input_pointcloud->points[pointIdxNKNSearch[i]].intensity;
                // point.ring = input_pointcloud->points[pointIdxNKNSearch[i]].ring;
                organized_pc->points.push_back(point);
            }
        }
    }

public:
    string store_dir = "/home/jing/Data/Dataset/FOLLOWING/evaluation-of-width-distance/all2/";
    string _tracks_topic = "/mono_tracking/tracks";
    string _lidar_topic = "/velodyne_points";

private:
    ros::NodeHandle node_;
    message_filters::Subscriber<sensor_msgs::PointCloud2>* lidar_sub_;
    message_filters::Subscriber<mono_tracking::TrackArray>* tracks_sub_;
    message_filters::Synchronizer<slamSyncPolicy>* sync_;

};


int main(int argc, char** argv) {
  ros::init(argc, argv, "save_image_distance");
  std::unique_ptr<EstimatorNode> node(new EstimatorNode());
  ros::spin();

  return 0;
}