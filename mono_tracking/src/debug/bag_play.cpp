#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>


int main(int argc, char** argv) {

    rosbag::Bag bag;
    bag.open("/home/jing/Data/Projects/HumanFollowing/datasets/evaluation-of-width-distance/2022-07/mocap/2022-07-15-17-09-34.bag");

    for(rosbag::MessageInstance const m: rosbag::View(bag)){
        std_msgs::Int32::ConstPtr i = m.instantiate<std_msgs::Int32>();
        if (i != nullptr)
            std::cout << i->data << std::endl;
    }
    bag.close();
    return 0;
}