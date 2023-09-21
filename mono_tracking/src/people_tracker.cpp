#include <mono_tracking/people_tracker.hpp>

#include <kkl/alg/data_association.hpp>
#include <kkl/alg/nearest_neighbor_association.hpp>

#include <mono_tracking/track_system.hpp>


namespace mono_tracking {

PeopleTracker::PeopleTracker(ros::NodeHandle& private_nh, const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
    id_gen = 0;
    remove_counts_thresh = private_nh.param<int>("tracking_remove_counts_thresh", 5.0);
    remove_trace_thresh = private_nh.param<double>("tracking_remove_trace_thresh", 5.0);
    dist_to_exists_thresh = private_nh.param<double>("tracking_newtrack_dist2exists_thersh", 100.0);

    data_association.reset(new kkl::alg::NearestNeighborAssociation<PersonTracker::Ptr, Observation::Ptr, AssociationDistance>(AssociationDistance(private_nh)));
    track_system.reset(new TrackSystem(private_nh, tf_listener, camera_frame_id, camera_info_msg));
}

PeopleTracker::~PeopleTracker() {

}

void PeopleTracker::predict(ros::NodeHandle& nh, const ros::Time& stamp) {
    // track_system->update_matrices(stamp);
    // cout << stamp.toSec() << endl;
    for(const auto& person : people) {
        // cout << "Predict A" << endl;
        person->predict(stamp);
    }
}

void PeopleTracker::correct(ros::NodeHandle& nh, const ros::Time& stamp, const std::vector<Observation::Ptr>& observations) {
    if(!observations.empty()) {

        // cout << "correct A" << endl;
        std::vector<bool> associated(observations.size(), false);
        auto associations = data_association->associate(people, observations);
        for(const auto& assoc : associations) {
            associated[assoc.observation] = true;
            // cout << "correct B" << endl;
            people[assoc.tracker]->correct(stamp, observations[assoc.observation]);
        }

        for(int i=0; i<observations.size(); i++) {
            // if(!associated[i] && observations[i]->ankle) {
            // cout << "HI" << endl;
            const auto &measurement_pair = observations[i]->measurement_pair();
            // cout << "People correct " << " " << measurement_pair.first.count() << endl;
            // if(!associated[i] && measurement_pair.first.count() == 4) {
            if(!associated[i]) {
                if(observations[i]->min_distance && *observations[i]->min_distance < dist_to_exists_thresh) {
                    continue;
                }
                // cout << "Init" << endl;
                // cout << "visible: " << measurement_pair.first << endl;
                // for (int i = 0; i <measurement_pair.first.count(); i++)
                //     cout << "Measurement: " << measurement_pair.second.block<2,1>(i*2, 0) << endl;
                // if(observations[i]->close2border) {
                //     continue;
                // }

                PersonTracker::Ptr tracker(new PersonTracker(nh, track_system, stamp, id_gen++, observations[i]));
                // DO NOT init person tracker that is far away from the robot
                Eigen::Vector2f person_pos = tracker->pos();
                if (person_pos(0)>0 && sqrt(person_pos(0)*person_pos(0) + person_pos(1)*person_pos(1)) < 10.0 ){
                    tracker->correct(stamp, observations[i]);
                    people.push_back(tracker);
                }
                // cout << "correct C" << endl;
                // cout << 
            }
        }
    }
    
    auto remove_loc = std::partition(people.begin(), people.end(), [&](const PersonTracker::Ptr& tracker) {
        // return tracker->trace() < remove_trace_thresh;
        // cout << tracker->prediction_count() << " " << remove_counts_thresh;
        return tracker->prediction_count() < remove_counts_thresh;
    });
    // removed_people.clear();
    // std::copy(remove_loc, people.end(), std::back_inserter(removed_people));
    people.erase(remove_loc, people.end());
}

}
