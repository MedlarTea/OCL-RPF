#include <mono_tracking/track_system.hpp>
using namespace std;

namespace mono_tracking{

    Eigen::VectorXf TrackSystem::h(const Eigen::VectorXf& state, const std::bitset<4> &visible_states) const {
        const int visible_counts = visible_states.count();
        Eigen::VectorXf estimated_observation(visible_counts*2);
        int indicator = 0;
        for (int i=0; i < visible_states.size(); i++){
            if (visible_states[i] == false)
                continue;
            Eigen::Vector4f pos_3d;
            pos_3d = odom2camera * footprint2base * Eigen::Vector4f(state[0], state[1], joints_heights[i], 1.0f);
            Eigen::Vector3f pos_2d = camera_matrix * pos_3d.head<3>();
            estimated_observation.block<2,1>(indicator*2,0) = pos_2d.head<2>() / pos_2d.z();
            indicator++;
        }
        return estimated_observation;
    }

    Eigen::Vector2f TrackSystem::expected_measurement(const Eigen::VectorXf& state, const float &real_width) const {
        Eigen::Vector4f pos_in_cam;
        Eigen::Vector2f expected_measurement;
        pos_in_cam = odom2camera * footprint2base * Eigen::Vector4f(state[0], state[1], 0.0f, 1.0f);
        Eigen::Vector3f coord_homo = camera_matrix * pos_in_cam.head<3>();
        expected_measurement.x() = coord_homo.x() / coord_homo.z();  // the center_u of the bounding box

        expected_measurement.y() = camera_matrix(0,0) * real_width / pos_in_cam.z();  // the with of the bounding box

        return expected_measurement;
    }

    Eigen::MatrixXf TrackSystem::measurementNoiseCov(const std::bitset<4> &visible_states) const {
        // TODO: change runtime-determination to compile-determination
        const int visible_counts = visible_states.count();
        Eigen::MatrixXf measurement_noise(visible_counts*2, visible_counts*2);
        measurement_noise.setIdentity();
        measurement_noise *= measurement_noise_scalar;
        return measurement_noise;
    }
}