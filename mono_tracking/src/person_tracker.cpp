#include <mono_tracking/person_tracker.hpp>

#include <cppoptlib/problem.h>
#include <cppoptlib/solver/neldermeadsolver.h>
#include "ceres/ceres.h"

#include <kkl/alg/unscented_kalman_filter.hpp>
#include <mono_tracking/track_system.hpp>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using namespace std;

namespace mono_tracking {

PersonTracker::PersonTracker(ros::NodeHandle& nh, const std::shared_ptr<TrackSystem>& track_system, const ros::Time& stamp, long id, const Observation::Ptr& observation)
	: id_(id)
{
	validation_correction_count = nh.param<int>("validation_correction_cound", 5);
	
	const auto &measurement = observation->measurement_pair();
	const auto &box = observation->bbox_vector();
	// [x,y,v_x,v_y]
	Eigen::VectorXf mean = Eigen::VectorXf::Zero(4);
	// initialize the target model with the first target
	if (id_ == 0 && measurement.first.count()==4){
		Eigen::Matrix<float, 5, 1> result = estimate_init_state(track_system, measurement);
		mean.head<2>() = result.head<2>();
		track_system->joints_heights[0] = result(2);
		track_system->joints_heights[1] = result(3);
		track_system->joints_heights[2] = result(4);
		Eigen::Vector4f pos_in_cam = track_system->footprint2camera * Eigen::Vector4f(mean(0), mean(1), 0.0, 1.0);
		real_width=  (box(2)-box(0)) * pos_in_cam(2) / track_system->camera_matrix(0,0);
		cout << "Real width: " << real_width << endl;
	}
	else{
		cout << "Init person tracker" << endl;
		mean.head<2>() = estimate_init_state_by_single(track_system, measurement);
		Eigen::Vector4f pos_in_cam = track_system->footprint2camera * Eigen::Vector4f(mean(0), mean(1), 0.0, 1.0);
		real_width =  (box(2)-box(0)) * pos_in_cam(2) / track_system->camera_matrix(0,0);
	}
	// else{

	// }
	// mean[2] = 1.4f;

	Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(4, 4) * nh.param<double>("init_cov_scale", 1.0);

	// self.alpha**2 * (n +self.kappa) - n = 1**2 * (7-4) - 7 = 
	ukf.reset(new UnscentedKalmanFilter(track_system, mean, cov, 1.0));

	prev_stamp = stamp;
	correction_count_ = 0;
	prediction_count_ = 0;
	last_associated = nullptr;
}

PersonTracker::~PersonTracker() {

}

void PersonTracker::predict(const ros::Time& stamp) {
	expected_measurement_dist = boost::none;
	last_associated = nullptr;

	ukf->system->set_dt((stamp - prev_stamp).toSec());
	// ukf->system->set_dt(0.001);
	// cout << "dt: " << ukf->system->dt << endl;
	prev_stamp = stamp;
	prediction_count_ ++;
	ukf->predict(Eigen::Vector2f::Zero());
}

void PersonTracker::correct(const ros::Time& stamp, const Observation::Ptr& observation) {
	expected_measurement_dist = boost::none;
	last_associated = observation;
	correction_count_ ++;
	prediction_count_ --;
	ukf->correct(observation->measurement_pair());
	// outlier
	const auto &box = observation->bbox_vector();
	Eigen::Vector4f pos_in_cam = ukf->system->footprint2camera * Eigen::Vector4f(ukf->mean(0), ukf->mean(1), 0.0, 1.0);
	real_width = (box(2)-box(0)) * pos_in_cam(2) / ukf->system->camera_matrix(0,0);
}


Eigen::Vector2f PersonTracker::expected_measurement_distribution(const float &real_width) const {
	expected_measurement_dist = ukf->expected_measurement_distribution(real_width);
	return *expected_measurement_dist;
}

double PersonTracker::trace() const {
	return ukf->cov.trace();
}

Eigen::Vector2f PersonTracker::pos() const {
	return ukf->mean.head<2>();
}

float PersonTracker::real_width_value() const {
	return real_width;
}

Eigen::Vector2f PersonTracker::vel() const {
	return ukf->mean.tail<2>();
}

Eigen::MatrixXf PersonTracker::cov() const {
	return ukf->cov;
}

// struct MyCostFunctorWithKnowHeight{
// public:
// 	MyCostFunctor(const Eigen::Matrix<double, 3, 4> proj_matrix): proj_matrix(proj_matrix) {}
// 	template <typename T>
//     bool operator()(const T* const x , const T* const y, T* residual) const {

// 	}
// }

struct MyCostFunctor{
public:
	MyCostFunctor(const Eigen::Matrix<double, 3, 4> proj_matrix): proj_matrix(proj_matrix) {}
	template <typename T>
    bool operator()(const T* const x , const T* const y, T* residual) const {
        T neck_3d[3] = {x[0], x[1], x[2]};
        T neck_coord[2];
        project(neck_3d, neck_coord);

        T waist_3d[3] = {x[0], x[1], x[3]};
        T waist_coord[2];
        project(waist_3d, waist_coord);

        T kneel_3d[3] = {x[0], x[1], x[4]};
        T kneel_coord[2];
        project(kneel_3d, kneel_coord);

        T ankle_3d[3] = {x[0], x[1], T(0.0)};  // assume z=0, on the ground
        T ankle_coord[2];
        project(ankle_3d, ankle_coord);

        residual[0] = y[0] - neck_coord[0];
        residual[1] = y[1] - neck_coord[1];
        residual[2] = y[2] - waist_coord[0];
        residual[3] = y[3] - waist_coord[1];
        residual[4] = y[4] - kneel_coord[0];
        residual[5] = y[5] - kneel_coord[1];
        residual[6] = y[6] - ankle_coord[0];
        residual[7] = y[7] - ankle_coord[1];
        return true;
    }
    template <typename T>
    void project(const T x[3], T result[2]) const{
        T proj_3d[3];
        proj_3d[0] = proj_matrix(0,0)*x[0] + proj_matrix(0,1)*x[1] + proj_matrix(0,2)*x[2] + proj_matrix(0,3);
        proj_3d[1] = proj_matrix(1,0)*x[0] + proj_matrix(1,1)*x[1] + proj_matrix(1,2)*x[2] + proj_matrix(1,3);
        proj_3d[2] = proj_matrix(2,0)*x[0] + proj_matrix(2,1)*x[1] + proj_matrix(2,2)*x[2] + proj_matrix(2,3);
        result[0] = proj_3d[0] / proj_3d[2];
        result[1] = proj_3d[1] / proj_3d[2];
    }

private:
    const Eigen::Matrix<double, 3, 4> proj_matrix;
};

Eigen::Matrix<float,5,1> PersonTracker::estimate_init_state(const std::shared_ptr<TrackSystem>& track_system,  const std::pair<std::bitset<4>, Eigen::MatrixXf> &measurement){
	Eigen::Matrix<float, 3, 4> project_matrix = track_system->camera_matrix * track_system->footprint2camera.matrix().block<3, 4>(0, 0);
	double x[5] = {5.0, 0.0, 1.4, 0.9, 0.5};
	Eigen::MatrixXf neck_waist_kneel_ankle = measurement.second;
	double observation[8] = {neck_waist_kneel_ankle(0,0), neck_waist_kneel_ankle(1,0), neck_waist_kneel_ankle(2,0), neck_waist_kneel_ankle(3,0), neck_waist_kneel_ankle(4,0), neck_waist_kneel_ankle(5,0), neck_waist_kneel_ankle(6,0), neck_waist_kneel_ankle(7,0)};
	
	// Build the problem.
	Problem problem;

	//Construct model
	MyCostFunctor functor(project_matrix.cast<double>());
	CostFunction* cost_function = new AutoDiffCostFunction<MyCostFunctor, 8, 5, 8>(&functor, ceres::DO_NOT_TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_function, nullptr, x, observation);

	// Run the solver!
	Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	Solver::Summary summary;
	Solve(options, &problem, &summary);

	Eigen::Map<Eigen::Matrix<double, 5, 1>> result = Eigen::Map<Eigen::Matrix<double, 5, 1>>(x, 5, 1);
	return result.cast<float>();
}

Eigen::Vector2f PersonTracker::estimate_init_state_by_single(const std::shared_ptr<TrackSystem>& track_system,  const std::pair<std::bitset<4>, Eigen::MatrixXf> &measurement){
	// by single measurement
	// int indicator = 0;
	for (int i = 0; i < measurement.first.size(); i++){
		if (measurement.first[i] == false)
			continue;
		Eigen::Vector3f obs_measurement = Eigen::Vector3f::Ones();  // (3,1)
		obs_measurement.head<2>() = measurement.second.block<2,1>(0,0);
		Eigen::MatrixXf cam_inv = track_system->camera_matrix.inverse();  // (3,3)
		Eigen::Vector3f fake_ray = cam_inv * obs_measurement;
		float den = fake_ray.transpose() * track_system->gt_normal;
		float residual_dist = track_system->joints_heights[i]>track_system->gt_distance ? (track_system->joints_heights[i]-track_system->gt_distance):track_system->gt_distance;
		Eigen::Vector4f vec_in_cam = Eigen::Vector4f::Ones();
		vec_in_cam.head<3>() = fake_ray * residual_dist / den;
		Eigen::Vector4f vec_in_base = track_system->footprint2camera.matrix().inverse() * vec_in_cam;
		Eigen::Vector2f result = vec_in_base.head<2>();
		return result;
	}
}

}
