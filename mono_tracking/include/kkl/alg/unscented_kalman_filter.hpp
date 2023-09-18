/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <memory>
#include <Eigen/Dense>

namespace kkl {
  namespace alg {

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  UnscentedKalmanFilterX(const std::shared_ptr<System>& system, const VectorXt& mean, const MatrixXt& cov, double _lambda)
    : mean(mean),
    cov(cov),
    system(system),
    lambda(_lambda)
  {
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict(const VectorXt& control) {
    const int N = mean.rows();
    VectorXt weights = calcWeights(N);

    // calculate sigma points
    ensurePositiveFinite(cov);
    MatrixXt sigma_points = computeSigmaPoints(mean, cov);
    for (int i = 0; i < 2 * N + 1; i++) {
      sigma_points.row(i) = system->f(sigma_points.row(i), control);
    }

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < 2 * N + 1; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < 2 * N + 1; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += system->processNoiseCov();

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement object, not a simple vector
   */
  // template<typename Measurement>
  void correct(const std::pair<std::bitset<4>, Eigen::MatrixXf>& measurement) {
    const int N = mean.rows();
    const int K = measurement.second.rows();

    VectorXt ext_weights = calcWeights(N + K);

    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    ext_cov_pred.bottomRightCorner(K, K) = system->measurementNoiseCov(measurement.first);

    ensurePositiveFinite(ext_cov_pred);
    MatrixXt ext_sigma_points = computeSigmaPoints(ext_mean_pred, ext_cov_pred);

    // unscented transform
    MatrixXt expected_measurements(2 * (N + K) + 1, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system->h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1), measurement.first);
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    MatrixXt kalman_gain = sigma * expected_measurement_cov.inverse();

    VectorXt ext_mean = ext_mean_pred + kalman_gain * (measurement.second - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - kalman_gain * expected_measurement_cov * kalman_gain.transpose();

    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  // template<typename Measurement>
  Eigen::Vector2f expected_measurement_distribution(const float &real_width) const {
    Eigen::Vector2f expected_measurement;
    expected_measurement = system->expected_measurement(mean, real_width);
    return expected_measurement;
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }

  System& getSystem() { return *system; }
  const System& getSystem() const { return *system; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  /**
   * @brief calculate weights for sigma points
   * @param N   dimension of state
   * @return weights (2 * N + 1)
   */
  VectorXt calcWeights(int N) const {
    VectorXt weights(2 * N + 1, 1);
    weights[0] = lambda / (N + lambda);
    for(int i=1; i<2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }
    return weights;
  }

  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @return calculated sigma points
   */
  MatrixXt computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov) const {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    MatrixXt sigma_points(2 * n + 1, n);

    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }

    return sigma_points;
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) const {
    return;
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt D = solver.pseudoEigenvalueMatrix();
    MatrixXt V = solver.pseudoEigenvectors();
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }

    cov = V * D * V.inverse();
  }

public:
  const T lambda;
  std::shared_ptr<System> system;

  VectorXt mean;
  MatrixXt cov;
};

  }
}


#endif
