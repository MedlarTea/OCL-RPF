#include <iostream>
#include <Eigen/Dense>
#include <cppoptlib/problem.h>
#include <cppoptlib/solver/neldermeadsolver.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
class Projection : public cppoptlib::Problem<float>
{
public:
    Projection(const Eigen::Matrix3f& camera_matrix, const Eigen::Matrix<float, 4, 4>& footprint2camera, float _coeff, float _bias, const Eigen::Vector2f& _observation)
        : Tcf(footprint2camera),
          proj(camera_matrix * footprint2camera.block<3, 4>(0, 0)),
          coeff(_coeff), bias(_bias),
          observation(_observation)
    {}

    float value(const Eigen::VectorXf& x) 
    {
        float _error = error(x, observation);
        // cout << _error << endl;
        return _error;
    }

    void gradient(const Eigen::VectorXf& x, Eigen::VectorXf& grad) 
    {
        grad = grad_error(x, observation);
    }

private:
    // Return "box-center-u" value in the pixel frame
    float get_center_u(const Eigen::Vector3f& x_) const
    {
        Eigen::Vector4f x(x_[0], x_[1], x_[2], 1.0f);
        Eigen::Vector3f uvs = proj * x;
        cout << "x" << endl << x << endl;
        cout << "uvs" << endl << uvs << endl;
        return uvs[0] / uvs[2];
    }

    // Return: "box-width" value in the camera frame
    float get_width(const Eigen::Vector3f& x_) const
    {
        Eigen::Vector4f x(x_[0], x_[1], x_[2], 1.0f);
        Eigen::Vector4f pointCamera = Tcf * x;
        return (coeff*pointCamera[2]+bias);
    }

    // Return gradient (set the gradient of z is 0, so it will not be updated)
    // Derivative matrix of pixel-error to world point (from classic equation)
    Eigen::Matrix<float, 2, 3> grad_center_u(const Eigen::Vector3f& x_) const
    {
        Eigen::Matrix<float, 3, 4> P = proj;
        // P.col(1).setZero();
        P.col(2).setZero();  // Do not update z

        Eigen::Vector4f x(x_[0], x_[1], x_[2], 1.0f);
        Eigen::Vector3f l = P * x;

        auto lhs = P.block<2, 3>(0, 0) * l[2];
        auto rhs = l.head<2>() * P.block<1, 3>(2, 0);
        auto grad = (lhs - rhs) / (l[2] * l[2]);

        return grad;
    }

    // Derivative matrix of camera point to world point (my equation)
    Eigen::Matrix <float, 3, 3> grad_width(const Eigen::Vector3f& x_) const
    {
        Eigen::Matrix<float, 3, 3> R = Tcf.topLeftCorner(3,3);
        R.col(2).setZero();
        return (coeff*R);
    }


    
    float error(const Eigen::Vector3f& x, const Eigen::Vector2f& y) const
    {
        Eigen::Vector3f x_copy = x;
        float box_center_u = get_center_u(x_copy);
        float box_width = get_width(x_copy);

        Eigen::Vector2f estimated(box_center_u, box_width);

        return (estimated - y).squaredNorm();
    }

    Eigen::Vector3f grad_error(const Eigen::Vector3f& x, const Eigen::Vector2f& y) const
    {
        Eigen::Vector2f estimated;
        Eigen::Matrix<float, 2, 3> grads;

        Eigen::Vector3f x_copy = x;
        estimated[0] = get_center_u(x_copy);
        grads.block<1,3>(0,0) = grad_center_u(x_copy).block<1,3>(0,0);

        estimated[1] = get_width(x_copy);
        grads.block<1,3>(1,0) = grad_width(x_copy).block<1,3>(2,0);

        Eigen::Vector2f errors = 2*(estimated-y);

        return grads.transpose()*errors;
    }

private:
    float coeff, bias;
    Eigen::Matrix<float, 4, 4> Tcf;  // Transform_footprint^camera
    Eigen::Matrix<float, 3, 4> proj;  // robot->pixel
    Eigen::Vector2f observation;  // center_u,width
};

/**
ORIGINAL:
Obs: 
673
3.5
Person init state: 
 6.99219
       0
-21.1244
**/
void test01()
{
    Eigen::Matrix3f camera_matrix;
    camera_matrix << 914.046, 0, 637.45, 0, 911.948, 368.851, 0, 0, 1;
    Eigen::Matrix<float, 4, 4> Tcf;
    Tcf << 0.0040285, -1, -0.0106647, 0.0244131, -0.00153179, 0.0106386, -1, 0.879909, 1, 0.00404457, -0.00148883, 0.00134002, 0, 0, 0, 1;
    double _coeff = -48.494;
    double _bias = 349.495;
    Eigen::Vector2f _obs(673, 3.5);
    Projection f(camera_matrix, Tcf, _coeff, _bias, _obs);
    Eigen::VectorXf x0 = Eigen::Vector3f(0.0f, 0.0f, 0.0f);  // x0 is the person state in robot coordinate
    cppoptlib::NelderMeadSolver<Projection> solver;
    solver.minimize(f, x0);
    cout << "state" << endl;
    cout << x0 << endl;
}

void test02(){
    Eigen::Matrix3f camera_matrix;
    camera_matrix << 914.046, 0, 637.45, 0, 911.948, 368.851, 0, 0, 1;
    Eigen::Matrix<float, 4, 4> Tcf;
    Tcf << 0.0040285, -1, -0.0106647, 0.0244131, -0.00153179, 0.0106386, -1, 0.879909, 1, 0.00404457, -0.00148883, 0.00134002, 0, 0, 0, 1;
    Eigen::Vector2f _obs(673, 3.5);

    Eigen::Matrix2f A;
    A.block<1,2>(0,0) = Tcf.block<1,2>(0,0);
    A.block<1,2>(1,0) = Tcf.block<1,2>(2,0);
    Eigen::Vector2f b;
    b[0] = _obs[1]*(_obs[0]-camera_matrix(0,2))/camera_matrix(0,0)-Tcf(0,3);
    b[1] = _obs[1] - Tcf(2,3);
    Eigen::Vector2f state = A.inverse()*b;
    cout << "Tcf:" << endl << Tcf << endl;
    cout << "A:" << endl << A << endl;
    cout << "obs: " << endl << _obs << endl;
    cout << "state: " << endl << state << endl;
    // Eigen::Vector3f x;
    // x.setZero();
    // cout << x << endl;
    // Eigen::Vector4f stateW;
    // stateW.head<2>() = state;
    // stateW[2] = 0;
    // stateW[3] = 1;
    // cout << "Point in camera frame:" << endl << Tcf*stateW << endl;

}

void test03(){
    cv::KalmanFilter *kf = new cv::KalmanFilter(4, 2, 0);
    Eigen::MatrixXf transition_matrix = Eigen::MatrixXf::Identity(4,4);
    Eigen::MatrixXf process_noise = Eigen::MatrixXf::Identity(4,4);
    Eigen::MatrixXf measurement = Eigen::MatrixXf::Zero(4,4);
    transition_matrix(0,2) = transition_matrix(1,3) = 0.1f;
    measurement(0,0) = measurement(1,1) = 1.0f;
    cv::eigen2cv(transition_matrix, kf->transitionMatrix);
    cv::eigen2cv(process_noise, kf->processNoiseCov);
    cv::eigen2cv(measurement, kf->measurementMatrix);

    Eigen::Vector4f mean(3.1f, -0.086f, 0.0f, 0.0f);
    cv::eigen2cv(mean, kf->statePost);
    // cv::cv2eigen(kf->statePre, mean);

    
    cout << kf->statePost << endl;
    cout << kf->processNoiseCov << endl;
    cout << kf->transitionMatrix << endl;
    cout << kf->errorCovPost << endl;
    kf->predict();
    cout << kf->statePost << endl;
    cout << kf->errorCovPost << endl;

}

int main(){
    test03();

    return 0;
}