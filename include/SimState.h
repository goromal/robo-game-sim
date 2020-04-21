#pragma once
#include <Eigen/Core>
#include "utils.h"
#include <vector>

enum {PX = 0, PY = 1, VX = 2, VY = 3};
enum {TEAM_A = 0, TEAM_B = 1};

using namespace Eigen;

struct SimState
{
    enum {
        TAS = 0,
        TBS = 1,
        PK = 2,
        A1 = 6,
        A2 = 10,
        B1 = 14,
        B2 = 18,
        SIZE = 22
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<double, SIZE, 1> arr;
    double& TeamAScore;
    double& TeamBScore;
    Eigen::Map<Eigen::Vector4d> x_ball;
    Eigen::Map<Eigen::Vector4d> x_A1;
    Eigen::Map<Eigen::Vector4d> x_A2;
    Eigen::Map<Eigen::Vector4d> x_B1;
    Eigen::Map<Eigen::Vector4d> x_B2;

    SimState() :
        TeamAScore(*(arr.data()+TAS)),
        TeamBScore(*(arr.data()+TBS)),
        x_ball(arr.data()+PK),
        x_A1(arr.data()+A1),
        x_A2(arr.data()+A2),
        x_B1(arr.data()+B1),
        x_B2(arr.data()+B2)
    {
        arr.setZero();
    }

    Eigen::Matrix<double, SIZE, 1> vector()
    {
        return arr;
    }
};

