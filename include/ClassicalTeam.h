#pragma once
#include <Eigen/Core>
#include <vector>
#include "SimState.h"

class ClassicalTeam
{
    void planTraj(const Eigen::Vector4d &x0, const Eigen::Vector4d &xT, const double &dt, const int &T, std::vector<Eigen::Vector2d> &vels);
public:
    ClassicalTeam();
    ClassicalTeam(int team);
    ~ClassicalTeam();
    void reset();
    void runControl(const double &t, const SimState &state, Eigen::Vector2d &vel1, Eigen::Vector2d &vel2);
private:
    int team_;
};
