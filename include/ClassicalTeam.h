#pragma once
#include <Eigen/Core>
#include "SimState.h"

class ClassicalTeam
{
public:
    ClassicalTeam();
    ClassicalTeam(int team);
    ~ClassicalTeam();
    void reset();
    void runControl(const double &t, const SimState &state, Eigen::Vector2d &vel1, Eigen::Vector2d &vel2);
private:
    int team_;
};
