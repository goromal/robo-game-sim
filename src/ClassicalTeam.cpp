#include "ClassicalTeam.h"

ClassicalTeam::ClassicalTeam()
{
    // ++++
}

ClassicalTeam::~ClassicalTeam()
{
    // ++++
}

void ClassicalTeam::reset()
{
    // ++++
}

void ClassicalTeam::runControl(const double &t, const SimState &state, Eigen::Vector2d &vel1, Eigen::Vector2d &vel2)
{
    // ++++
    vel1(0) = -1.35*state.x_A1(PX); vel1(1) = -1.35*state.x_A1(PY);
    vel2(0) = -1.35*state.x_A2(PX); vel2(1) = -1.35*state.x_A2(PY);
}
