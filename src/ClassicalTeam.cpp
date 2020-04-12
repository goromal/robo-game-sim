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
    if (t < 5.0)
    {
        vel1(0) = 0.5; vel1(1) = 0.5;
        vel2(0) = 0.5; vel2(1) = -0.5;
    }
    else if(t < 10.0)
    {
        vel1 << -1.0, 0;
        vel2 << 0.0, -1.0;
    }
    else
    {
        vel1 << 1.2*cos(t), 1.2*sin(t);
        vel2 << 1.2*cos(t), 1.2*sin(t);
    }
}
