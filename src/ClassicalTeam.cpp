#include "ClassicalTeam.h"

ClassicalTeam::ClassicalTeam()
{
    team_ = TEAM_A;
    // ++++
}

ClassicalTeam::ClassicalTeam(int team)
{
    team_ = team;
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
  switch(team_)
  {
    case TEAM_A:
      vel1 << -(state.x_A1.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
      vel2 << -(state.x_A2.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
      break;
    case TEAM_B:
      vel1 << -(state.x_B1.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
      vel2 << -(state.x_B2.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
      break;
    default:
      vel1 << 0.0, 0.0;
      vel2 << 0.0, 0.0;
  }
}


void planTraj(const Eigen::Vector4d &x0, const Eigen::Vector4d &xT, const double &dt, const int &T,
              std::vector<Eigen::Vector2d> &vels)
{

}
