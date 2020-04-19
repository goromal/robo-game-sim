#include "ClassicalTeam.h"


//ClassicalTeam::ClassicalTeam()
//{
//    team_ = TEAM_A;
//    // ++++
//}
//
//ClassicalTeam::ClassicalTeam(int team)
//{
//    team_ = team;
//    // ++++
//}

ClassicalTeam::ClassicalTeam(int team, const Eigen::Ref<const Eigen::MatrixXd> &A,
                             const Eigen::Ref<const Eigen::MatrixXd> &B,
                             const Eigen::Ref<const Eigen::MatrixXd> &C,
                             const Eigen::Ref<const Eigen::MatrixXd> &D,
                             double dt): sys_{A, B, C, D, dt}
{
  team_ = team;
  dt_ = dt;
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
//  switch(team_)
//  {
//    case TEAM_A:
//      vel1 << -(state.x_A1.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
//      vel2 << -(state.x_A2.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
//      break;
//    case TEAM_B:
//      vel1 << -(state.x_B1.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
//      vel2 << -(state.x_B2.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
//      break;
//    default:
//      vel1 << 0.0, 0.0;
//      vel2 << 0.0, 0.0;
//  }

  vel1 << 0.0, 0.0;
  vel2 << 0.0, 0.0;
  // When running for the first time, plan a trajectory and save
  // Subsequently, run the saved trajectory
  planTraj(state.x_A1, state.x_ball, 3.0);
}


void ClassicalTeam::planTraj(const Eigen::Vector4d &x0, const Eigen::Vector4d &xT, const double &T)
{
  int num_time_samples = int(T/dt_);
  // Mathematical program
  drake::systems::trajectory_optimization::DirectTranscription prog(&sys_, *sys_.CreateDefaultContext(), num_time_samples);

  // Add constraints
  prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state());
  prog.AddBoundingBoxConstraint(xT, xT, prog.final_state());

  // Control limits
  prog.AddConstraintToAllKnotPoints(prog.input()[0] <= 2);
  prog.AddConstraintToAllKnotPoints(prog.input()[0] >= -2);

  // TODO: Robot remains in the field
  // TODO: Add control penalty
  // prog.AddRunningCost(prog.input()[0]*prog.input()[0]);

  // Solve
  auto result = drake::solvers::Solve(prog);
  if (result.is_success()) {
    std::cout << "Optimization successful." << std::endl;
  } else {
    std::cout << "Optimization failed." << std::endl;
  }

  auto x_sol = prog.ReconstructStateTrajectory(result);
  auto u_sol = prog.ReconstructInputTrajectory(result);
  std::cout << "x_sol " << x_sol.vector_values(x_sol.get_segment_times()) << std::endl;
  std::cout << "u_sol " << u_sol.vector_values(u_sol.get_segment_times()) << std::endl;

}