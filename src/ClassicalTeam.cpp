#include "ClassicalTeam.h"


//ClassicalTeam::ClassicalTeam()
//{
//    team_ = TEAM_A;
//    // ++++
//}
//
//ClassicalTeam::ClassicalTeam(int team)
//: sys_(Eigen::MatrixXd(),Eigen::MatrixXd(), Eigen::MatrixXd(), Eigen::MatrixXd(), 0.0)
//{
//    team_ = team;
//    // ++++
//}

ClassicalTeam::ClassicalTeam(int team, const Eigen::Ref<const Eigen::MatrixXd> &A,
                             const Eigen::Ref<const Eigen::MatrixXd> &B,
                             const Eigen::Ref<const Eigen::MatrixXd> &C,
                             const Eigen::Ref<const Eigen::MatrixXd> &D,
                             const double &dt,
                             double arena_X,
                             double arena_Y,
                             double P_rad,
                             double p_rad,
                             double goal_height,
                             double goal_X)
                             : sys_(A, B, C, D, dt), arena_X_(arena_X), arena_Y_(arena_Y),
                               P_rad_(P_rad), p_rad_(p_rad), goal_height_(goal_height), goal_X_(goal_X)
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
  switch(team_)
  {
    case TEAM_A:

      if (t <= 4) {

        // Plan traj for player 1 first
        if (traj_set_) {
          get_control(t, vel1);
        } else {
          // get vector from ball to goal
          Eigen::Vector2d shoot_direction;
          shoot_direction << Eigen::Vector2d{goal_X_, 0.0}-state.x_ball.block<2,1>(PX,0);
          shoot_direction.normalize();

          Eigen::Vector4d xT;
          xT << (state.x_ball.block<2,1>(PX,0) - shoot_direction * (p_rad_+P_rad_)),
              8 * shoot_direction;

          traj_set_ = planTraj(state.x_A1, xT, 4.0);
        }

        vel2 << 0.0, 0.0;

      } else {
        vel1 << 0.0, 0.0;
        vel2 << 0.0, 0.0;
      }

      break;

    case TEAM_B:
//      vel1 << -(state.x_B1.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
//      vel2 << -(state.x_B2.block<2,1>(PX,0) - state.x_ball.block<2,1>(PX,0));
      vel1 << 0.0, 0.0;
      vel2 << 0.0, 0.0;
      break;

    default:
      vel1 << 0.0, 0.0;
      vel2 << 0.0, 0.0;
  }
}


bool ClassicalTeam::planTraj(const Eigen::Vector4d &x0, const Eigen::Vector4d &xT, const double &T)
{
  int num_time_samples = int(T/dt_);
  // Mathematical program
  drake::systems::trajectory_optimization::DirectTranscription prog(&sys_, *sys_.CreateDefaultContext(), num_time_samples);

  // Add constraints
  prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state());
  prog.AddBoundingBoxConstraint(xT, xT, prog.final_state());

  // Control limits: inifinity norm less than 2
  double input_limit = 10;
  prog.AddConstraintToAllKnotPoints(prog.input()[0] <= input_limit);
  prog.AddConstraintToAllKnotPoints(prog.input()[1] <= input_limit);
  prog.AddConstraintToAllKnotPoints(prog.input()[0] >= -input_limit);
  prog.AddConstraintToAllKnotPoints(prog.input()[1] >= -input_limit);

  // Robot remains in the field
  prog.AddConstraintToAllKnotPoints(prog.state()[0]+P_rad_ <= arena_X_/2.0);
  prog.AddConstraintToAllKnotPoints(prog.state()[0]-P_rad_ >= -arena_X_/2.0);
  prog.AddConstraintToAllKnotPoints(prog.state()[1]+P_rad_ <= arena_Y_/2.0);
  prog.AddConstraintToAllKnotPoints(prog.state()[1]-P_rad_ >= -arena_Y_/2.0);

  // Add control penalty: |u|^2
   prog.AddRunningCost(prog.input()[0]*prog.input()[0] + prog.input()[1]*prog.input()[1]);

  // Solve
  auto result = drake::solvers::Solve(prog);
  if (result.is_success()) {
    std::cout << "Optimization successful." << std::endl;

    u_traj_.clear();
    x_traj_.clear();
    auto x_sol = prog.ReconstructStateTrajectory(result);
    auto u_sol = prog.ReconstructInputTrajectory(result);
    auto x_traj_matrix = x_sol.vector_values(x_sol.get_segment_times());
    auto u_traj_matrix = u_sol.vector_values(u_sol.get_segment_times());

    u_traj_.reserve(num_time_samples);
    x_traj_.reserve(num_time_samples);

    // Save trajectories
    for (int i = 0; i < num_time_samples; ++i) {
      u_traj_.emplace_back(u_traj_matrix.col(i));
      x_traj_.emplace_back(x_traj_matrix.col(i));
    }

  } else {
    std::cout << "Optimization failed." << std::endl;
  }

  return result.is_success();

}

void ClassicalTeam::get_control(const double &t, Eigen::Vector2d &vel)
{
  int time_step = int(t/dt_);
  vel << u_traj_.at(time_step);
}