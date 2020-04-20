#pragma once
#include <Eigen/Core>
#include <vector>
#include "SimState.h"
#include <drake/solvers/mathematical_program.h>
#include <drake/common/symbolic_variable.h>
#include <drake/solvers/solve.h>
#include <drake/solvers/constraint.h>

#include <drake/systems/primitives/linear_system.h>
#include <drake/systems/trajectory_optimization/direct_transcription.h>

class ClassicalTeam
{
public:
//    ClassicalTeam();
//    ClassicalTeam(int team);
    ClassicalTeam(int team, const Eigen::Ref< const Eigen::MatrixXd > &A, const Eigen::Ref< const Eigen::MatrixXd > &B,
        const Eigen::Ref< const Eigen::MatrixXd > &C, const Eigen::Ref< const Eigen::MatrixXd > &D, const double &dt,
        double arena_X, double arena_Y, double P_rad, double p_rad, double goal_height, double goal_X);
    ~ClassicalTeam();
    void reset();
    void runControl(const double &t, const SimState &state, Eigen::Vector2d &vel1, Eigen::Vector2d &vel2);
private:
    int team_;
    double dt_;

    // Some game information
    double arena_X_;
    double arena_Y_;
    double P_rad_;
    double p_rad_;
    double goal_height_;
    double goal_X_;

    // Drake sys_
    drake::systems::LinearSystem<double> sys_;

    // Trajectory variables
    bool traj_set_{false};
    std::vector<Eigen::Vector2d> u_traj_;
    std::vector<Eigen::Vector4d> x_traj_;

    bool planTraj(const Eigen::Vector4d &x0, const Eigen::Vector4d &xT, const double &T);
    void get_control(const double &t, Eigen::Vector2d &vel);



};
