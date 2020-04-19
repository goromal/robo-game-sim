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
        const Eigen::Ref< const Eigen::MatrixXd > &C, const Eigen::Ref< const Eigen::MatrixXd > &D, double dt);
    ~ClassicalTeam();
    void reset();
    void runControl(const double &t, const SimState &state, Eigen::Vector2d &vel1, Eigen::Vector2d &vel2);
private:
    int team_;
    double dt_;
    drake::systems::LinearSystem<double> sys_;


    void planTraj(const Eigen::Vector4d &x0, const Eigen::Vector4d &xT, const double &T);



};
