#pragma once
#include <Eigen/Core>
#include <assert.h>
#include <logging-utils-lib/logging.h>
#include <math-utils-lib/modeling.h>
#include <math-utils-lib/tools.h>
#include "SimState.h"
#include "ClassicalTeam.h"

enum {NO_SCORE, TEAM_A_SCORE, TEAM_B_SCORE};

class GameSim
{
public:
    GameSim();
    ~GameSim();
    void reset(const bool &external, const double &dt, const int &winning_score,
               const Eigen::Vector4d &x0_ball, const bool &log, const std::string &logname);
    bool undecided();
    // Sim Iteration via External Reasoning
    Eigen::Matrix<double, 22, 1> setAwayTeamVelocities(const Eigen::Vector2d &vel_B1, const Eigen::Vector2d &vel_B2);
    // Sim Iteration via Internal Reasoning
    void run();

    double get_arena_X() const {return arena_X_;};
    double get_arena_Y() const {return arena_Y_;};
    double get_puck_rad() const {return p_rad_;};
    double get_player_rad() const {return P_rad_;};

private:
    void f_player(Ref<VectorXd> xdot, const Ref<const VectorXd> &x, const Ref<const VectorXd> &u);
    void f_puck(Ref<VectorXd> xdot, const Ref<const VectorXd> &x, const Ref<const VectorXd> &u);
    void updateSim(const Eigen::Vector2d &vel_A1, const Eigen::Vector2d &vel_A2,
                   const Eigen::Vector2d &vel_B1, const Eigen::Vector2d &vel_B2);
    unsigned int checkWallCollisions();
    void checkAgentCollisions();
    void collideElastically(const double &m1, const double &m2, const double &r1, const double &r2,
                            Ref<Vector4d> x1, Ref<Vector4d> x2);
    double mass(const unsigned int &id);
    double radius(const unsigned int &id);

    std::vector<unsigned int> agents_;
    std::vector<unsigned int> entities_;

    double arena_X_;
    double arena_Y_;
    double P_rad_;
    double p_rad_;
    double goal_height_;

    double tau_player_;
    double tau_puck_;
    modeling::RK4<Vector4d> puck_rk4_;
    modeling::RK4<Vector4d> player_rk4_;

    bool external_;
    bool log_;
    double dt_;
    double t_;
    int winning_score_;
    double player_mass_;
    double puck_mass_;

    SimState state_;
    ClassicalTeam* HomeTeam_;
    ClassicalTeam* AwayTeam_;
    logging::Logger logger_;

    // System matrices for drake solver
    Eigen::Matrix<double, 4, 4> A_;
    Eigen::Matrix<double, 4, 2> B_;
    Eigen::Matrix<double, 4, 4> C_;
    Eigen::Matrix<double, 4, 2> D_;

};
