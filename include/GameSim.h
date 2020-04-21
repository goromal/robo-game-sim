#pragma once
#include <Eigen/Core>
#include "SimState.h"

enum {NO_SCORE, TEAM_A_SCORE, TEAM_B_SCORE};

class GameSim
{
public:
    GameSim();
    ~GameSim();
    Eigen::Matrix<double, SimState::SIZE, 1> reset(const double &dt, const int &winning_score,
               const Eigen::Vector4d &x0_ball, const bool &log, const std::string &logname);
    bool undecided();
    Eigen::Matrix<double, SimState::SIZE, 1> run(const Eigen::Vector2d &vel_A1, const Eigen::Vector2d &vel_A2,
                                                 const Eigen::Vector2d &vel_B1, const Eigen::Vector2d &vel_B2);

private:
    void f_player(Ref<Vector4d> xdot, const Ref<Vector4d> &x, const Vector2d &u);
    void f_puck(Ref<Vector4d> xdot, const Ref<Vector4d> &x, const Vector2d &u);
    void RK4_player(Ref<Vector4d> x, const Vector2d &u, const double &dt);
    void RK4_puck(Ref<Vector4d> x, const Vector2d &u, const double &dt);
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

    bool log_;
    double dt_;
    double t_;
    int winning_score_;
    double player_mass_;
    double puck_mass_;

    SimState state_;
    Logger logger_;

};
