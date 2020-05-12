#pragma once
#include <Eigen/Core>
#include "Collision.h"
#include <random>

#define COLLISION_GRID_POINTS 50
#define COLLISION_COUNTER_LIM 50
#define MAX_CONCURRENT_COLLS  10
#define OVERLAP_BUFFER        1.1

enum {NO_SCORE, TEAM_A_SCORE, TEAM_B_SCORE};

enum {A1GRID = 0, A2GRID = 7, B1GRID = 14, B2GRID = 21, PKGRID = 28};
enum {GRID_T = 0, GRID_U = 1, GRID_S = 3,  GRID_P = 3,  GRID_V = 5};

class GameSim
{
public:
    GameSim();
    ~GameSim();
    Eigen::Matrix<double, SimState::SIZE, 1> reset(const double &dt, const int &winning_score,
               const Eigen::Vector4d &x0_ball, const double& noise, const bool &log, const std::string &logname);
    bool undecided();
    Eigen::Matrix<double, SimState::SIZE, 1> run(const Eigen::Vector2d &vel_A1, const Eigen::Vector2d &vel_A2,
                                                 const Eigen::Vector2d &vel_B1, const Eigen::Vector2d &vel_B2);

private:
    void f_player(Ref<Vector4d> xdot, const Ref<const Vector4d> &x, const Vector2d &u);
    void f_puck(Ref<Vector4d> xdot, const Ref<const Vector4d> &x, const Vector2d &u);
    Vector4d RK4_player(const Ref<const Vector4d> &x, const Vector2d &u, const double &dt);
    Vector4d RK4_puck(const Ref<const Vector4d> &x, const Vector2d &u, const double &dt);
    void updateSim(const Eigen::Vector2d &vel_A1, const Eigen::Vector2d &vel_A2,
                   const Eigen::Vector2d &vel_B1, const Eigen::Vector2d &vel_B2);
    void populateStateGrid(const Eigen::Vector2d &A1v, const Eigen::Vector2d &A2v, const Eigen::Vector2d &B1v, const Eigen::Vector2d &B2v);
    std::vector<int> carryOutFirstCollision(std::vector<Collision> &collisions, int &base_idx, std::map<double, int> &col_tracker);
    void insertCollisions(const std::vector<int> &checks, std::vector<Collision> &collisions, const int &base_idx);
    void getEntityInfo(const int &id, const int &idx, Vector2d &pos, Vector2d &vel, double &mass, double &radius, double &t);
    Vector4d gridSimAgnostic(const int &id, const int &idx, const double &dt);
    Vector4d simAgnostic(const int &id, const Vector4d &x, const Vector2d &u, const double &dt);
    int GStoSSIdx(const int &GS_idx);
    bool correctOverlap(const int &i, const int &j, const int &idx, const double &r_i, const double &r_j);
    bool correctOverlap(const int &i, const int &idx, const double &r_i, const int &WALL_TYPE);

    Matrix<double, 35, COLLISION_GRID_POINTS + 1> state_grid_;
    std::vector<Collision> collisions_;
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
    double dt_col_;
    double t_;
    int winning_score_;
    double player_mass_;
    double puck_mass_;

    SimState state_;
    Logger logger_;

    std::default_random_engine reng_;
    double w_stdev_;
    std::normal_distribution<double> w_dist_;
};
