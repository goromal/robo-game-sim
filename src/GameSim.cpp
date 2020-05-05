#include "GameSim.h"

GameSim::GameSim() : reng_(), w_stdev_(0.0), w_dist_(0.0, 1.0)
{
    arena_X_ = 10.0;
    arena_Y_ = 5.0;
    P_rad_ = 0.2;
    p_rad_ = 0.175;
    goal_height_ = 1.0;

    agents_   = {SimState::A1, SimState::A2, SimState::B1, SimState::B2};
    entities_ = {SimState::PK, SimState::A1, SimState::A2, SimState::B1, SimState::B2};

    tau_puck_ = 0.1;
    tau_player_ = 0.5;
    player_mass_ = 1.0;
    puck_mass_ = 0.5;
}

GameSim::~GameSim() {}

Eigen::Matrix<double, SimState::SIZE, 1> GameSim::reset(const double &dt=0.05,
                    const int &winning_score=3, const Eigen::Vector4d &x0_ball=Eigen::Vector4d::Zero(),
                    const double &noise=0.0, const bool &log=false, const std::string &logname="~/gamelog.log")
{
    dt_ = dt;
    t_ = 0.0;
    w_stdev_ = noise;
    winning_score_ = winning_score;
    state_.TeamAScore = 0;
    state_.TeamBScore = 0;
    state_.x_ball = x0_ball;
    state_.x_A1 = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);//Eigen::Vector4d(-arena_X_/4.0,  arena_Y_/4.0, 0.0, 0.0);
    state_.x_A2 = Eigen::Vector4d(-arena_X_/4.0, -arena_Y_/4.0, 0.0, 0.0);
    state_.x_B1 = Eigen::Vector4d( arena_X_/4.0,  arena_Y_/4.0, 0.0, 0.0);
    state_.x_B2 = Eigen::Vector4d( arena_X_/4.0, -arena_Y_/4.0, 0.0, 0.0);
    state_.damage.setZero();

    log_ = log;
    if (log_)
        logger_.open(logname);
    else
        if (logger_.file_.is_open())
            logger_.file_.close();
    return state_.vector();
}

bool GameSim::undecided()
{
    return (state_.TeamAScore < winning_score_) && (state_.TeamBScore < winning_score_);
}

void GameSim::f_player(Ref<Vector4d> xdot, const Ref<Vector4d> &x, const Vector2d &u)
{
    xdot(PX,0) = x(VX,0);
    xdot(PY,0) = x(VY,0);
    xdot(VX,0) = (u(0,0) - x(VX,0)) / tau_player_ + w_stdev_ * w_dist_(reng_);
    xdot(VY,0) = (u(1,0) - x(VY,0)) / tau_player_ + w_stdev_ * w_dist_(reng_);
}

void GameSim::RK4_player(Ref<Vector4d> x, const Vector2d &u, const double &dt)
{
    static Vector4d k1, k2, k3, k4, x2, x3, x4;
    f_player(k1, x, u);

    x2 = x;
    x2 += k1 * (dt/2.0);
    f_player(k2, x2, u);

    x3 = x;
    x3 += k2 * (dt/2.0);
    f_player(k3, x3, u);

    x4 = x;
    x4 += k3 * dt;
    f_player(k4, x4, u);

    x += (k1 + k2*2.0 + k3*2.0 + k4) * (dt / 6.0);
}

void GameSim::f_puck(Ref<Vector4d> xdot, const Ref<Vector4d> &x, const Vector2d &u)
{
    xdot(PX,0) = x(VX,0);
    xdot(PY,0) = x(VY,0);
    xdot(VX,0) = (u(0,0) - 0.2 * x(VX,0)) / tau_puck_;
    xdot(VY,0) = (u(1,0) - 0.2 * x(VY,0)) / tau_puck_;
}

void GameSim::RK4_puck(Ref<Vector4d> x, const Vector2d &u, const double &dt)
{
    static Vector4d k1, k2, k3, k4, x2, x3, x4;
    f_puck(k1, x, u);

    x2 = x;
    x2 += k1 * (dt/2.0);
    f_puck(k2, x2, u);

    x3 = x;
    x3 += k2 * (dt/2.0);
    f_puck(k3, x3, u);

    x4 = x;
    x4 += k3 * dt;
    f_puck(k4, x4, u);

    x += (k1 + k2*2.0 + k3*2.0 + k4) * (dt / 6.0);
}

Eigen::Matrix<double, SimState::SIZE, 1> GameSim::run(const Eigen::Vector2d &vel_A1, const Eigen::Vector2d &vel_A2,
                                                      const Eigen::Vector2d &vel_B1, const Eigen::Vector2d &vel_B2)
{
    if (undecided())
        updateSim(vel_A1, vel_A2, vel_B1, vel_B2);
    return state_.vector();
}

void GameSim::updateSim(const Eigen::Vector2d &vel_A1, const Eigen::Vector2d &vel_A2,
                        const Eigen::Vector2d &vel_B1, const Eigen::Vector2d &vel_B2)
{
    // Update entity dynamics
    RK4_player(state_.arr.block<4,1>(SimState::A1,0), vel_A1, dt_);
    RK4_player(state_.arr.block<4,1>(SimState::A2,0), vel_A2, dt_);
    RK4_player(state_.arr.block<4,1>(SimState::B1,0), vel_B1, dt_);
    RK4_player(state_.arr.block<4,1>(SimState::B2,0), vel_B2, dt_);
    RK4_puck(state_.arr.block<4,1>(SimState::PK,0), Vector2d(0., 0.), dt_);

    // Handle inter-agent-inter-puck collisions
    checkAgentCollisions();

    // Handle collisions with walls for each player and puck + scoring check
    unsigned int score_status = checkWallCollisions();

    // Handle scoring check
    switch(score_status)
    {
    case TEAM_A_SCORE:
        state_.TeamAScore ++;
        state_.x_ball.setZero();
        break;
    case TEAM_B_SCORE:
        state_.TeamBScore++;
        state_.x_ball.setZero();
        break;
    default:
        break;
    }

    // Update time
    t_ += dt_;

    // Update logs
    if (log_)
    {
        // 13 double fields
        logger_.log(t_, static_cast<double>(state_.TeamAScore), static_cast<double>(state_.TeamBScore));
        logger_.logVectors(state_.x_ball.block<2,1>(PX,0), state_.x_A1.block<2,1>(PX,0),
                           state_.x_A2.block<2,1>(PX,0),   state_.x_B1.block<2,1>(PX,0),
                           state_.x_B2.block<2,1>(PX,0),   state_.damage.row(0),
                           state_.damage.row(1),           state_.damage.row(2),
                           state_.damage.row(3));
    }
}

unsigned int GameSim::checkWallCollisions()
{
    unsigned int score_status = NO_SCORE;

    // Top and bottom wall collision check
    for (auto entity : entities_)
    {
        if (state_.arr(entity + PY,0) + P_rad_ > arena_Y_ / 2.0)
        {
            state_.arr(entity + PY,0) = arena_Y_ / 2.0 - P_rad_;
            state_.arr(entity + VY,0) *= -1.0;
        }
        else if (state_.arr(entity + PY) - P_rad_ < -arena_Y_ / 2.0)
        {
            state_.arr(entity + PY,0) = -arena_Y_ / 2.0 + P_rad_;
            state_.arr(entity + VY,0) *= -1.0;
        }
    }

    // Left and right wall collision checks for AGENTS
    for (auto agent : agents_)
    {
        if (state_.arr(agent + PX,0) - P_rad_ < -arena_X_ / 2.0)
        {
            state_.arr(agent + PX,0) = -arena_X_ / 2.0 + P_rad_;
            state_.arr(agent + VX,0) *= -1.0;
        }
        else if (state_.arr(agent + PX,0) + P_rad_ > arena_X_ / 2.0)
        {
            state_.arr(agent + PX,0) = arena_X_ / 2.0 - P_rad_;
            state_.arr(agent + VX,0) *= -1.0;
        }
    }

    // Left and right wall collision checks for PUCK
    if (state_.x_ball(PX) - P_rad_ < -arena_X_/2.0)
    {
        if (state_.x_ball(PY) < goal_height_/2.0 && state_.x_ball(PY) > -goal_height_/2.0)
        {
            score_status = TEAM_B_SCORE;
        }
        else
        {
            state_.x_ball(PX) = -arena_X_/2.0 + P_rad_;
            state_.x_ball(VX) *= -1.0;
        }
    }
    else if (state_.x_ball(PX) + P_rad_ > arena_X_/2.0)
    {
        if (state_.x_ball(PY) < goal_height_/2.0 && state_.x_ball(PY) > -goal_height_/2.0)
        {
            score_status = TEAM_A_SCORE;
        }
        else
        {
            state_.x_ball(PX) = arena_X_/2.0 - P_rad_;
            state_.x_ball(VX) *= -1.0;
        }
    }

    return score_status;
}

void GameSim::checkAgentCollisions()
{
    // Check for collisions in every pairwise combination of entities
    do
    {
        unsigned int ID1 = entities_[0];
        unsigned int ID2 = entities_[1];
        if ((state_.arr.block<4,1>(ID1,0) - state_.arr.block<4,1>(ID2,0)).block<2,1>(0,0).norm()
            <= radius(ID1) + radius(ID2))
        {
            double penalty1, penalty2;
            collideElastically(mass(ID1), mass(ID2), radius(ID1), radius(ID2),
                               state_.arr.block<4,1>(ID1,0), state_.arr.block<4,1>(ID2,0),
                               (ID1 != SimState::PK && ID2 != SimState::PK), ID1, ID2);
        }
    }
    while(next_combination(entities_.begin(), entities_.begin() + 2, entities_.end()));
}

void GameSim::collideElastically(const double &m1, const double &m2, const double &r1, const double &r2,
                                 Ref<Vector4d> x1, Ref<Vector4d> x2, const bool &penalize,
                                 const int &ID1, const int &ID2)
{
    const static double damage_coeff = 0.01;

    Vector2d p1 = x1.block<2,1>(PX,0);
    Vector2d v1 = x1.block<2,1>(VX,0);
    Vector2d p2 = x2.block<2,1>(PX,0);
    Vector2d v2 = x2.block<2,1>(VX,0);

    // Place at sufficient distance apart to avoid overlapping long-term
    Vector2d p12 = p2 - p1; // vector from p1 -> p2
    double np12 = p12.norm() - (r1 + r2); // negative if there's overlap
    if (np12 < 0.0)
    {
        x1.block<2,1>(PX,0) += np12 * 2.0 * p12; /// 2.0 * p12;
        x2.block<2,1>(PX,0) -= np12 * 2.0 * p12; /// 2.0 * p12;
    }

    // https://en.wikipedia.org/wiki/Elastic_collision
    x1.block<2,1>(VX,0) = v1 - 2*m2/(m1+m2) * (v1-v2).dot(p1-p2)/(p1-p2).dot(p1-p2)*(p1-p2);
    x2.block<2,1>(VX,0) = v2 - 2*m1/(m1+m2) * (v2-v1).dot(p2-p1)/(p2-p1).dot(p2-p1)*(p2-p1);

    if (penalize)
    {
        double dv_sq = damage_coeff * (v1-v2).dot(v1-v2);
        state_.addDamage(ID1, ID2, m1 * dv_sq);
        state_.addDamage(ID2, ID1, m2 * dv_sq);
    }
}

double GameSim::mass(const unsigned int &id)
{
    double mass = 0.0;
    switch(id)
    {
    case SimState::PK:
        mass = puck_mass_;
        break;
    case SimState::A1:
        mass = player_mass_;
        break;
    case SimState::A2:
        mass = player_mass_;
        break;
    case SimState::B1:
        mass = player_mass_;
        break;
    case SimState::B2:
        mass = player_mass_;
        break;
    default:
        mass = 0.0;
        break;
    }
    return mass;
}

double GameSim::radius(const unsigned int &id)
{
    double radius = 0.0;
    switch(id)
    {
    case SimState::PK:
        radius = p_rad_;
        break;
    case SimState::A1:
        radius = P_rad_;
        break;
    case SimState::A2:
        radius = P_rad_;
        break;
    case SimState::B1:
        radius = P_rad_;
        break;
    case SimState::B2:
        radius = P_rad_;
        break;
    default:
        radius = 0.0;
        break;
    }
    return radius;
}
