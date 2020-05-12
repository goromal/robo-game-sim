#include "GameSim.h"

GameSim::GameSim() : reng_(), w_stdev_(0.0), w_dist_(0.0, 1.0)
{
    arena_X_ = 10.0;
    arena_Y_ = 5.0;
    P_rad_ = 0.2;
    p_rad_ = 0.175;
    goal_height_ = 1.0;

    entities_ = {A1GRID, A2GRID, B1GRID, B2GRID, PKGRID};

    tau_puck_ = 0.1; // set to 1.0 for bounce_kick to work (also in run_sim.py)
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
    dt_col_ = dt / COLLISION_GRID_POINTS;
    t_ = 0.0;
    w_stdev_ = noise;
    winning_score_ = winning_score;
    state_.TeamAScore = 0;
    state_.TeamBScore = 0;
    state_.x_ball = x0_ball;
    state_.x_A1 = Eigen::Vector4d(-arena_X_/4.0,  arena_Y_/4.0, 0.0, 0.0);
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

void GameSim::f_player(Ref<Vector4d> xdot, const Ref<const Vector4d> &x, const Vector2d &u)
{
    xdot(PX,0) = x(VX,0);
    xdot(PY,0) = x(VY,0);
    xdot(VX,0) = (u(0,0) - x(VX,0)) / tau_player_ + w_stdev_ * w_dist_(reng_);
    xdot(VY,0) = (u(1,0) - x(VY,0)) / tau_player_ + w_stdev_ * w_dist_(reng_);
}

Vector4d GameSim::RK4_player(const Ref<const Vector4d> &x, const Vector2d &u, const double &dt)
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

    return x + (k1 + k2*2.0 + k3*2.0 + k4) * (dt / 6.0);
}

void GameSim::f_puck(Ref<Vector4d> xdot, const Ref<const Vector4d> &x, const Vector2d &u)
{
    xdot(PX,0) = x(VX,0);
    xdot(PY,0) = x(VY,0);
    xdot(VX,0) = (u(0,0) - 0.2 * x(VX,0)) / tau_puck_;
    xdot(VY,0) = (u(1,0) - 0.2 * x(VY,0)) / tau_puck_;
}

Vector4d GameSim::RK4_puck(const Ref<const Vector4d> &x, const Vector2d &u, const double &dt)
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

    return x + (k1 + k2*2.0 + k3*2.0 + k4) * (dt / 6.0);
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
    int base_idx = 1; // there shouldn't be any collision overlaps at time = t_

    // Populate state grid by simulating at collision time grid points
    populateStateGrid(vel_A1, vel_A2, vel_B1, vel_B2);

    // Carry out event-based collision detection and handling
    std::vector<Collision> collisions;
    std::vector<int> checks;
    int counter = 0;
    std::map<double, int> collision_tracker;
    do
    {
        if (collisions.size() > 0)
            checks = carryOutFirstCollision(collisions, base_idx, collision_tracker);
        else
            checks = {A1GRID, A2GRID, B1GRID, B2GRID, PKGRID};

        insertCollisions(checks, collisions, base_idx);
        counter++;
    }
    while (collisions.size() > 0 && counter < COLLISION_COUNTER_LIM);

    // Set final sim states to values at the final grid point
    state_.arr.block<4,1>(SimState::A1, 0) = state_grid_.block<4,1>(A1GRID + GRID_S, COLLISION_GRID_POINTS);
    state_.arr.block<4,1>(SimState::A2, 0) = state_grid_.block<4,1>(A2GRID + GRID_S, COLLISION_GRID_POINTS);
    state_.arr.block<4,1>(SimState::B1, 0) = state_grid_.block<4,1>(B1GRID + GRID_S, COLLISION_GRID_POINTS);
    state_.arr.block<4,1>(SimState::B2, 0) = state_grid_.block<4,1>(B2GRID + GRID_S, COLLISION_GRID_POINTS);
    state_.arr.block<4,1>(SimState::PK, 0) = state_grid_.block<4,1>(PKGRID + GRID_S, COLLISION_GRID_POINTS);
    t_ += dt_;

    // Update logs
    if (log_)
    {
        // 29 double fields
        logger_.log(t_, static_cast<double>(state_.TeamAScore), static_cast<double>(state_.TeamBScore));
        logger_.logVectors(state_.x_ball.block<2,1>(PX,0), state_.x_A1.block<2,1>(PX,0),
                           state_.x_A2.block<2,1>(PX,0),   state_.x_B1.block<2,1>(PX,0),
                           state_.x_B2.block<2,1>(PX,0),   state_.damage.row(0),
                           state_.damage.row(1),           state_.damage.row(2),
                           state_.damage.row(3));
    }
}

void GameSim::populateStateGrid(const Eigen::Vector2d &A1v, const Eigen::Vector2d &A2v, const Eigen::Vector2d &B1v, const Eigen::Vector2d &B2v)
{
    state_grid_.col(0)           [A1GRID + GRID_T]    = t_;
    state_grid_.col(0).block<2,1>(A1GRID + GRID_U, 0) = A1v;
    state_grid_.col(0).block<4,1>(A1GRID + GRID_S, 0) = state_.arr.block<4,1>(SimState::A1,0);
    state_grid_.col(0)           [A2GRID + GRID_T]    = t_;
    state_grid_.col(0).block<2,1>(A2GRID + GRID_U, 0) = A2v;
    state_grid_.col(0).block<4,1>(A2GRID + GRID_S, 0) = state_.arr.block<4,1>(SimState::A2,0);
    state_grid_.col(0)           [B1GRID + GRID_T]    = t_;
    state_grid_.col(0).block<2,1>(B1GRID + GRID_U, 0) = B1v;
    state_grid_.col(0).block<4,1>(B1GRID + GRID_S, 0) = state_.arr.block<4,1>(SimState::B1,0);
    state_grid_.col(0)           [B2GRID + GRID_T]    = t_;
    state_grid_.col(0).block<2,1>(B2GRID + GRID_U, 0) = B2v;
    state_grid_.col(0).block<4,1>(B2GRID + GRID_S, 0) = state_.arr.block<4,1>(SimState::B2,0);
    state_grid_.col(0)           [PKGRID + GRID_T]    = t_;
    state_grid_.col(0).block<2,1>(PKGRID + GRID_U, 0) = Vector2d(0.0, 0.0);
    state_grid_.col(0).block<4,1>(PKGRID + GRID_S, 0) = state_.arr.block<4,1>(SimState::PK,0);

    Vector2d _1, _2;
    double _3, m_A1, r_A1, m_A2, r_A2, m_B1, r_B1, m_B2, r_B2, m_PK, r_PK;
    getEntityInfo(A1GRID, 0, _1, _2, m_A1, r_A1, _3);
    getEntityInfo(A2GRID, 0, _1, _2, m_A2, r_A2, _3);
    getEntityInfo(B1GRID, 0, _1, _2, m_B1, r_B1, _3);
    getEntityInfo(B2GRID, 0, _1, _2, m_B2, r_B2, _3);
    getEntityInfo(PKGRID, 0, _1, _2, m_PK, r_PK, _3);

    for (int i = 1; i <= COLLISION_GRID_POINTS; i++)
    {
        state_grid_.col(i)           [A1GRID + GRID_T]    = t_ + dt_col_ * i;
        state_grid_.col(i).block<2,1>(A1GRID + GRID_U, 0) = A1v;
        state_grid_.col(i).block<4,1>(A1GRID + GRID_S, 0) = gridSimAgnostic(A1GRID, i-1, dt_col_);
        correctOverlap(A1GRID, A2GRID, i, r_A1, r_A2);
        correctOverlap(A1GRID, B1GRID, i, r_A1, r_B1);
        correctOverlap(A1GRID, B2GRID, i, r_A1, r_B2);
        correctOverlap(A1GRID, PKGRID, i, r_A1, r_PK);
        correctOverlap(A1GRID, i, r_A1, WALL_UP);
        correctOverlap(A1GRID, i, r_A1, WALL_DOWN);
        correctOverlap(A1GRID, i, r_A1, WALL_RIGHT);
        correctOverlap(A1GRID, i, r_A1, WALL_LEFT);
        state_grid_.col(i)           [A2GRID + GRID_T]    = t_ + dt_col_ * i;
        state_grid_.col(i).block<2,1>(A2GRID + GRID_U, 0) = A2v;
        state_grid_.col(i).block<4,1>(A2GRID + GRID_S, 0) = gridSimAgnostic(A2GRID, i-1, dt_col_);
        correctOverlap(A2GRID, A1GRID, i, r_A2, r_A1);
        correctOverlap(A2GRID, B1GRID, i, r_A2, r_B1);
        correctOverlap(A2GRID, B2GRID, i, r_A2, r_B2);
        correctOverlap(A2GRID, PKGRID, i, r_A2, r_PK);
        correctOverlap(A2GRID, i, r_A2, WALL_UP);
        correctOverlap(A2GRID, i, r_A2, WALL_DOWN);
        correctOverlap(A2GRID, i, r_A2, WALL_RIGHT);
        correctOverlap(A2GRID, i, r_A2, WALL_LEFT);
        state_grid_.col(i)           [B1GRID + GRID_T]    = t_ + dt_col_ * i;
        state_grid_.col(i).block<2,1>(B1GRID + GRID_U, 0) = B1v;
        state_grid_.col(i).block<4,1>(B1GRID + GRID_S, 0) = gridSimAgnostic(B1GRID, i-1, dt_col_);
        correctOverlap(B1GRID, A1GRID, i, r_B1, r_A1);
        correctOverlap(B1GRID, A2GRID, i, r_B1, r_A2);
        correctOverlap(B1GRID, B2GRID, i, r_B1, r_B2);
        correctOverlap(B1GRID, PKGRID, i, r_B1, r_PK);
        correctOverlap(B1GRID, i, r_B1, WALL_UP);
        correctOverlap(B1GRID, i, r_B1, WALL_DOWN);
        correctOverlap(B1GRID, i, r_B1, WALL_RIGHT);
        correctOverlap(B1GRID, i, r_B1, WALL_LEFT);
        state_grid_.col(i)           [B2GRID + GRID_T]    = t_ + dt_col_ * i;
        state_grid_.col(i).block<2,1>(B2GRID + GRID_U, 0) = B2v;
        state_grid_.col(i).block<4,1>(B2GRID + GRID_S, 0) = gridSimAgnostic(B2GRID, i-1, dt_col_);
        correctOverlap(B2GRID, A1GRID, i, r_B2, r_A1);
        correctOverlap(B2GRID, A2GRID, i, r_B2, r_A2);
        correctOverlap(B2GRID, B1GRID, i, r_B2, r_B1);
        correctOverlap(B2GRID, PKGRID, i, r_B2, r_PK);
        correctOverlap(B2GRID, i, r_B2, WALL_UP);
        correctOverlap(B2GRID, i, r_B2, WALL_DOWN);
        correctOverlap(B2GRID, i, r_B2, WALL_RIGHT);
        correctOverlap(B2GRID, i, r_B2, WALL_LEFT);
        state_grid_.col(i)           [PKGRID + GRID_T]    = t_ + dt_col_ * i;
        state_grid_.col(i).block<2,1>(PKGRID + GRID_U, 0) = Vector2d(0.0, 0.0);
        state_grid_.col(i).block<4,1>(PKGRID + GRID_S, 0) = gridSimAgnostic(PKGRID, i-1, dt_col_);
        correctOverlap(PKGRID, A1GRID, i, r_PK, r_A1);
        correctOverlap(PKGRID, A2GRID, i, r_PK, r_A2);
        correctOverlap(PKGRID, B1GRID, i, r_PK, r_B1);
        correctOverlap(PKGRID, B2GRID, i, r_PK, r_B2);
        correctOverlap(PKGRID, i, r_PK, WALL_UP);
        correctOverlap(PKGRID, i, r_PK, WALL_DOWN);
        correctOverlap(PKGRID, i, r_PK, WALL_RIGHT);
        correctOverlap(PKGRID, i, r_PK, WALL_LEFT);
    }
}

std::vector<int> GameSim::carryOutFirstCollision(std::vector<Collision> &collisions, int &base_idx, std::map<double, int> &col_tracker)
{
    const static double damage_coeff = 0.01;

    // Initially empty check ID vector
    std::vector<int> checks;

    // Get most imminent collision
    Collision imminent_collision = collisions[collisions.size()-1]; collisions.pop_back();
    col_tracker[collisionToKey(imminent_collision)]++;

    // Update base index
    base_idx = static_cast<int>(floor((imminent_collision.t_ - t_) / dt_col_));

//    // If this same collision has been handled too many times within dt, then kill collision logic for the bodies involved
//    // for the rest of the time window (skip forward in time until the input changes OR some other collision causes a change)
//    if (col_tracker[collisionToKey(imminent_collision)] >= MAX_CONCURRENT_COLLS)
//    {
//        switch (imminent_collision.COLLISION_ID_)
//        {
//        case INTER_AGENT:
//        {
//            int id_i = imminent_collision.i_;
//            int id_j = imminent_collision.j_;

//            // In a situation like this, there very well may be overlap, so correct it (with a buffer to avoid future deadlocks)
//            double _1, _4, r_i, r_j;
//            Vector2d pos_i, pos_j, _3;
//            getEntityInfo(id_i, base_idx, pos_i, _3, _4, r_i, _1);
//            getEntityInfo(id_j, base_idx, pos_j, _3, _4, r_j, _1);
//            Vector2d pij = pos_j - pos_i;
//            double overlap = r_i + r_j - pij.norm();
//            if (overlap > 0)
//            {
//                state_grid_.col(base_idx).block<2,1>(id_i + GRID_P, 0) -= 1.01 * r_i/(r_i+r_j) * overlap * pij;
//                state_grid_.col(base_idx).block<2,1>(id_j + GRID_P, 0) += 1.01 * r_j/(r_i+r_j) * overlap * pij;
//            }

//            for (int i = base_idx+1; i <= COLLISION_GRID_POINTS; i++)
//            {
//                state_grid_.col(i).block<2,1>(id_i + GRID_P, 0) = state_grid_.col(base_idx).block<2,1>(id_i + GRID_P, 0);
//                state_grid_.col(i).block<2,1>(id_i + GRID_V, 0).setZero();
//                state_grid_.col(i).block<2,1>(id_j + GRID_P, 0) = state_grid_.col(base_idx).block<2,1>(id_j + GRID_P, 0);
//                state_grid_.col(i).block<2,1>(id_j + GRID_V, 0).setZero();
//            }
//            break;
//        }
//        default:
//        {
//            int id_i = imminent_collision.i_;
//            for (int i = base_idx+1; i <= COLLISION_GRID_POINTS; i++)
//            {
//                state_grid_.col(i).block<2,1>(id_i + GRID_P, 0) = state_grid_.col(base_idx).block<2,1>(id_i + GRID_P, 0);
//                state_grid_.col(i).block<2,1>(id_i + GRID_V, 0).setZero();
//            }
//            break;
//        }
//        }
//        return checks;
//    }

    // Carry out collision and update simulated grid points for collided objects from base index -> end
    switch(imminent_collision.COLLISION_ID_)
    {
    case INTER_AGENT:
    {
        int id_i = imminent_collision.i_;
        int id_j = imminent_collision.j_;

        // Extract relevant physical parameters
        double _1, m_i, m_j, r_i, r_j;
        Vector2d _2, _3;
        getEntityInfo(id_i, base_idx, _2, _3, m_i, r_i, _1);
        getEntityInfo(id_j, base_idx, _2, _3, m_j, r_j, _1);

        // Simulate up to collision point
        double dt_t = imminent_collision.t_ - state_grid_(id_i + GRID_T, base_idx);
        Vector4d x_i_t = gridSimAgnostic(id_i, base_idx, dt_t);
        Vector4d x_j_t = gridSimAgnostic(id_j, base_idx, dt_t);

        // Correct any overlap due to linear constant velocity approximation
        Vector2d pij = x_j_t.block<2,1>(PX,0) - x_i_t.block<2,1>(PX,0);
        double overlap = r_i + r_j - pij.norm();
        if (overlap > 0)
        {
            x_i_t.block<2,1>(PX,0) -= 1.01 * r_i/(r_i+r_j) * overlap * pij;
            x_j_t.block<2,1>(PX,0) += 1.01 * r_j/(r_i+r_j) * overlap * pij;
        }

        // Carry out collision
        Vector2d p1 = x_i_t.block<2,1>(PX,0);
        Vector2d v1 = x_i_t.block<2,1>(VX,0);
        Vector2d p2 = x_j_t.block<2,1>(PX,0);
        Vector2d v2 = x_j_t.block<2,1>(VX,0);
        // https://en.wikipedia.org/wiki/Elastic_collision
        x_i_t.block<2,1>(VX,0) = v1 - 2*m_j/(m_i+m_j) * (v1-v2).dot(p1-p2)/(p1-p2).dot(p1-p2)*(p1-p2);
        x_j_t.block<2,1>(VX,0) = v2 - 2*m_i/(m_i+m_j) * (v2-v1).dot(p2-p1)/(p2-p1).dot(p2-p1)*(p2-p1);

        // Add damage penalty if the puck isn't involved
        if (id_i != PKGRID && id_j != PKGRID)
        {
            double dv_sq = damage_coeff * (v1-v2).dot(v1-v2);
            state_.addDamage(GStoSSIdx(id_i), GStoSSIdx(id_j), m_i * dv_sq);
            state_.addDamage(GStoSSIdx(id_j), GStoSSIdx(id_i), m_j * dv_sq);
        }

        // Propagate states to next grid points
        Vector2d u_i_t = state_grid_.col(base_idx).block<2,1>(id_i + GRID_U, 0);
        Vector2d u_j_t = state_grid_.col(base_idx).block<2,1>(id_j + GRID_U, 0);
        state_grid_.col(base_idx + 1).block<4,1>(id_i + GRID_S, 0) = simAgnostic(id_i, x_i_t, u_i_t, dt_col_ - dt_t);
        state_grid_.col(base_idx + 1).block<4,1>(id_j + GRID_S, 0) = simAgnostic(id_j, x_j_t, u_j_t, dt_col_ - dt_t);
        correctOverlap(id_i, id_j, base_idx + 1, r_i, r_j);

        // Update rest of grid points
        for (int i = base_idx + 2; i <= COLLISION_GRID_POINTS; i++)
        {
            state_grid_.col(i).block<4,1>(id_i + GRID_S, 0) = gridSimAgnostic(id_i, i-1, dt_col_);
            state_grid_.col(i).block<4,1>(id_j + GRID_S, 0) = gridSimAgnostic(id_j, i-1, dt_col_);
            correctOverlap(id_i, id_j, i, r_i, r_j);
        }
        break;
    }
    default: // A wall collision
    {
        int id = imminent_collision.i_;
        Vector2d _2, _3;
        double _1, _4, r_i;
        getEntityInfo(id, base_idx, _2, _3, _1, r_i, _4);

        // Simulate up to collision point
        double dt_t = imminent_collision.t_ - state_grid_(id + GRID_T, base_idx);
        Vector4d x_i_t = gridSimAgnostic(id, base_idx, dt_t);

        // Carry out collision
        switch(imminent_collision.COLLISION_ID_)
        {
        case WALL_UP:    x_i_t(VY) *= -1.0; break;
        case WALL_DOWN:  x_i_t(VY) *= -1.0; break;
        case WALL_RIGHT: x_i_t(VX) *= -1.0; break;
        case WALL_LEFT:  x_i_t(VX) *= -1.0; break;
        }

        // If it's a puck and someone scored, then increment score count and reset puck state
        if (id == PKGRID && (imminent_collision.COLLISION_ID_ == WALL_LEFT || imminent_collision.COLLISION_ID_ == WALL_RIGHT) &&
            x_i_t(PY) < goal_height_ / 2.0 && x_i_t(PY) > -goal_height_ / 2.0)
        {
            imminent_collision.COLLISION_ID_ == WALL_LEFT ? state_.TeamAScore++ : state_.TeamBScore++;
            state_grid_.col(base_idx + 1).block<4,1>(id + GRID_S, 0).setZero();
        }
        else // Otherwise, propagate state to next grid point
        {
            Vector2d u_i_t = state_grid_.col(base_idx).block<2,1>(id + GRID_U, 0);
            state_grid_.col(base_idx + 1).block<4,1>(id + GRID_S, 0) = simAgnostic(id, x_i_t, u_i_t, dt_col_ - dt_t);
            correctOverlap(id, base_idx + 1, r_i, imminent_collision.COLLISION_ID_);
        }

        // Update rest of grid points
        for (int i = base_idx + 2; i <= COLLISION_GRID_POINTS; i++)
        {
            state_grid_.col(i).block<4,1>(id + GRID_S, 0) = gridSimAgnostic(id, i-1, dt_col_);
            correctOverlap(id, i, r_i, imminent_collision.COLLISION_ID_);
        }

        break;
    }
    }

    // Remove all obsoleted collisions
    auto new_end = std::remove_if(collisions.begin(), collisions.end(),
                                  [imminent_collision](const Collision &col) { return (imminent_collision.i_ == col.i_)                 ||
                                                                                      (imminent_collision.i_ == col.j_ && col.j_ != -1) ||
                                                                                      (imminent_collision.j_ == col.i_)                 ||
                                                                                      (imminent_collision.j_ == col.j_ && col.j_ != -1); });
    collisions.erase(new_end, collisions.end());

    // Return affected indices for checking for new collisions
    checks.push_back(imminent_collision.i_);
    if (imminent_collision.j_ != -1) checks.push_back(imminent_collision.j_);
    return checks;
}

void GameSim::insertCollisions(const std::vector<int> &checks, std::vector<Collision> &collisions, const int &base_idx)
{
    for (auto ID : checks)
    {
        // Check all remaining state grid points for the first collision overlap
        for (int i = base_idx; i < COLLISION_GRID_POINTS + 1; i++)
        {
             // std::cout << "GP_RUN\n";
            Collision imminent_collision;
            Vector2d pos_k, vel_k, pos_km1, vel_km1, o_pos_k, o_vel_k, o_pos_km1, o_vel_km1;
            double mass_k, radius_k, t_k, mass_km1, radius_km1, t_km1, o_mass_k, o_radius_k, o_t_k, o_mass_km1, o_radius_km1, o_t_km1;
            getEntityInfo(ID, i, pos_k, vel_k, mass_k, radius_k, t_k);

            // Check for WALL_UP collisions
            if (pos_k.y() + radius_k >= arena_Y_ / 2.0)
            {
                if (i == 0)
                {
                    double t_col = t_k;
                    if (t_col < imminent_collision.t_)
                        imminent_collision = Collision(t_col, ID, WALL_UP);
                }
                else
                {
                    // Backtrack and get exact collision time
                    getEntityInfo(ID, i-1, pos_km1, vel_km1, mass_km1, radius_km1, t_km1);
                    if (vel_km1.y() > 0)
                    {
                        double tau = fmax(0.0, (arena_Y_ / 2.0 - radius_km1 - pos_km1.y()) / vel_km1.y());
                        double t_col = t_km1 + tau;
                        if (t_col < imminent_collision.t_)
                            imminent_collision = Collision(t_col, ID, WALL_UP);
                    }
                }
            }

            // Check for WALL_DOWN collisions
            if (pos_k.y() - radius_k <= -arena_Y_ / 2.0)
            {
                if (i == 0)
                {
                    double t_col = t_k;
                    if (t_col < imminent_collision.t_)
                        imminent_collision = Collision(t_col, ID, WALL_DOWN);
                }
                else
                {
                    // Backtrack and get exact collision time
                    getEntityInfo(ID, i-1, pos_km1, vel_km1, mass_km1, radius_km1, t_km1);
                    if (vel_km1.y() < 0)
                    {
                        double tau = fmax(0.0, (-arena_Y_ / 2.0 + radius_km1 - pos_km1.y()) / vel_km1.y());
                        double t_col = t_km1 + tau;
                        if (t_col < imminent_collision.t_)
                            imminent_collision = Collision(t_col, ID, WALL_DOWN);
                    }
                }
            }

            // Check for WALL_RIGHT collisions
            if (pos_k.x() + radius_k >= arena_X_ / 2.0)
            {
                if (i == 0)
                {
                    double t_col = t_k;
                    if (t_col < imminent_collision.t_)
                        imminent_collision = Collision(t_col, ID, WALL_RIGHT);
                }
                else
                {
                    // Backtrack and get exact collision time
                    getEntityInfo(ID, i-1, pos_km1, vel_km1, mass_km1, radius_km1, t_km1);
                    if (vel_km1.x() > 0)
                    {
                        double tau = fmax(0.0, (arena_X_ / 2.0 - radius_km1 - pos_km1.x()) / vel_km1.x());
                        double t_col = t_km1 + tau;
                        if (t_col < imminent_collision.t_)
                            imminent_collision = Collision(t_col, ID, WALL_RIGHT);
                    }
                }
            }

            // Check for WALL_LEFT collisions
            if (pos_k.x() - radius_k <= -arena_X_ / 2.0)
            {
                if (i == 0)
                {
                    double t_col = t_k;
                    if (t_col < imminent_collision.t_)
                        imminent_collision = Collision(t_col, ID, WALL_LEFT);
                }
                else
                {
                    // Backtrack and get exact collision time
                    getEntityInfo(ID, i-1, pos_km1, vel_km1, mass_km1, radius_km1, t_km1);
                    if (vel_km1.x() < 0)
                    {
                        double tau = fmax(0.0, (-arena_X_ / 2.0 + radius_km1 - pos_km1.x()) / vel_km1.x());
                        double t_col = t_km1 + tau;
                        if (t_col < imminent_collision.t_)
                            imminent_collision = Collision(t_col, ID, WALL_LEFT);
                    }
                }
            }

            // Check for INTER_AGENT collisions
            for (auto entity : entities_)
            {
                if (entity != ID)
                {
                    getEntityInfo(entity, i, o_pos_k, o_vel_k, o_mass_k, o_radius_k, o_t_k);
                    if ((pos_k - o_pos_k).norm() <= radius_k + o_radius_k)
                    {
                        if (i == 0)
                        {
                            double t_col = t_k;
                            if (t_col < imminent_collision.t_)
                                imminent_collision = Collision(t_col, ID, entity, INTER_AGENT);
                        }
                        else
                        {
                            // Backtrack and get exact collision time
                            getEntityInfo(ID,     i-1,   pos_km1,   vel_km1,   mass_km1,   radius_km1,   t_km1);
                            getEntityInfo(entity, i-1, o_pos_km1, o_vel_km1, o_mass_km1, o_radius_km1, o_t_km1);
                            Vector2d Delta_p = o_pos_km1 - pos_km1;
                            Vector2d Delta_v = o_vel_km1 - vel_km1;
                            double sigma = radius_km1 + o_radius_km1;
                            double d = Delta_p.dot(Delta_v) * Delta_p.dot(Delta_v) - (Delta_v.dot(Delta_v)) * (Delta_p.dot(Delta_p) - sigma * sigma);
                            if (Delta_v.dot(Delta_p) < 0 && d >= 0)
                            {
                                double tau = fmax(0.0, -(Delta_v.dot(Delta_p) + sqrt(d)) / Delta_v.dot(Delta_v));
                                double t_col = t_km1 + tau;
                                if (t_col < imminent_collision.t_)
                                    imminent_collision = Collision(t_col, ID, entity, INTER_AGENT);
                            }
                        }
                    }
                }
            }

            // If there's an imminent collision, add it to collisions vector and end analysis for this ID
            if (imminent_collision.t_ - t_ < dt_)
            {
                collisions.push_back(imminent_collision);
                break;
            }
        }
    }

    // Sort collision list by insertion time
    std::sort(collisions.begin(), collisions.end(), CollisionComparator());
}

void GameSim::getEntityInfo(const int &id, const int &idx, Vector2d &pos, Vector2d &vel, double &mass, double &radius, double &t)
{
    pos = state_grid_.col(idx).block<2,1>(id + GRID_P, 0);
    vel = state_grid_.col(idx).block<2,1>(id + GRID_V, 0);
    t   = state_grid_(id + GRID_T, idx);

    switch(id)
    {
    case PKGRID:
        radius = p_rad_;
        mass = puck_mass_;
        break;
    case A1GRID:
        radius = P_rad_;
        mass = player_mass_;
        break;
    case A2GRID:
        radius = P_rad_;
        mass = player_mass_;
        break;
    case B1GRID:
        radius = P_rad_;
        mass = player_mass_;
        break;
    case B2GRID:
        radius = P_rad_;
        mass = player_mass_;
        break;
    }
}

Vector4d GameSim::gridSimAgnostic(const int &id, const int &idx, const double &dt)
{
    return simAgnostic(id, state_grid_.col(idx).block<4,1>(id + GRID_S, 0), state_grid_.col(idx).block<2,1>(id + GRID_U, 0), dt);
}

Vector4d GameSim::simAgnostic(const int &id, const Vector4d &x, const Vector2d &u, const double &dt)
{
    if (id == PKGRID)
        return RK4_puck(x, u, dt);
    else
        return RK4_player(x, u, dt);
}

int GameSim::GStoSSIdx(const int &GS_idx)
{
    int SS_idx = -1;
    switch(GS_idx)
    {
    case A1GRID:
        SS_idx = SimState::A1;
        break;
    case A2GRID:
        SS_idx = SimState::A2;
        break;
    case B1GRID:
        SS_idx = SimState::B1;
        break;
    case B2GRID:
        SS_idx = SimState::B2;
        break;
    case PKGRID:
        SS_idx = SimState::PK;
        break;
    }
    return SS_idx;
}

bool GameSim::correctOverlap(const int &i, const int &j, const int &idx, const double &r_i, const double &r_j)
{
    Vector2d pij = state_grid_.col(idx).block<2,1>(j + GRID_P, 0) - state_grid_.col(idx).block<2,1>(i + GRID_P, 0);
    double overlap = r_i + r_j - pij.norm();
    if (overlap > 0)
    {
        state_grid_.col(idx).block<2,1>(i + GRID_P, 0) -= OVERLAP_BUFFER * r_i/(r_i+r_j) * overlap * pij;
        state_grid_.col(idx).block<2,1>(j + GRID_P, 0) += OVERLAP_BUFFER * r_j/(r_i+r_j) * overlap * pij;
        return true;
    }
    return false;
}

bool GameSim::correctOverlap(const int &i, const int &idx, const double &r_i, const int &WALL_TYPE)
{
    double overlap = 0.0;
    Vector2d correction;
    switch(WALL_TYPE)
    {
    case WALL_UP:
        overlap = state_grid_(i + GRID_P + PY, idx) + r_i - arena_Y_ / 2.0;
        correction << 0., -1.;
        break;
    case WALL_DOWN:
        overlap = -state_grid_(i + GRID_P + PY, idx) + r_i - arena_Y_ / 2.0;
        correction << 0., 1.;
        break;
    case WALL_RIGHT:
        overlap = state_grid_(i + GRID_P + PX, idx) + r_i - arena_X_ / 2.0;
        correction << -1., 0.;
        break;
    case WALL_LEFT:
        overlap = -state_grid_(i + GRID_P + PX, idx) + r_i - arena_X_ / 2.0;
        correction << 1., 0.;
        break;
    }
    if (overlap > 0)
    {
        state_grid_.col(idx).block<2,1>(i + GRID_P, 0) += OVERLAP_BUFFER * overlap * correction;
        return true;
    }
    return false;
}
