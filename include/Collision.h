#pragma once

#include "SimState.h"

enum {WALL_UP = 0, WALL_DOWN = 1, WALL_RIGHT = 2, WALL_LEFT = 3, INTER_AGENT = 4};

struct Collision
{
    Collision() : t_(1.0e9), i_(-1), j_(-1), COLLISION_ID_(-1) {}
    Collision(const double &t, const int &i, const int &COLLISION_ID) : t_(t), i_(i), j_(-1), COLLISION_ID_(COLLISION_ID) {}
    Collision(const double &t, const int &i, const int &j, const int &COLLISION_ID) : t_(t), i_(i), j_(j), COLLISION_ID_(COLLISION_ID) {}

    double t_;
    int i_;
    int j_;
    int COLLISION_ID_;
};

inline double collisionToKey(const Collision &collision)
{
    return floor(collision.t_ + 100.0 * static_cast<double>(collision.i_) + 10000.0 * static_cast<double>(collision.j_) + 1000000.0 * static_cast<double>(collision.COLLISION_ID_));
}

struct CollisionComparator
{
    inline bool operator() (const Collision &coll1, const Collision &coll2)
    {
        return (coll1.t_ > coll2.t_); // SMALLEST TIME SHOULD BE AT _END_ OF SORTED VECTOR
    }
};

struct CollisionMapComparator
{
    inline bool operator() (const Collision &coll1, const Collision &coll2)
    {
        return (coll1.t_ + coll1.i_ + coll1.j_ < coll2.t_ + coll2.i_ + coll2.j_);
    }
};
