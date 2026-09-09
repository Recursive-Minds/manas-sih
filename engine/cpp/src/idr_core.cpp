#include "idr_core.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#if __has_include(<Eigen/Dense>)
  #include <Eigen/Dense>
  #define HAS_EIGEN 1
#else
  #define HAS_EIGEN 0
#endif

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
constexpr double MAX_BG = 0.008726; // +/- 0.5 deg/s max gyro bias

// State representation (15 states)
struct IDRState {
    double p[3];       // East, North, Up (m)
    double v[3];       // vE, vN, vU (m/s)
    double heading;    // Azimuth (rad, 0 = North, pi/2 = East)
    double ba[3];      // Accel bias (m/s^2)
    double bg[3];      // Gyro bias (rad/s)
    double P[15][15];  // Error-state covariance
};

static IDRState g_state;
static bool g_initialized = false;

inline double wrap_heading(double h) {
    h = std::fmod(h, TWO_PI);
    if (h < 0.0) h += TWO_PI;
    return h;
}

void init_covariance(double P[15][15]) {
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            P[i][j] = 0.0;
        }
    }
    // Position std: 1.5m -> 2.25
    P[0][0] = 2.25; P[1][1] = 2.25; P[2][2] = 9.0;
    // Velocity std: 0.5m/s -> 0.25
    P[3][3] = 0.25; P[4][4] = 0.25; P[5][5] = 0.25;
    // Attitude std: ~1.8 deg -> 0.001
    P[6][6] = 0.001; P[7][7] = 0.001; P[8][8] = 0.001;
    // Accel bias std: 0.01 m/s^2 -> 1e-4
    P[9][9] = 1e-4; P[10][10] = 1e-4; P[11][11] = 1e-4;
    // Gyro bias std: 0.00002 rad/s -> 4e-10
    P[12][12] = 4e-10; P[13][13] = 4e-10; P[14][14] = 4e-10;
}

} // anonymous namespace

extern "C" {

IDR_EXPORT void idr_init(double init_heading_rad) {
    for (int i = 0; i < 3; ++i) {
        g_state.p[i] = 0.0;
        g_state.v[i] = 0.0;
        g_state.ba[i] = 0.0;
        g_state.bg[i] = 0.0;
    }
    g_state.heading = wrap_heading(init_heading_rad);
    init_covariance(g_state.P);
    g_initialized = true;
}

IDR_EXPORT void idr_step_200hz(
    double dt,
    double ax_veh, double ay_veh, double az_veh,
    double gx_veh, double gy_veh, double gz_veh,
    double forward_speed_ref
) {
    if (!g_initialized) {
        idr_init(0.0);
    }

    if (dt <= 0.0 || dt > 0.1) {
        dt = 0.005; // Default to 200Hz step (5ms)
    }

    // 1. Correct gyro by estimated biases
    double gz_corr = gz_veh - g_state.bg[2];

    // 2. Propagate nominal navigation azimuth
    g_state.heading = wrap_heading(g_state.heading - gz_corr * dt);

    // 3. Project forward velocity along heading using Non-Holonomic Constraint (NHC)
    double sin_h = std::sin(g_state.heading);
    double cos_h = std::cos(g_state.heading);

    double v_fwd = std::max(0.0, forward_speed_ref);
    g_state.v[0] = v_fwd * sin_h; // East
    g_state.v[1] = v_fwd * cos_h; // North
    g_state.v[2] = 0.0;           // Up (leveled NHC assumption)

    // 4. Propagate ENU position
    g_state.p[0] += g_state.v[0] * dt;
    g_state.p[1] += g_state.v[1] * dt;
    g_state.p[2] += g_state.v[2] * dt;

    // 5. Covariance propagation (discrete time step Q)
    const double q_pos = 1e-6 * dt;
    const double q_vel = 1e-4 * dt;
    const double q_att = 1e-7 * dt;

    g_state.P[0][0] += g_state.P[3][3] * dt * dt + q_pos;
    g_state.P[1][1] += g_state.P[4][4] * dt * dt + q_pos;
    g_state.P[2][2] += g_state.P[5][5] * dt * dt + q_pos;

    g_state.P[3][3] += q_vel;
    g_state.P[4][4] += q_vel;
    g_state.P[5][5] += q_vel;

    g_state.P[8][8] += g_state.P[14][14] * dt * dt + q_att;
}

IDR_EXPORT void idr_get_nav_state(
    double* east,
    double* north,
    double* up,
    double* heading
) {
    if (east) *east = g_state.p[0];
    if (north) *north = g_state.p[1];
    if (up) *up = g_state.p[2];
    if (heading) *heading = g_state.heading;
}

IDR_EXPORT void idr_set_position(double east, double north, double up) {
    g_state.p[0] = east;
    g_state.p[1] = north;
    g_state.p[2] = up;
}

IDR_EXPORT void idr_reset(void) {
    idr_init(0.0);
}

} // extern "C"
