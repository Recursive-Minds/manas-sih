#ifndef IDR_CORE_H
#define IDR_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define IDR_EXPORT __declspec(dllexport)
#else
  #define IDR_EXPORT __attribute__((visibility("default")))
#endif

/**
 * @brief Initialize or re-anchor the 200Hz IDR Core Engine.
 * @param init_heading_rad Initial navigation azimuth / heading in radians [0, 2*pi).
 */
IDR_EXPORT void idr_init(double init_heading_rad);

/**
 * @brief High-rate 200Hz inertial step update.
 * @param dt Time delta in seconds (typically 0.005s for 200Hz).
 * @param ax_veh Forward acceleration in vehicle frame (m/s^2).
 * @param ay_veh Lateral acceleration in vehicle frame (m/s^2).
 * @param az_veh Earth-down / vertical acceleration in vehicle frame (m/s^2).
 * @param gx_veh Roll rate in vehicle frame (rad/s).
 * @param gy_veh Pitch rate in vehicle frame (rad/s).
 * @param gz_veh Yaw rate in vehicle frame (rad/s).
 * @param forward_speed_ref AI or wheel-speed forward velocity reference (m/s).
 */
IDR_EXPORT void idr_step_200hz(
    double dt,
    double ax_veh, double ay_veh, double az_veh,
    double gx_veh, double gy_veh, double gz_veh,
    double forward_speed_ref
);

/**
 * @brief Retrieve current 3D position (ENU) and navigation heading.
 * @param east Output pointer for East coordinate in meters.
 * @param north Output pointer for North coordinate in meters.
 * @param up Output pointer for Up coordinate in meters.
 * @param heading Output pointer for Navigation azimuth in radians [0, 2*pi).
 */
IDR_EXPORT void idr_get_nav_state(
    double* east,
    double* north,
    double* up,
    double* heading
);

/**
 * @brief Manually set the ENU origin / position.
 */
IDR_EXPORT void idr_set_position(double east, double north, double up);

/**
 * @brief Reset the filter state and covariance to nominal rest.
 */
IDR_EXPORT void idr_reset(void);

#ifdef __cplusplus
}
#endif

#endif // IDR_CORE_H
