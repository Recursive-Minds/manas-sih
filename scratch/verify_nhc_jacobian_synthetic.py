import numpy as np
from scipy.spatial.transform import Rotation as R

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def main():
    print("=== SYNTHETIC CIRCLE KINEMATICS & NHC JACOBIAN VERIFICATION ===")
    
    # Vehicle moving in a circle of radius R=50m at speed v=10 m/s
    radius_m = 50.0
    speed_mps = 10.0
    w_z_true = speed_mps / radius_m  # 0.2 rad/s
    
    heading_rad = 0.0  # Facing North
    yaw_enu_rad = np.pi / 2.0 - heading_rad
    q_b_to_n = R.from_euler("z", yaw_enu_rad)
    C_b_n = q_b_to_n.as_matrix()
    C_n_b = C_b_n.T
    
    # Velocity in ENU: Moving North -> [0, 10, 0]
    v_enu = np.array([0.0, speed_mps, 0.0])
    v_b = C_n_b @ v_enu  # Should be [10, 0, 0] in body (Forward, Lateral, Vertical)
    print(f"Body Velocity (Forward, Lat, Vert): {v_b}")
    
    # Construct NHC Error State Jacobian H_nhc (2 x 15)
    # Mapping error state [d_p, d_v, d_theta, d_ba, d_bg] to lateral and vertical body velocity innovation
    H_nhc = np.zeros((2, 15), dtype=np.float64)
    H_nhc[0, 3:6] = C_n_b[1, :]
    H_nhc[0, 6:9] = -(C_n_b @ skew(v_enu))[1, :]
    H_nhc[1, 3:6] = C_n_b[2, :]
    H_nhc[1, 6:9] = -(C_n_b @ skew(v_enu))[2, :]
    
    print("\nJacobian Sub-matrix H_v (w.r.t ENU Velocity):")
    print(H_nhc[:, 3:6])
    print("\nJacobian Sub-matrix H_theta (w.r.t Orientation Error):")
    print(H_nhc[:, 6:9])
    
    # Test perturbed heading error d_theta = +0.1 rad (~5.7 deg right of North)
    # Position velocity becomes [10*sin(0.1), 10*cos(0.1), 0] = [0.998, 9.95, 0] in ENU
    # In true body frame, lateral velocity v_lat = -0.998 m/s (slipping left)
    dtheta = 0.1
    v_enu_pert = np.array([speed_mps * np.sin(dtheta), speed_mps * np.cos(dtheta), 0.0])
    v_b_pert = C_n_b @ v_enu_pert
    y_nhc = np.array([0.0 - v_b_pert[1], 0.0 - v_b_pert[2]])
    print(f"\nActual Lateral Innovation (0 - v_b_lat): {y_nhc[0]:.4f} m/s")
    
    # Expected Jacobian prediction H_theta * [0, 0, dtheta]:
    dtheta_vec = np.array([0.0, 0.0, dtheta])
    predicted_y = (H_nhc[0, 6:9] @ dtheta_vec)
    print(f"Jacobian Predicted Innovation:           {predicted_y:.4f} m/s")
    
    err = abs(y_nhc[0] - predicted_y)
    print(f"Linearization Residual Error:             {err:.6f} m/s")
    
    if err < 0.01:
        print("\nSUCCESS: NHC Jacobian H_NHC signs and cross-product terms match theoretical kinematics perfectly!")
    else:
        print("\nFAILURE: Mismatch in Jacobian derivation!")

if __name__ == "__main__":
    main()
