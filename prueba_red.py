# -*- coding: utf-8 -*-
"""
Baseline Reference Tracking for On-Demand Adaptive PID Control.
Evaluates the robustness of the EKF-SNPID controller using standard 
kinematic trajectories (sinusoidal and rose curves) alongside real-time 
Hamiltonian prediction via TensorFlow.
"""

import time
import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- ZMQ Client ---
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERROR] Missing dependency. Please run: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

# --- TensorFlow Backend ---
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
except Exception as e:
    print(f"[WARNING] TensorFlow not available. Physical H will be plotted exclusively. Details: {e}")
    TF_AVAILABLE = False

MODEL_PATH = "hamiltonian_tf_model.keras"

def rmse_keras(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2_keras(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - ss_res / (ss_tot + K.epsilon())

def _load_tf_model(path):
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(path):
        print(f"[INFO] TF model for H not found: {path}")
        return None
    try:
        print(f"[OK] Loading TF model (safe_mode=False): {path}")
        return keras.models.load_model(
            path,
            custom_objects={"rmse_keras": rmse_keras, "r2_keras": r2_keras},
            safe_mode=False
        )
    except TypeError:
        try:
            keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
        print(f"[OK] Loading TF model (unsafe mode): {path}")
        return keras.models.load_model(
            path,
            custom_objects={"rmse_keras": rmse_keras, "r2_keras": r2_keras}
        )
    except Exception as e:
        print(f"[WARNING] Failed to load TF model. Details: {e}")
        return None

def r2_score_np(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

# --- Utilities ---
def wrap_angle(a): 
    return math.atan2(math.sin(a), math.cos(a))

def draw_robot_ax(ax, pose, L, l, color='C1'):
    x, y, th = pose
    pts_body = np.array([[ L,  0],[ 0,  l],[-L,  0],[ 0, -l],[ L,  0]], float).T
    Rm = np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
    pts_world = Rm @ pts_body + np.array([[x],[y]])
    ax.plot(pts_world[0], pts_world[1], color=color, lw=2)
    tip = np.array([x+0.25*math.cos(th), y+0.25*math.sin(th)])
    ax.arrow(x, y, tip[0]-x, tip[1]-y, head_width=0.05, length_includes_head=True, color=color)

# --- YouBot Wrapper ---
class YouBot:
    def __init__(self, host='127.0.0.1', port=23000, wheel_radius=0.10, w_min=-2.5, w_max=2.5):
        self.r = wheel_radius
        self.w_min = np.full(4, w_min, float)
        self.w_max = np.full(4, w_max, float)
        print("[INFO] Connecting to CoppeliaSim via ZMQ...")
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.getObject('sim')
        print("[OK] Connected.")
        
        self.youBot   = self.sim.getObject('/youBot')
        self.motor_fl = self.sim.getObject('/youBot/rollingJoint_fl')
        self.motor_fr = self.sim.getObject('/youBot/rollingJoint_fr')
        self.motor_rl = self.sim.getObject('/youBot/rollingJoint_rl')
        self.motor_rr = self.sim.getObject('/youBot/rollingJoint_rr')
        print("[OK] Handles retrieved.")

    def get_pose(self):
        p = self.sim.getObjectPosition(self.youBot, -1)
        o = self.sim.getObjectOrientation(self.youBot, -1)
        return np.array([float(p[0]), float(p[1]), float(o[2])], float)

    def set_wheel_velocities(self, w):
        w = np.asarray(w, float).flatten()
        w = np.maximum(w, self.w_min); w = np.minimum(w, self.w_max)
        self.sim.setJointTargetVelocity(self.motor_fl, float(-w[0]))
        self.sim.setJointTargetVelocity(self.motor_fr, float(-w[1]))
        self.sim.setJointTargetVelocity(self.motor_rl, float(-w[2]))
        self.sim.setJointTargetVelocity(self.motor_rr, float(-w[3]))

    def stop(self): 
        self.set_wheel_velocities([0,0,0,0])

# --- EKF-SNPID Core ---
def ekf_snpid_freeze(x1, x2, x3, w, P, Q, R, n, alpha, umax, freeze_flag):
    x = np.array([x1,x2,x3], float)
    v = float(w @ x)
    u = alpha * np.tanh(v)
    u_sat = np.clip(u, -umax, umax)
    es    = u_sat - u
    e_aw  = x1 + es
    sech2 = 1.0 - np.tanh(v)**2
    H = (alpha * sech2) * x
    S = float(H @ (P @ H) + R)
    
    if freeze_flag:
        return u_sat, w.copy(), P.copy()
        
    Kf     = (P @ H) / S
    P_new  = (np.eye(3) - np.outer(Kf, H)) @ P + Q
    w_new  = w + (n * Kf) * e_aw
    return u_sat, w_new, P_new

# --- Baseline Reference Trajectories ---
def ref_original(t, mode):
    if mode == "sinusoidal":
        xd  = 0.1 * t
        yd  = 0.4 * math.sin(0.5 * t)
        thd = -math.pi/8
        return xd, yd, thd
    elif mode == "rose":
        a   = 0.2 + 0.05*math.cos(5*0.05*t)
        xd  = a * math.cos(0.05*t) - 0.25
        yd  = a * math.sin(0.05*t)
        thd = math.pi/4
        return xd, yd, thd
    else:
        raise ValueError("Invalid reference mode. Use 'sinusoidal' or 'rose'.")

# --- Main Execution ---
def main():
    REF_MODE = "rose"  
    S        = 180.0             # Total simulation duration [s]

    L   = 0.2355    
    l   = 0.15      

    # EKF-SNPID Initialization
    P_init = np.eye(3); Q = 0.1*np.eye(3); Rm = 1e-4
    n_x  = np.array([0.1, 0.1, 0.01])
    n_y  = np.array([0.1, 0.1, 0.01])
    n_th = np.array([0.1, 0.1, 0.01])
    alpha = 1.5
    uMax  = np.array([1.5, 1.5, 1.5])

    # Hysteresis normalization bounds
    e_thr_freeze = 0.03
    e_thr_unfz   = 0.05

    # Tuned weights
    w_x = np.array([1.25, -0.05, 0.08])
    w_y = np.array([8.0, 0.1, 0.2])
    w_t = np.array([3.5, 0.001, 0.2])
    
    P_x = P_init.copy(); P_y = P_init.copy(); P_t = P_init.copy()

    dx3 = 0.0; dy3 = 0.0; dt3 = 0.0
    freeze_x = False; freeze_y = False; freeze_t = False

    bot = YouBot(port=23000, wheel_radius=0.10, w_min=-2.5, w_max=2.5)
    bot.stop(); time.sleep(0.05)

    # Logging structures
    t_log=[]; p_log=[]; pd_log=[]; w_log=[]
    ux_log=[]; uy_log=[]; uth_log=[]
    ex_log=[]; ey_log=[]; eth_log=[]
    exn_log=[]; eyn_log=[]; ethn_log=[]
    Wx_log=[]; Wy_log=[]; Wt_log=[]; H_log=[]

    # Canonical Hamiltonian physical parameters
    m_base = 20.0; m_w = 1.0; m_L = 0.0; r_w = 0.0475
    m_tot = m_base + 4*m_w + m_L
    Iz_chasis = (1/3)*m_base*(L**2 + l**2)
    Iz_rueda  = m_w*(L**2 + l**2) + 0.25*m_w*(r_w**2)
    Iz        = Iz_chasis + 4*Iz_rueda

    model = _load_tf_model(MODEL_PATH) if TF_AVAILABLE else None
    H_pred_list = []  

    print(f"[INFO] EKF-SNPID tracking started. Reference Mode: '{REF_MODE}'.")
    t0 = time.time(); t_prev=None; p_prev=None
    ex_prev=0.0; ey_prev=0.0; eth_prev=0.0

    try:
        while True:
            t = time.time() - t0
            if t > S: break

            xd, yd, thd = ref_original(t, REF_MODE)
            p = bot.get_pose(); xk, yk, th = p

            ex  = xd - xk
            ey  = yd - yk
            eth = wrap_angle(thd - th)

            dx1, dy1, dt1 = ex, ey, eth
            dx2, dy2, dt2 = ex-ex_prev, ey-ey_prev, eth-eth_prev

            if t_prev is not None and p_prev is not None:
                if np.linalg.norm(p - p_prev) > 1e-12:
                    dx3 += ex; dy3 += ey; dt3 += eth
            else:
                dx3 += ex; dy3 += ey; dt3 += eth

            ux,  w_x, P_x = ekf_snpid_freeze(dx1,dx2,dx3, w_x,P_x,Q,Rm, n_x, alpha,uMax[0], freeze_x)
            uy,  w_y, P_y = ekf_snpid_freeze(dy1,dy2,dy3, w_y,P_y,Q,Rm, n_y, alpha,uMax[1], freeze_y)
            uth, w_t, P_t = ekf_snpid_freeze(dt1,dt2,dt3, w_t,P_t,Q,Rm, n_th,alpha,uMax[2], freeze_t)

            exn = abs(ex)
            eyn = abs(ey)
            ethn = abs(eth)

            if (not freeze_x) and (exn < e_thr_freeze): freeze_x=True
            elif freeze_x and (exn > e_thr_unfz):       freeze_x=False

            if (not freeze_y) and (eyn < e_thr_freeze): freeze_y=True
            elif freeze_y and (eyn > e_thr_unfz):       freeze_y=False

            if (not freeze_t) and (ethn < e_thr_freeze): freeze_t=True
            elif freeze_t and (ethn > e_thr_unfz):      freeze_t=False

            alpha_m = th + math.pi/4.0
            A = np.array([
                [ math.sqrt(2)*math.sin(alpha_m), -math.sqrt(2)*math.cos(alpha_m), -(L+l)],
                [ math.sqrt(2)*math.cos(alpha_m),  math.sqrt(2)*math.sin(alpha_m),  (L+l)],
                [ math.sqrt(2)*math.cos(alpha_m),  math.sqrt(2)*math.sin(alpha_m), -(L+l)],
                [ math.sqrt(2)*math.sin(alpha_m), -math.sqrt(2)*math.cos(alpha_m),  (L+l)]
            ], float)
            u_vec = np.array([ux, uy, uth], float)
            w_cmd = A @ u_vec
            bot.set_wheel_velocities(w_cmd)

            H = 0.5*m_tot*(ux**2 + uy**2) + 0.5*Iz*(uth**2)

            H_pred_inst = None
            if model is not None:
                X = np.array([[ux, uy, uth, m_tot]], dtype=np.float32) 
                try:
                    Hp = model.predict(X, verbose=0).ravel()[0]
                    H_pred_inst = float(Hp)
                except Exception as e:
                    if len(H_pred_list) == 0:
                        print(f"[WARNING] Online TF prediction failed. Details: {e}")
                    H_pred_inst = None

            t_log.append(t); p_log.append(p); pd_log.append([xd,yd,thd]); w_log.append(w_cmd)
            ux_log.append(ux); uy_log.append(uy); uth_log.append(uth)
            ex_log.append(ex); ey_log.append(ey); eth_log.append(eth)
            exn_log.append(exn); eyn_log.append(eyn); ethn_log.append(ethn)
            Wx_log.append(w_x.copy()); Wy_log.append(w_y.copy()); Wt_log.append(w_t.copy())
            H_log.append(H); H_pred_list.append(H_pred_inst)

            t_prev=t; p_prev=p
            ex_prev,ey_prev,eth_prev = ex,ey,eth

            time.sleep(0.01)

    finally:
        bot.stop()
        print("[INFO] Simulation stopped.")

    # --- Data Formatting ---
    t_log=np.array(t_log)
    p_log=np.array(p_log).T if p_log else np.zeros((3,0))
    pd_log=np.array(pd_log).T if pd_log else np.zeros((3,0))
    w_log=np.array(w_log).T if w_log else np.zeros((4,0))
    ux_log=np.array(ux_log); uy_log=np.array(uy_log); uth_log=np.array(uth_log)
    ex_log=np.array(ex_log); ey_log=np.array(ey_log); eth_log=np.array(eth_log)
    exn_log=np.array(exn_log); eyn_log=np.array(eyn_log); ethn_log=np.array(ethn_log)
    Wx_log=np.array(Wx_log).T if Wx_log else np.zeros((3,0))
    Wy_log=np.array(Wy_log).T if Wy_log else np.zeros((3,0))
    Wt_log=np.array(Wt_log).T if Wt_log else np.zeros((3,0))
    H_log=np.array(H_log, float)

    if len(H_pred_list) == len(t_log):
        H_pred = np.array([np.nan if v is None else float(v) for v in H_pred_list], float)
    else:
        H_pred = None

    if H_pred is not None and np.isfinite(H_pred).any():
        mask = np.isfinite(H_pred) & np.isfinite(H_log)
        if mask.sum() > 5:
            err  = H_pred[mask] - H_log[mask]
            mse  = np.mean(err**2)
            rmse = np.sqrt(mse)
            mae  = np.mean(np.abs(err))
            r2   = r2_score_np(H_log[mask], H_pred[mask])
            rng  = (H_log[mask].max() - H_log[mask].min()) + 1e-12
            stdH = np.std(H_log[mask]) + 1e-12
            print(f"[Hamiltonian Comparison] RMSE={rmse:.6e}  MAE={mae:.6e}  R2={r2:.6f}  "
                  f"NRMSE(range)={rmse/rng:.6e}  NRMSE(std)={rmse/stdH:.6e}")
        else:
            print("[INFO] Insufficient valid points for Hamiltonian metrics.")

    # --- Plotting Subroutines ---
    plt.figure("Position vs Time", figsize=(8,8))
    for i, lab in enumerate(['x [m]', 'y [m]', r'$\theta$ [rad]']):
        ax = plt.subplot(3,1,i+1); ax.grid(True)
        if pd_log.shape[1]>0: ax.plot(t_log, pd_log[i], '--', lw=1.5, label=f'{lab}_d')
        if p_log.shape[1]>0:  ax.plot(t_log, p_log[i], lw=1.5, label=lab)
        ax.set_ylabel(lab)
        if i==2: ax.set_xlabel('t [s]')
        ax.legend(loc='best')

    plt.figure("XY Trajectory", figsize=(7,7))
    ax = plt.gca(); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    if pd_log.shape[1]>0: ax.plot(pd_log[0], pd_log[1], 'k--', lw=1.5, label='Reference')
    if p_log.shape[1]>0:
        ax.plot(p_log[0], p_log[1], 'b', lw=1.8, label='Real')
        if p_log.shape[1]>0:
            draw_robot_ax(ax, p_log[:,-1], L, l)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(loc='best'); ax.set_title(f"EKF-SNPID Tracking ({REF_MODE})")

    plt.figure("Control Signals", figsize=(9,4))
    plt.grid(True)
    plt.plot(t_log, ux_log, label='$u_x$')
    plt.plot(t_log, uy_log, label='$u_y$')
    plt.plot(t_log, uth_log, label=r'$u_\theta$')
    plt.xlabel('t [s]'); plt.ylabel('Control Action'); plt.legend(loc='best')

    plt.figure("Errors", figsize=(9,4))
    plt.grid(True)
    plt.plot(t_log, ex_log, label='$e_x$')
    plt.plot(t_log, ey_log, label='$e_y$')
    plt.plot(t_log, eth_log, label=r'$e_\theta$')
    plt.xlabel('t [s]'); plt.legend(loc='best')

    plt.figure("Normalized Errors", figsize=(9,4))
    plt.grid(True)
    plt.plot(t_log, exn_log, label=r'$e_{x,n}$')
    plt.plot(t_log, eyn_log, label=r'$e_{y,n}$')
    plt.plot(t_log, ethn_log, label=r'$e_{\theta,n}$')
    plt.axhline(e_thr_freeze, linestyle='--', color='k', label='Freeze Threshold')
    plt.axhline(e_thr_unfz,   linestyle='--', color='r', label='Unfreeze Threshold')
    plt.xlabel('t [s]'); plt.legend(loc='best')

    plt.figure("Learned Gains", figsize=(9,9))
    ax=plt.subplot(3,1,1); ax.grid(True)
    if Wx_log.shape[1]>0: ax.plot(t_log, Wx_log[0], t_log, Wx_log[1], t_log, Wx_log[2])
    ax.legend(['$Kp_x$', '$Kd_x$', '$Ki_x$']); ax.set_title('X-Axis Gains')
    
    ax=plt.subplot(3,1,2); ax.grid(True)
    if Wy_log.shape[1]>0: ax.plot(t_log, Wy_log[0], t_log, Wy_log[1], t_log, Wy_log[2])
    ax.legend(['$Kp_y$', '$Kd_y$', '$Ki_y$']); ax.set_title('Y-Axis Gains')
    
    ax=plt.subplot(3,1,3); ax.grid(True)
    if Wt_log.shape[1]>0: ax.plot(t_log, Wt_log[0], t_log, Wt_log[1], t_log, Wt_log[2])
    ax.legend([r'$Kp_\theta$', r'$Kd_\theta$', r'$Ki_\theta$']); ax.set_title(r'$\theta$ Gains'); ax.set_xlabel('t [s]')

    if w_log.shape[1]>0:
        plt.figure("Wheel Velocities", figsize=(9,4))
        plt.grid(True)
        for i in range(4): plt.plot(t_log, w_log[i], label=f'$v_{i+1}$')
        plt.xlabel('t [s]'); plt.ylabel('Velocity [rad/s]'); plt.legend(loc='best')

    plt.figure("Hamiltonian", figsize=(10,4))
    plt.grid(True)
    plt.plot(t_log, H_log, lw=1.2, label='Real H (Physical)')
    if H_pred is not None and np.isfinite(H_pred).any():
        plt.plot(t_log, H_pred, '--', label='Predicted H (TF)')
    plt.xlabel('t [s]'); plt.ylabel('Energy [J]')
    plt.title('Canonical Hamiltonian')
    plt.legend(loc='best')

    if H_pred is not None and np.isfinite(H_pred).any():
        plt.figure("Hamiltonian Error", figsize=(10,3))
        diff = H_pred - H_log
        plt.plot(t_log, diff)
        plt.grid(True); plt.xlabel("t [s]"); plt.ylabel("H_pred - H_real [J]")
        plt.title("Absolute Hamiltonian Error")
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()