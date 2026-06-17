# -*- coding: utf-8 -*-
"""
Online EKF-SNPID Controller with Real-Time Neural Network Re-adaptation.
Part of the "On-Demand Adaptive PID Control via Neural Networks" framework.
Features dynamic payload handling, Hamiltonian predictions, and online 
model updating using a FIFO experience replay buffer.
"""

import time
import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Load Configuration (Cuboid) ---
CUBOID_PATH   = '/Cuboid'   
ML_INIT       = 0.0               # Initial load mass [kg]
ML_NEW        = 10.0              # Switched load mass [kg]
ML_SWITCH_T   = None              # Time of mass switch (None = S/2)
ML_EPS        = 1e-6              # Fallback epsilon for minimum mass

# --- Hamiltonian Trigger Configuration ---
H_DIFF_ABS_THR = 0.5              # Absolute discrepancy threshold |H_pred - H_real| [J]
H_DIFF_REL_THR = 0.25             # Relative discrepancy threshold vs |H_real|
H_EPS_DEN      = 1e-6             # Epsilon to prevent division by zero
UNFREEZE_H_HOLD = 3.0             # Duration to hold unfreeze after H-trigger [s]
PLOT_SWITCH_MARK = True           # Mark switch and H events on plots

# --- Online Learning (Re-adaptation) Configuration ---
ONLINE_BUFFER_SIZE = 512
ONLINE_BATCH_SIZE  = 128
ONLINE_STEPS_PER_TRIGGER = 32
ONLINE_LR = 1e-3
ONLINE_MIN_POINTS = 50
COOLDOWN_BETWEEN_TRIGGERS = 2.0   # [s]

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

def _load_tf_model_for_online(path):
    if not TF_AVAILABLE or not os.path.exists(path):
        print(f"[INFO] TF model not found or TF unavailable: {path}")
        return None
    try:
        print(f"[OK] Loading TF model (safe_mode=False): {path}")
        model = keras.models.load_model(
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
        model = keras.models.load_model(
            path,
            custom_objects={"rmse_keras": rmse_keras, "r2_keras": r2_keras}
        )
    except Exception as e:
        print(f"[WARNING] Failed to load TF model. Details: {e}")
        return None

    model.compile(optimizer=keras.optimizers.Adam(ONLINE_LR), loss="mse")
    return model

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
        
        try:
            self.cuboid = self.sim.getObject(CUBOID_PATH)
            print(f"[OK] Cuboid linked: {CUBOID_PATH}")
        except Exception as e:
            self.cuboid = None
            print(f"[WARNING] Cuboid not found at {CUBOID_PATH}. Details: {e}")

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

# --- Reference Trajectories ---
def ref_original(t, mode):
    if mode == "sinusoidal":
        xd  = 0.1 * t
        yd  = 0.4 * math.sin(0.5 * t)
        thd = -math.pi/8
        return xd, yd, thd
    elif mode == "rosa":
        a   = 0.2 + 0.05*math.cos(5*0.05*t)
        xd  = a * math.cos(0.05*t) - 0.25
        yd  = a * math.sin(0.05*t)
        thd = math.pi/4
        return xd, yd, thd
    else:
        raise ValueError("Invalid reference mode. Use 'sinusoidal' or 'rosa'.")

# --- FIFO Experience Replay Buffer ---
class OnlineBuffer:
    """Stores recent state-Hamiltonian pairs for continuous network readaptation."""
    def __init__(self, capacity=ONLINE_BUFFER_SIZE):
        self.cap = capacity
        self.X = np.zeros((capacity,3), dtype=np.float32)
        self.y = np.zeros((capacity,1), dtype=np.float32)
        self.n = 0
        self.ptr = 0

    def push(self, ux, uy, uth, H):
        self.X[self.ptr] = (ux, uy, uth)
        self.y[self.ptr,0] = H
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch_size):
        if self.n == 0: 
            return None, None
        idx = np.random.choice(self.n, size=min(batch_size, self.n), replace=False)
        return self.X[idx], self.y[idx]

    def ready(self, min_points=ONLINE_MIN_POINTS):
        return self.n >= min_points

# --- Main Execution ---
def main():
    REF_MODO = "rosa"  
    S        = 180.0             # Total simulation duration [s]

    L = 0.1981; l = 0.1990

    # EKF-SNPID Initialization
    P_init = np.eye(3); Q = 0.1*np.eye(3); Rm = 1e-4
    n_x  = np.array([0.1, 0.1, 0.01])
    n_y  = np.array([0.1, 0.1, 0.01])
    n_th = np.array([0.1, 0.1, 0.01])
    alpha = 1.5
    uMax  = np.array([1.5, 1.5, 1.5])

    # Hysteresis normalization bounds
    e_thr_freeze = 0.05; e_thr_unfz = 0.10

    # Tuned initial weights
    w_x = np.array([0.8, 0, 0.06])
    w_y = np.array([6.0, 0.05, 0.02])
    w_t = np.array([1.0, 0.08, 0.02])
    
    P_x = P_init.copy(); P_y = P_init.copy(); P_t = P_init.copy()

    dx3 = 0.0; dy3 = 0.0; dt3 = 0.0
    freeze_x = False; freeze_y = False; freeze_t = False

    bot = YouBot(port=23000, wheel_radius=0.10, w_min=-2.5, w_max=2.5)
    bot.stop(); time.sleep(0.05)

    m_base = 20.0; m_w = 1.0; r_w = 0.0475 

    m_L = ML_INIT
    if bot.cuboid is not None:
        try:
            bot.sim.setShapeMass(bot.cuboid, ML_INIT)
        except Exception as e:
            print(f"[WARNING] Engine rejected 0 mass. Using epsilon. Details: {e}")
            bot.sim.setShapeMass(bot.cuboid, ML_EPS)
            m_L = ML_EPS
    else:
        print("[WARNING] Cuboid missing from scene: m_L tracking will be internal only.")

    t_switch = (S*0.5) if (ML_SWITCH_T is None) else float(ML_SWITCH_T)
    mass_switched = False
    switch_time_logged = None

    # Logging structures
    t_log=[]; p_log=[]; pd_log=[]; w_log=[]
    ux_log=[]; uy_log=[]; uth_log=[]
    ex_log=[]; ey_log=[]; eth_log=[]
    exn_log=[]; eyn_log=[]; ethn_log=[]
    Wx_log=[]; Wy_log=[]; Wt_log=[]; H_log=[]
    H_pred_list = []   
    h_unfreeze_events = [] 

    model = _load_tf_model_for_online(MODEL_PATH) if TF_AVAILABLE else None
    buffer = OnlineBuffer()
    last_trigger_time = -1e9
    hold_unfreeze_until = 0.0

    print(f"[INFO] EKF-SNPID tracking started. Reference Mode: '{REF_MODO}'.")
    t0 = time.time(); t_prev=None; p_prev=None
    ex_prev=0.0; ey_prev=0.0; eth_prev=0.0

    try:
        while True:
            t = time.time() - t0
            if t > S: break

            xd, yd, thd = ref_original(t, REF_MODO)
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

            exn = abs(ex); eyn = abs(ey); ethn = abs(eth)

            if t < hold_unfreeze_until:
                freeze_x = freeze_y = freeze_t = False
            else:
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

            m_tot = m_base + 4*m_w + m_L
            Iz_chasis = (1/3)*m_base*(L**2 + l**2)
            Iz_rueda  = m_w*(L**2 + l**2) + 0.25*m_w*(r_w**2)
            Iz        = Iz_chasis + 4*Iz_rueda
            H_real = 0.5*m_tot*(ux**2 + uy**2) + 0.5*Iz*(uth**2)

            # --- Dynamic Mass Switching ---
            if (not mass_switched) and (t >= (t_switch if ML_SWITCH_T is not None else (S*0.5))):
                if bot.cuboid is not None:
                    try:
                        bot.sim.setShapeMass(bot.cuboid, ML_NEW)
                        print(f"[INFO] Load mass (m_L) changed from {m_L:.6f} to {ML_NEW:.6f} kg @ t={t:.2f}s")
                        m_L = ML_NEW
                    except Exception as e:
                        print(f"[WARNING] setShapeMass failed. Details: {e}")
                        m_L = ML_NEW
                else:
                    m_L = ML_NEW
                    print(f"[INFO] Internal load mass changed to {m_L:.6f} kg @ t={t:.2f}s (no cuboid)")
                mass_switched = True
                switch_time_logged = t  

            # --- TensorFlow Prediction (3 Inputs) ---
            H_pred_inst = None
            if model is not None:
                X = np.array([[ux, uy, uth]], dtype=np.float32)
                try:
                    Hp = model.predict(X, verbose=0).ravel()[0]
                    H_pred_inst = float(Hp)
                except Exception as e:
                    if len(H_pred_list) == 0:
                        print(f"[WARNING] Online TF prediction failed. Details: {e}")
                    H_pred_inst = None

            # --- Hamiltonian Discrepancy Trigger: Unfreeze + Hold + Re-adapt ---
            trigger = False
            if H_pred_inst is not None and np.isfinite(H_pred_inst):
                diff_abs = abs(H_pred_inst - H_real)
                denom    = max(abs(H_real), H_EPS_DEN)
                diff_rel = diff_abs / denom
                
                if (diff_abs > H_DIFF_ABS_THR) or (diff_rel > H_DIFF_REL_THR):
                    print(f"[H-TRIGGER] Absolute Error: {diff_abs:.4f} | Relative Error: {diff_rel:.4f}")
                    trigger = True
                    freeze_x = freeze_y = freeze_t = False
                    hold_unfreeze_until = t + UNFREEZE_H_HOLD
                    h_unfreeze_events.append((t, diff_abs, diff_rel))
                    buffer.push(ux, uy, uth, H_real)

            if not trigger:
                buffer.push(ux, uy, uth, H_real)

            # --- Online Updates ---
            if model is not None and buffer.ready() and (t - last_trigger_time) >= COOLDOWN_BETWEEN_TRIGGERS:
                for _ in range(ONLINE_STEPS_PER_TRIGGER):
                    Xb, yb = buffer.sample(ONLINE_BATCH_SIZE)
                    model.train_on_batch(Xb, yb)
                last_trigger_time = t

            # --- Logging ---
            t_log.append(t); p_log.append(p); pd_log.append([xd,yd,thd]); w_log.append(w_cmd)
            ux_log.append(ux); uy_log.append(uy); uth_log.append(uth)
            ex_log.append(ex); ey_log.append(ey); eth_log.append(eth)
            exn_log.append(exn); eyn_log.append(eyn); ethn_log.append(ethn)
            Wx_log.append(w_x.copy()); Wy_log.append(w_y.copy()); Wt_log.append(w_t.copy())
            H_log.append(H_real); H_pred_list.append(H_pred_inst)

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
    def mark_switch(ax):
        if PLOT_SWITCH_MARK and (ML_SWITCH_T is None or switch_time_logged is not None):
            ts = (switch_time_logged if switch_time_logged is not None else (t_log[-1]/2.0))
            ax.axvline(ts, linestyle='--', linewidth=1.2)
            ax.text(ts, ax.get_ylim()[1]*0.9, 'm_L Switch', rotation=90, va='top', ha='right')

    plt.figure("Position vs Time", figsize=(8,8))
    for i, lab in enumerate(['x [m]', 'y [m]', r'$\theta$ [rad]']):
        ax = plt.subplot(3,1,i+1); ax.grid(True)
        if pd_log.shape[1]>0: ax.plot(t_log, pd_log[i], '--', lw=1.5, label=f'{lab}_d')
        if p_log.shape[1]>0:  ax.plot(t_log, p_log[i], lw=1.5, label=lab)
        ax.set_ylabel(lab)
        if i==2: ax.set_xlabel('t [s]')
        ax.legend(loc='best'); mark_switch(ax)

    plt.figure("XY Trajectory", figsize=(7,7))
    ax = plt.gca(); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    if pd_log.shape[1]>0: ax.plot(pd_log[0], pd_log[1], 'k--', lw=1.5, label='Reference')
    if p_log.shape[1]>0:
        ax.plot(p_log[0], p_log[1], 'b', lw=1.8, label='Real')
        if p_log.shape[1]>0:
            draw_robot_ax(ax, p_log[:,-1], L, l)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(loc='best'); ax.set_title(f"EKF-SNPID Tracking ({REF_MODO})")

    plt.figure("Control Signals", figsize=(9,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, ux_log, label='$u_x$')
    ax.plot(t_log, uy_log, label='$u_y$')
    ax.plot(t_log, uth_log, label=r'$u_\theta$')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Control Action'); ax.legend(loc='best'); mark_switch(ax)

    plt.figure("Errors", figsize=(9,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, ex_log, label='$e_x$')
    ax.plot(t_log, ey_log, label='$e_y$')
    ax.plot(t_log, eth_log, label=r'$e_\theta$')
    ax.set_xlabel('t [s]'); ax.legend(loc='best'); mark_switch(ax)

    plt.figure("Normalized Errors", figsize=(9,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, exn_log, label=r'$e_{x,n}$')
    ax.plot(t_log, eyn_log, label=r'$e_{y,n}$')
    ax.plot(t_log, ethn_log, label=r'$e_{\theta,n}$')
    ax.axhline(e_thr_freeze, linestyle='--', color='k', label='Freeze Threshold')
    ax.axhline(e_thr_unfz,   linestyle='--', color='r', label='Unfreeze Threshold')
    ax.set_xlabel('t [s]'); ax.legend(loc='best'); mark_switch(ax)

    plt.figure("Learned Gains", figsize=(9,9))
    ax=plt.subplot(3,1,1); ax.grid(True)
    if Wx_log.shape[1]>0: ax.plot(t_log, Wx_log[0], t_log, Wx_log[1], t_log, Wx_log[2])
    ax.legend(['$Kp_x$', '$Kd_x$', '$Ki_x$']); ax.set_title('X-Axis Gains'); mark_switch(ax)
    
    ax=plt.subplot(3,1,2); ax.grid(True)
    if Wy_log.shape[1]>0: ax.plot(t_log, Wy_log[0], t_log, Wy_log[1], t_log, Wy_log[2])
    ax.legend(['$Kp_y$', '$Kd_y$', '$Ki_y$']); ax.set_title('Y-Axis Gains'); mark_switch(ax)
    
    ax=plt.subplot(3,1,3); ax.grid(True)
    if Wt_log.shape[1]>0: ax.plot(t_log, Wt_log[0], t_log, Wt_log[1], t_log, Wt_log[2])
    ax.legend([r'$Kp_\theta$', r'$Kd_\theta$', r'$Ki_\theta$']); ax.set_title(r'$\theta$ Gains'); ax.set_xlabel('t [s]'); mark_switch(ax)

    if w_log.shape[1]>0:
        plt.figure("Wheel Velocities", figsize=(9,4))
        ax = plt.gca(); ax.grid(True)
        for i in range(4): ax.plot(t_log, w_log[i], label=f'$v_{i+1}$')
        ax.set_xlabel('t [s]'); ax.set_ylabel('Velocity [rad/s]'); ax.legend(loc='best'); mark_switch(ax)

    plt.figure("Hamiltonian", figsize=(10,4))
    ax = plt.gca(); ax.grid(True)
    ax.plot(t_log, H_log, lw=1.2, label='Real H (Physical)')
    if H_pred is not None and np.isfinite(H_pred).any():
        ax.plot(t_log, H_pred, '--', label='Predicted H (TF)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Energy [J]')
    ax.set_title('Canonical Hamiltonian')
    ax.legend(loc='best'); mark_switch(ax)

    if H_pred is not None and np.isfinite(H_pred).any():
        plt.figure("Hamiltonian Error", figsize=(10,3))
        ax = plt.gca()
        diff = H_pred - H_log
        ax.plot(t_log, diff)
        ax.grid(True); ax.set_xlabel("t [s]"); ax.set_ylabel("H_pred - H_real [J]")
        ax.set_title("Absolute Hamiltonian Error"); mark_switch(ax)
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()