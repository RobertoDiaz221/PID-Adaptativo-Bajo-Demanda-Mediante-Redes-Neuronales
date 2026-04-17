# ZMQ + EKF-SNPID (freeze/histeresis) + referencias ORIGINAL "sinusoidal" y "rosa" + gráficas
# + PREDICCIÓN DE HAMILTONIANO CON TENSORFLOW (comparación con H fí­sico)
# + SEGUIMIENTO POR PUNTOS (x,y,theta) con tolerancias y avance por error, y stop al terminar (loop=False)

import time, math, sys, os
import numpy as np
import matplotlib.pyplot as plt

# -------- Cliente ZMQ --------
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERROR] Instala:  pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

# -------- TensorFlow (opcional) --------
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
except Exception as e:
    print("[WARN] TensorFlow no disponible, sólo se graficará H fí­sico. Detalle:", e)
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
        print("[INFO] No se encontró el modelo TF para H:", path)
        return None
    try:
        print(f"[OK] Cargando modelo TF (safe_mode=False): {path}")
        return keras.models.load_model(
            path,
            custom_objects={"rmse_keras": rmse_keras, "r2_keras": r2_keras},
            safe_mode=False
        )
    except TypeError:
        try:
            print("[INFO] Reintentando con enable_unsafe_deserialization()")
            keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
        print(f"[OK] Cargando modelo TF (modo inseguro): {path}")
        return keras.models.load_model(
            path,
            custom_objects={"rmse_keras": rmse_keras, "r2_keras": r2_keras}
        )
    except Exception as e:
        print("[WARN] Falló la carga del modelo TF. Detalle:", e)
        return None

def r2_score_np(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

# -------- Utilidades --------
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

# -------- Envoltura YouBot --------
class YouBot:
    def __init__(self, host='127.0.0.1', port=23000, wheel_radius=0.10, w_min=-2.5, w_max=2.5):
        self.r = wheel_radius
        self.w_min = np.full(4, w_min, float)
        self.w_max = np.full(4, w_max, float)
        print("[INFO] Conectando a CoppeliaSim (ZMQ)...")
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.getObject('sim')
        print("[OK] Conectado.")
        print("[INFO] Obteniendo handles...")
        self.youBot   = self.sim.getObject('/youBot')
        self.motor_fl = self.sim.getObject('/youBot/rollingJoint_fl')
        self.motor_fr = self.sim.getObject('/youBot/rollingJoint_fr')
        self.motor_rl = self.sim.getObject('/youBot/rollingJoint_rl')
        self.motor_rr = self.sim.getObject('/youBot/rollingJoint_rr')
        print("[OK] Handles listos.")

    def get_pose(self):
        p = self.sim.getObjectPosition(self.youBot, -1)
        o = self.sim.getObjectOrientation(self.youBot, -1)  # [ax,ay,az]
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

# -------- EKF-SNPID (freeze/histeresis) --------
def ekf_snpid_freeze(x1,x2,x3, w,P,Q,R, n,alpha, umax, freeze_flag):
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

# -------- Referencias por tiempo --------
def ref_original(t, modo):
    """
    xd, yd, thetad, modos: "sinusoidal" | "rosa"
    """
    if modo == "sinusoidal":
        xd  = 0.1 * t
        yd  = 0.4 * math.sin(0.5 * t)
        thd = math.pi/8
        return xd, yd, thd
    elif modo == "rosa":
        a   = 1 + 0.5*math.cos(5*0.05*t)
        xd  = a * math.cos(0.05*t) - 1.5
        yd  = a * math.sin(0.05*t)
        thd = math.pi/2
        return xd, yd, thd
    else:
        raise ValueError("Modo de referencia no válido. Usa 'sinusoidal' o 'rosa'.")

# -------- Seguimiento por puntos con orientaciÃ³n --------
class WaypointTrackerPose:

    def __init__(self, waypoints_xyz, tol_pos=0.05, tol_th=5*math.pi/180,
                 require_theta=False, loop=True):
        W = []
        for w in waypoints_xyz:
            if len(w) == 2:
                W.append((float(w[0]), float(w[1]), 0.0))
            else:
                W.append((float(w[0]), float(w[1]), float(w[2])))
        self.wps = np.array(W, float).reshape(-1, 3)
        if self.wps.size == 0:
            raise ValueError("Se requiere al menos un waypoint.")
        self.tol_pos = float(tol_pos)
        self.tol_th  = float(tol_th)
        self.require_theta = bool(require_theta)
        self.loop = bool(loop)
        self.idx = 0
        self.finished = False  # bandera de finalización

    def current_target(self):
        return self.wps[self.idx]

    def step(self, xk, yk, thk):
        if self.finished:
            xd, yd, thd = self.current_target()
            dx = xd - xk; dy = yd - yk
            dist = math.hypot(dx, dy)
            eth  = wrap_angle(thd - thk)
            return xd, yd, thd, self.idx, dist, abs(eth), True

        xd, yd, thd = self.current_target()
        dx = xd - xk; dy = yd - yk
        dist = math.hypot(dx, dy)
        eth  = wrap_angle(thd - thk)
        cond_pos = (dist <= self.tol_pos)
        cond_th  = (abs(eth) <= self.tol_th)

        if (cond_pos and (cond_th if self.require_theta else True)):
            if self.idx < len(self.wps) - 1:
                self.idx += 1
            else:
                if self.loop:
                    self.idx = 0
                else:
                    self.finished = True

            xd, yd, thd = self.current_target()
            dx = xd - xk; dy = yd - yk
            dist = math.hypot(dx, dy)
            eth  = wrap_angle(thd - thk)

        return xd, yd, thd, self.idx, dist, abs(eth), self.finished

# -------- Main --------
def main():
    # ===== Selección de referencia =====
    # Opciones: "sinusoidal" | "rosa" | "puntos"
    REF_MODO = "puntos"

    S        = 15.0    # duración máx. de simulación [s]

    # ===== Geometrí­a YouBot =====
    L = 0.1981; l = 0.1990

    # ===== EKF-SNPID =====
    P_init = np.eye(3); Q = 0.1*np.eye(3); Rm = 1e-4
    n_x  = np.array([0.1, 0.1, 0.01])
    n_y  = np.array([0.1, 0.1, 0.01])
    n_th = np.array([0.1, 0.1, 0.01])
    alpha = 1.5
    uMax  = np.array([1.5, 1.5, 1.5])

    # Freeze por normalización (histéresis)
    e_thr_freeze=0.45; e_thr_unfz=0.56

    # Pesos iniciales
    w_x = np.array([0, 0, 0])
    w_y = np.array([0, 0, 0])
    w_t = np.array([0, 0, 0])

    P_x = P_init.copy(); P_y = P_init.copy(); P_t = P_init.copy()

    # Integrales protegidas
    dx3=0.0; dy3=0.0; dt3=0.0
    freeze_x=False; freeze_y=False; freeze_t=False

    # ===== Conexión =====
    bot = YouBot(port=23000, wheel_radius=0.10, w_min=-2.5, w_max=2.5)
    bot.stop(); time.sleep(0.05)
    cuboid = bot.sim.getObject('/Cuboid')

    # ===== Waypoints con orientación (x, y, theta) =====
    WAYPOINTS_POSE = [
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (2, 0.0, 0.0),
    ]
    WP_TOL_POS   = 0.05               # tolerancia de posición [m]
    WP_TOL_TH    = math.radians(8)    # tolerancia angular [rad]
    WP_REQUIRE_TH= False              # True: exige también orientación para avanzar
    WP_LOOP      = True              # False => se detiene al terminar

    if REF_MODO == "puntos":
        wp_track = WaypointTrackerPose(
            WAYPOINTS_POSE,
            tol_pos=WP_TOL_POS,
            tol_th=WP_TOL_TH,
            require_theta=WP_REQUIRE_TH,
            loop=WP_LOOP
        )

    # ===== Logs =====
    t_log=[]; p_log=[]; pd_log=[]; w_log=[]
    ux_log=[]; uy_log=[]; uth_log=[]
    ex_log=[]; ey_log=[]; eth_log=[]
    exn_log=[]; eyn_log=[]; ethn_log=[]
    Wx_log=[]; Wy_log=[]; Wt_log=[]; H_log=[]
    dist_log=[]             # ||p - p_prev||
    wp_idx_log=[]           # í­ndice de waypoint activo
    wp_dist_log=[]          # distancia a waypoint activo
    wp_eth_log=[]           # |error angular| al waypoint activo

    # Hamiltoniano 
    m_base=20.0; m_w=1.0; m_L=0.0; r_w=0.0475
    m_tot = m_base + 4*m_w + m_L
    Iz_chasis = (1/3)*m_base*(L**2 + l**2)
    Iz_rueda  = m_w*(L**2 + l**2) + 0.25*m_w*(r_w**2)
    Iz        = Iz_chasis + 4*Iz_rueda

    # ===== Preparar modelo TF (si existe) =====
    model = _load_tf_model(MODEL_PATH) if TF_AVAILABLE else None
    H_pred_list = []  # loggear H_pred por muestra

    print(f"[INFO] EKF-SNPID con referencia '{REF_MODO}'.")
    if REF_MODO == "puntos":
        print(f"[INFO] Waypoints: {len(WAYPOINTS_POSE)} | tol_pos={WP_TOL_POS} m | tol_th={WP_TOL_TH:.3f} rad | require_th={WP_REQUIRE_TH} | loop={WP_LOOP}")
    t0 = time.time(); t_prev=None; p_prev=None;
    ex_prev=0.0; ey_prev=0.0; eth_prev=0.0

    try:
        while True:
            t = time.time() - t0
            if t > S: break

            # --- Estado actual ---
            p = bot.get_pose(); xk, yk, th = p

            # --- Referencia según modo ---
            if REF_MODO in ("sinusoidal","rosa"):
                xd, yd, thd = ref_original(t, REF_MODO)
                wp_idx = -1; wp_dist = np.nan; wp_eth = np.nan; wp_finished = False
            elif REF_MODO == "puntos":
                xd, yd, thd, wp_idx, wp_dist, wp_eth, wp_finished = wp_track.step(xk, yk, th)
            else:
                raise ValueError("REF_MODO inválido.")

            # --- Errores (mundo) ---
            ex  = xd - xk
            ey  = yd - yk
            eth = wrap_angle(thd - th)

            # Derivadas discretas
            dx1, dy1, dt1 = ex, ey, eth
            dx2, dy2, dt2 = ex-ex_prev, ey-ey_prev, eth-eth_prev

            # Integrales: sólo si hay avance
            if t_prev is not None and p_prev is not None:
                if np.linalg.norm(p - p_prev) > 2e-3:
                    dx3 += ex; dy3 += ey; dt3 += eth
            else:
                dx3 += ex; dy3 += ey; dt3 += eth

            # --- EKF-SNPID por eje ---
            ux,  w_x, P_x = ekf_snpid_freeze(dx1,dx2,dx3, w_x,P_x,Q,Rm, n_x, alpha,uMax[0], freeze_x)
            uy,  w_y, P_y = ekf_snpid_freeze(dy1,dy2,dy3, w_y,P_y,Q,Rm, n_y, alpha,uMax[1], freeze_y)
            uth, w_t, P_t = ekf_snpid_freeze(dt1,dt2,dt3, w_t,P_t,Q,Rm, n_th,alpha,uMax[2], freeze_t)

            # --- Freeze/unfreeze por normalización ---
            exn = abs(ex); eyn = abs(ey); ethn= abs(eth)
            if (not freeze_x) and (exn < e_thr_freeze): freeze_x=True
            elif freeze_x and (exn > e_thr_unfz):       freeze_x=False
            if (not freeze_y) and (eyn < e_thr_freeze): freeze_y=True
            elif freeze_y and (eyn > e_thr_unfz):       freeze_y=False
            if (not freeze_t) and (ethn < e_thr_freeze): freeze_t=True
            elif freeze_t and (ethn > e_thr_unfz):        freeze_t=False

            # --- Mapeo a ruedas (alpha = theta + pi/4) ---
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

            # --------- EJEMPLO: mover Cuboid después de 10 s ---------
            if t > 10:
                bot.sim.setObjectPosition(cuboid, -1, [1.2, 2.0, 0.05])
                # si es dinámico:
                try: bot.sim.resetDynamicObject(cuboid)
                except: pass

            # Si se terminó la ruta y no hay loop, detener y salir:
            if REF_MODO == "puntos" and wp_finished and not WP_LOOP:
                bot.stop()
                print("[INFO] último waypoint alcanzado. Ruta finalizada (loop=False).")
                break

            # --- Hamiltoniano canónico ---
            H = 0.5*m_tot*(ux**2 + uy**2) + 0.5*Iz*(uth**2)

            # --- Predicción de H con TF (si hay modelo) ---
            H_pred_inst = None
            if model is not None:
                X = np.array([[ux, uy, uth, m_tot]], dtype=np.float32)
                try:
                    Hp = model.predict(X, verbose=0).ravel()[0]
                    H_pred_inst = float(Hp)
                except Exception as e:
                    if len(H_pred_list) == 0:
                        print("[WARN] Falló la predicción con TF en lí­nea. Detalle:", e)
                    H_pred_inst = None

            # --- Logs ---
            t_log.append(t); p_log.append(p); pd_log.append([xd,yd,thd]); w_log.append(w_cmd)
            ux_log.append(ux); uy_log.append(uy); uth_log.append(uth)
            ex_log.append(ex); ey_log.append(ey); eth_log.append(eth)
            exn_log.append(exn); eyn_log.append(eyn); ethn_log.append(ethn)
            Wx_log.append(w_x.copy()); Wy_log.append(w_y.copy()); Wt_log.append(w_t.copy())
            H_log.append(H); H_pred_list.append(H_pred_inst)
            wp_idx_log.append(wp_idx); wp_dist_log.append(wp_dist); wp_eth_log.append(wp_eth)

            # Norma del avance entre pasos
            if p_prev is None:
                dstep = np.nan
            else:
                dstep = np.linalg.norm(p - p_prev)
            dist_log.append(dstep)

            t_prev=t; p_prev=p
            ex_prev,ey_prev,eth_prev = ex,ey,eth
            time.sleep(0.01)

    finally:
        bot.stop(); print("[INFO] Parado.")

    # -------- A arreglos --------
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
    dist_log = np.array(dist_log, float) if dist_log else np.array([], float)
    wp_idx_log = np.array(wp_idx_log, int) if wp_idx_log else np.array([], int)
    wp_dist_log = np.array(wp_dist_log, float) if wp_dist_log else np.array([], float)
    wp_eth_log = np.array(wp_eth_log, float) if wp_eth_log else np.array([], float)

    # Procesar H_pred (puede tener None)
    if len(H_pred_list) == len(t_log):
        H_pred = np.array([np.nan if v is None else float(v) for v in H_pred_list], float)
    else:
        H_pred = None

    # ====== Métricas de H si hay predicción válida ======
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
            print(f"[Comparación H] RMSE={rmse:.6e}  MAE={mae:.6e}  R2={r2:.6f}  "
                  f"NRMSE(range)={rmse/rng:.6e}  NRMSE(std)={rmse/stdH:.6e}")
        else:
            print("[INFO] Muy pocos puntos válidos para mÃ©tricas de H.")

    # -------- Gráficas --------
    plt.figure("Posición vs tiempo", figsize=(8,9))
    for i, lab in enumerate(['x [m]','y [m]', r'$\theta$ [rad]']):
        ax = plt.subplot(4,1,i+1); ax.grid(True)
        if pd_log.shape[1]>0: ax.plot(t_log, pd_log[i], '--', lw=1.5, label=f'{lab}_d')
        if p_log.shape[1]>0:  ax.plot(t_log, p_log[i], lw=1.5, label=lab)
        ax.set_ylabel(lab)
        ax.legend(loc='best')
    ax = plt.subplot(4,1,4); ax.grid(True)
    ax.plot(t_log, wp_idx_log, lw=1.2); ax.set_ylabel('wp_idx'); ax.set_xlabel('t [s]')
    ax.legend(['wp_idx'])

    plt.figure("Trayectoria XY", figsize=(7,7))
    ax = plt.gca(); ax.grid(True); ax.set_aspect('equal', adjustable='box')
    if pd_log.shape[1]>0: ax.plot(pd_log[0], pd_log[1], 'k--', lw=1.5, label='ref/objetivo')
    if p_log.shape[1]>0:
        ax.plot(p_log[0], p_log[1], 'b', lw=1.8, label='real')
        draw_robot_ax(ax, p_log[:,-1], L, l)
    
    if REF_MODO == "puntos":
        W = np.array(WAYPOINTS_POSE).T
        ax.plot(W[0], W[1], 'o-', label='waypoints (con 'r'$\theta$¸)')
        for i,(wx,wy,wth) in enumerate(WAYPOINTS_POSE):
            ax.text(wx, wy, f'{i}', fontsize=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(loc='best'); ax.set_title(f"Seguimiento EKF-SNPID ({REF_MODO})")

    plt.figure("Control (u_x, u_y, u_theta)", figsize=(9,4))
    plt.grid(True); plt.plot(t_log, ux_log, label='u_x')
    plt.plot(t_log, uy_log, label='u_y'); plt.plot(t_log, uth_log, label='u_'r'$\theta$¸')
    plt.xlabel('t [s]'); plt.ylabel('control'); plt.legend(loc='best')

    plt.figure("Errores", figsize=(9,4))
    plt.grid(True); plt.plot(t_log, ex_log, label='e_x')
    plt.plot(t_log, ey_log, label='e_y'); plt.plot(t_log, eth_log, label='e_'r'$\theta$¸')
    plt.xlabel('t [s]'); plt.legend(loc='best')

    plt.figure("Errores normalizados", figsize=(9,4))
    plt.grid(True); plt.plot(t_log, exn_log, label='e_{x,n}')
    plt.plot(t_log, eyn_log, label='e_{y,n}'); plt.plot(t_log, ethn_log, label='e_{'r'$\theta$¸,n}')
    plt.axhline(e_thr_freeze, linestyle='--', label='freeze')
    plt.axhline(e_thr_unfz,   linestyle='--', label='unfreeze')
    plt.xlabel('t [s]'); plt.legend(loc='best')

    plt.figure("Ganancias aprendidas", figsize=(9,9))
    ax=plt.subplot(3,1,1); ax.grid(True)
    if Wx_log.shape[1]>0: ax.plot(t_log, Wx_log[0], t_log, Wx_log[1], t_log, Wx_log[2])
    ax.legend(['Kp_x','Kd_x','Ki_x']); ax.set_title('X')
    ax=plt.subplot(3,1,2); ax.grid(True)
    if Wy_log.shape[1]>0: ax.plot(t_log, Wy_log[0], t_log, Wy_log[1], t_log, Wy_log[2])
    ax.legend(['Kp_y','Kd_y','Ki_y']); ax.set_title('Y')
    ax=plt.subplot(3,1,3); ax.grid(True)
    if Wt_log.shape[1]>0: ax.plot(t_log, Wt_log[0], t_log, Wt_log[1], t_log, Wt_log[2])
    ax.legend([r'Kp_'r'$\theta$¸',r'Kd_'r'$\theta$¸',r'Ki_'r'$\theta$¸']); ax.set_title(''r'$\theta$¸'); ax.set_xlabel('t [s]')

    if w_log.shape[1]>0:
        plt.figure("Velocidades de rueda", figsize=(9,4))
        plt.grid(True)
        for i in range(4): plt.plot(t_log, w_log[i], label=f'v_{i+1}')
        plt.xlabel('t [s]'); plt.ylabel('rad/s'); plt.legend(loc='best')

    # ====== Hamiltoniano real vs predicho ======
    plt.figure("Hamiltoniano", figsize=(10,4))
    plt.grid(True); plt.plot(t_log, H_log, lw=1.2, label='H real (fí­sico)')
    if H_pred is not None and np.isfinite(H_pred).any():
        plt.plot(t_log, H_pred, '--', label='H predicho (TF)')
    plt.xlabel('t [s]'); plt.ylabel('Energí­a [J]')
    plt.title('Hamiltoniano canónico')
    plt.legend(loc='best')

    # ====== Avance thetap ======
    plt.figure("Avance 'r'$\theta$p", figsize=(9,3))
    plt.grid(True)
    plt.plot(t_log, dist_log, lw=1.5)
    plt.xlabel('t [s]')
    plt.ylabel(r'||'r'$\theta$p||')
    plt.title('Norma del avance entre pasos (np.linalg.norm(p - p_prev))')

    plt.show()

if __name__ == "__main__":
    main()
