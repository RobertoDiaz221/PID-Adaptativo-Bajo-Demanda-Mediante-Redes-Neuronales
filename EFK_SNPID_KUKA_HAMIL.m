% Evaluates the overall robustness of the proposed EKF-SNPID system 
% utilizing a pure error normalization hysteresis mechanism.
% Freezing occurs if the normalized error falls below a minimum threshold, 
% and unfreezes when it exceeds an upper threshold.
%
% Part of the "On-Demand Adaptive PID Control via Neural Networks" framework.

clear; clc; close all;

%% Robot Parameters (YouBot Mecanum)
L = 0.2355;
l = 0.15; 
r = 0.0475;
dt = 0.05;    
Tsim = 280;
N  = round(Tsim/dt);

% Mass Parameters
m_base = 16;          % Chassis mass [kg]
m_w    = 2;           % Wheel mass [kg per wheel]
m_L    = 0;           % Centered payload mass [kg]
a_box  = 0.30;        % Box length [m] (if applicable)
b_box  = 0.25;        % Box width [m] (if applicable)

%% Reference Trajectory (Sinusoidal) + Hard Pause
t_original = (0:N-1)*dt;
xref_original  = 0.2 * t_original;
yref_original  = 0.4 * sin(0.2*t_original);
thref_original = -pi/8 * ones(1,N);

pause_start    = 35;  % [s]
pause_duration = 0;   % [s]
idx_start = round(pause_start/dt);
idx_pause = round(pause_duration/dt);
idx_end   = idx_start + idx_pause;

x_hold  = xref_original(idx_start);
y_hold  = yref_original(idx_start);
th_hold = thref_original(idx_start);

xref = [xref_original(1:idx_start), repmat(x_hold,1,idx_pause), xref_original(idx_start+1:end)];
yref = [yref_original(1:idx_start), repmat(y_hold,1,idx_pause), yref_original(idx_start+1:end)];
thref= [thref_original(1:idx_start), repmat(th_hold,1,idx_pause), thref_original(idx_start+1:end)];

t = (0:length(xref)-1)*dt;
N = length(t);

%% EKF-SNPID Initialization
P_init = eye(3);
Q      = 0.1 * eye(3);
R      = 1e-4;

n_x  = [0.1; 0.1; 0.01];
n_y  = [0.1; 0.1; 0.01];
n_th = [0.1; 0.1; 0.01];

alpha = 1.5;
uMax  = [1.5; 1.5; 1.5];

%% Pure Normalization Thresholds (Windowless)
e_norm_thr_freeze = 0.05;   % Freeze if e_norm < 5%
e_norm_thr_unfz   = 0.1;    % Unfreeze if e_norm > 10%

%% State and History Allocation
state = zeros(3,N); state(:,1) = [0;0;0];
ux = zeros(1,N); uy = zeros(1,N); uth = zeros(1,N);
ex = zeros(1,N); ey = zeros(1,N); eth = zeros(1,N);
W_x = zeros(3,N); W_y = zeros(3,N); W_t = zeros(3,N);
pp = zeros(1,N); p_step = 0;

% Normalized errors (for plotting/debugging)
exn = zeros(1,N); eyn = zeros(1,N); ethn = zeros(1,N);

% Weight Initialization
w_x = [0; 0; 0];
w_y = [0; 0; 0];
w_t = [0; 0; 0];

P_x = P_init; W_x(:,1) = w_x;
P_y = P_init; W_y(:,1) = w_y;
P_t = P_init; W_t(:,1) = w_t;

% Protected Discrete Integrals
dx3 = 0; dy3 = 0; dt3 = 0;

% Axis Freeze Flags
freeze_x=false; freeze_y=false; freeze_t=false;

%% Main Simulation Loop
for k = 2:N
    xk = state(1,k-1); yk = state(2,k-1); th = state(3,k-1);
    
    ex(k)  = xref(k) - xk;
    ey(k)  = yref(k) - yk;
    eth(k) = wrapToPi_local(thref(k) - th);
    
    % P, D, I Components
    dx1 = ex(k);               dx2 = ex(k) - ex(k-1);
    dy1 = ey(k);               dy2 = ey(k) - ey(k-1);
    dt1 = eth(k);              dt2 = eth(k) - eth(k-1);
    
    % Integral Calculation (prevents accumulation without progression)
    if k >= 3
        p_step = norm(state(:,k-1) - state(:,k-2));
        if p_step > 1e-12
            dx3 = dx3 + ex(k);
            dy3 = dy3 + ey(k);
            dt3 = dt3 + eth(k);
        end
    else
        dx3 = dx3 + ex(k);
        dy3 = dy3 + ey(k);
        dt3 = dt3 + eth(k);
    end
    
    % Control Law + EKF Update (with axis freezing)
    [ux(k), w_x, P_x] = ekf_snpid_freeze(dx1, dx2, dx3, w_x, P_x, Q, R, n_x, alpha, uMax(1), freeze_x);
    [uy(k), w_y, P_y] = ekf_snpid_freeze(dy1, dy2, dy3, w_y, P_y, Q, R, n_y, alpha, uMax(2), freeze_y);
    [uth(k),w_t, P_t] = ekf_snpid_freeze(dt1, dt2, dt3, w_t, P_t, Q, R, n_th, alpha, uMax(3), freeze_t);
    
    % Dynamics (incorporating hard pause block)
    if k >= idx_start && k < idx_end
        state(:,k) = state(:,k-1);
        m_L = 1;
    else
        state(:,k) = state(:,k-1) + dt * [ux(k); uy(k); uth(k)];
    end
    
    % Weight Logging
    W_x(:,k) = w_x; W_y(:,k) = w_y; W_t(:,k) = w_t;
    pp(:, k) = p_step;
    
    % ===== Simple Hysteresis Freezing =====
    exn(k)  = abs(ex(k));
    eyn(k)  = abs(ey(k));
    ethn(k) = abs(eth(k));
    
    % X-Axis
    if ~freeze_x && (exn(k) < e_norm_thr_freeze)
        freeze_x = true;
    elseif freeze_x && (exn(k) > e_norm_thr_unfz)
        freeze_x = false;
    end
    
    % Y-Axis
    if ~freeze_y && (eyn(k) < e_norm_thr_freeze)
        freeze_y = true;
    elseif freeze_y && (eyn(k) > e_norm_thr_unfz)
        freeze_y = false;
    end
    
    % Theta-Axis
    if ~freeze_t && (ethn(k) < e_norm_thr_freeze)
        freeze_t = true;
    elseif freeze_t && (ethn(k) > e_norm_thr_unfz)
        freeze_t = false;
    end
end

%% ===== Canonical Hamiltonian Computation =====
m_tot = m_base + 4*m_w + m_L;
Iz_chasis = (1/3) * m_base * (L^2 + l^2);                     
Iz_rueda  = m_w * (L^2 + l^2) + 0.25 * m_w * r^2;             
Iz        = Iz_chasis + 4 * Iz_rueda;

vx = ux; vy = uy; w = uth;                                    
H  = 0.5 * m_tot .* (vx.^2 + vy.^2) + 0.5 * Iz .* (w.^2);     

%% Plots
figure('Name','XY Plane','NumberTitle','off');
plot(state(1,:), state(2,:), 'b', xref, yref, '--r'); grid on;
xlabel('x [m]'); ylabel('y [m]'); legend('Real','Reference','Location','best'); title('XY Trajectory');

figure('Name','Errors','NumberTitle','off');
plot(t, ex, t, ey, t, eth); grid on; 
legend('e_x','e_y','e_{\theta}'); xlabel('t [s]'); title('Tracking Errors');

figure('Name','Control Signals','NumberTitle','off');
plot(t, ux, t, uy, t, uth); grid on; 
legend('u_x','u_y','u_{\theta}'); xlabel('t [s]'); title('Control Action');

figure('Name','Normalized Errors','NumberTitle','off');
plot(t, exn, t, eyn, t, ethn); grid on; hold on;
yline(e_norm_thr_freeze,'--k','Freeze Threshold'); yline(e_norm_thr_unfz,'--r','Unfreeze Threshold');
legend('e_{x,n}','e_{y,n}','e_{\theta,n}','Location','best');
xlabel('t [s]'); title('Normalized Errors (Hysteresis Bounds)');

figure('Name','Learned Gains','NumberTitle','off');
subplot(3,1,1); plot(t,W_x(1,:), t,W_x(2,:), t,W_x(3,:)); grid on; legend('Kp_x','Ki_x','Kd_x'); title('X-Axis');
subplot(3,1,2); plot(t,W_y(1,:), t,W_y(2,:), t,W_y(3,:)); grid on; legend('Kp_y','Ki_y','Kd_y'); title('Y-Axis');
subplot(3,1,3); plot(t,W_t(1,:), t,W_t(2,:), t,W_t(3,:)); grid on; legend('Kp_{\theta}','Ki_{\theta}','Kd_{\theta}'); title('\theta');

figure('Name','Canonical Hamiltonian','NumberTitle','off');
plot(t, H, 'LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('Energy [J]');
title('Canonical Hamiltonian (m_{base}=16 kg, m_{wheel}=2 kg c/u)');

figure('Name','Pose Change','NumberTitle','off');
plot(t,pp); grid on;
xlabel('t [s]'); ylabel('||\Delta p||');
title('Pose Step Magnitude');

%% ===== Subroutines =====
function [u_sat, w_new, P_new] = ekf_snpid_freeze(x1, x2, x3, w, P, Q, R, n, alpha, umax, freeze_flag)
    x = [x1; x2; x3];
    v = w' * x;
    u = alpha * tanh(v);
    
    % Simple back-calculation anti-windup
    u_sat = max(-umax, min(umax, u));
    es    = u_sat - u;
    e_aw  = x1 + es;  
    
    % Jacobian and innovation covariance
    sech2 = (1 - tanh(v)^2);
    H_jac = alpha * sech2 * x;
    S = H_jac' * P * H_jac + R;
    
    if freeze_flag
        w_new  = w; P_new = P; return;
    end
    
    % EKF Update
    K     = (P * H_jac) / S;
    P_new = P - K * H_jac' * P + Q;
    w_new = w + (n .* K) * e_aw;
end

function ang = wrapToPi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end