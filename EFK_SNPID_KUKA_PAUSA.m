% Baseline implementation of the EKF-SNPID controller with anti-windup 
% for an omnidirectional YouBot. 
% Part of the "On-Demand Adaptive PID Control via Neural Networks" framework.

clear; clc; close all;

%% Robot Parameters (YouBot Mecanum)
L  = 0.2355;      % [m] Half-length between front/rear axles
l  = 0.15;        % [m] Half-width between left/right wheels
r  = 0.0475;      % [m] Wheel radius
dt = 0.05;        % [s] Sampling time
Tsim = 80;        % [s] Total simulation time
N = round(Tsim/dt);

%% Reference Trajectory (Rose Curve with Offset)
t_original = (0:N-1)*dt;

% Rose trajectory formulation
a    = 1 + 0.5 * cos(5 * 0.05 * t_original);
xref_original = a .* cos(0.05 * t_original) - 1.5;
yref_original = a .* sin(0.05 * t_original);
thref_original= (pi/4) * ones(1,N);

% Hard Pause Implementation
pause_start = 15;    % [s]
pause_duration = 20; % [s]

idx_start = round(pause_start/dt);
idx_pause = round(pause_duration/dt);

% Capture state at pause initiation
x_hold  = xref_original(idx_start);
y_hold  = yref_original(idx_start);
th_hold = thref_original(idx_start);

% Construct extended trajectory with hold period
xref = [xref_original(1:idx_start), repmat(x_hold, 1, idx_pause), xref_original(idx_start+1:end)];
yref = [yref_original(1:idx_start), repmat(y_hold, 1, idx_pause), yref_original(idx_start+1:end)];
thref= [thref_original(1:idx_start), repmat(th_hold, 1, idx_pause), thref_original(idx_start+1:end)];

% Recompute total time vector
t = (0:length(xref)-1) * dt;
N = length(t);

%% EKF-SNPID Parameters
% EKF Covariances
P_init = eye(3);
Q      = 0.1 * eye(3);
R      = 1e-4;

% Learning rate gains (n vector)
n_x  = [0.1; 0.1; 0.01];
n_y  = [0.1; 0.1; 0.01];
n_th = [0.1; 0.1; 0.01];

alpha = 0.3;        % Tanh scaling factor
uMax  = [0.3; 0.3; 0.3];  % Saturation limits [m/s; m/s; rad/s]

%% State and Controller Initialization
state = zeros(3, N);     % [x; y; theta]
state(:,1) = [0; 0; 0];

% Neural PID Weights (w1->Kp, w2->Ki, w3->Kd) and EKF Matrices
w_x = zeros(3,1); P_x = P_init;
w_y = zeros(3,1); P_y = P_init;
w_t = zeros(3,1); P_t = P_init;

% Historical Data Preallocation
ux  = zeros(1,N); uy  = zeros(1,N); uth = zeros(1,N);
ex  = zeros(1,N); ey  = zeros(1,N); eth = zeros(1,N);
W_x = zeros(3,N); 
W_y = zeros(3,N); 
W_t = zeros(3,N); 

% Discrete Integrals (Real-time accumulators)
dx3 = 0; dy3 = 0; dt3 = 0;

%% Main Simulation Loop
for k = 2:N
    % 1) Current State Measurement
    xk = state(1,k-1);
    yk = state(2,k-1);
    th = state(3,k-1);
    
    % 2) Tracking Error
    ex(k)  = xref(k) - xk;
    ey(k)  = yref(k) - yk;
    eth(k) = wrapToPi(thref(k) - th);
    
    % 3) P, D, I Components (Discrete difference and cumulative sum)
    dx1 = ex(k);               dx2 = ex(k) - ex(k-1);      dx3 = dx3 + ex(k);
    dy1 = ey(k);               dy2 = ey(k) - ey(k-1);      dy3 = dy3 + ey(k);
    dt1 = eth(k);              dt2 = eth(k) - eth(k-1);    dt3 = dt3 + eth(k);
    
    % 4) Neural Controller with EKF and Anti-Windup
    [ux(k),  w_x, P_x] = ekf_snpid(dx1, dx2, dx3, w_x, P_x, Q, R, n_x,  alpha, uMax(1));
    [uy(k),  w_y, P_y] = ekf_snpid(dy1, dy2, dy3, w_y, P_y, Q, R, n_y,  alpha, uMax(2));
    [uth(k), w_t, P_t] = ekf_snpid(dt1, dt2, dt3, w_t, P_t, Q, R, n_th, alpha, uMax(3));
    
    % 5) Dynamics (with block during pause)
    idx_end = idx_start + idx_pause;
    if k >= idx_start && k < idx_end
        state(:,k) = state(:,k-1); % Robot frozen
    else
        state(:,k) = state(:,k-1) + dt * [ux(k); uy(k); uth(k)];
    end
    
    % Weight Logging
    W_x(:,k) = w_x;
    W_y(:,k) = w_y;
    W_t(:,k) = w_t;
end

%% Results Plotting
figure('Name','System Performance','NumberTitle','off', 'Position', [100, 100, 1000, 800]);

subplot(2,2,1)
plot(state(1,:), state(2,:), 'b', xref, yref, '--r'); grid on;
xlabel('x [m]'); ylabel('y [m]');
legend('Real Trajectory', 'Reference', 'Location', 'best')
title('XY Plane')

subplot(2,2,2)
plot(t, ex, t, ey, t, eth);
legend('e_x', 'e_y', 'e_{\theta}'); grid on;
xlabel('Time [s]'); title('Tracking Errors')

subplot(2,2,3)
plot(t, ux, t, uy, t, uth); grid on;
legend('u_x', 'u_y', 'u_{\theta}');
xlabel('Time [s]'); title('Control Signals')

figure('Name','Neural Network Gains Evolution','NumberTitle','off', 'Position', [150, 150, 800, 900]);

subplot(3,1,1)
plot(t, W_x(1,:), 'r', t, W_x(2,:), 'g', t, W_x(3,:), 'b');
legend('Kp_x', 'Ki_x', 'Kd_x'); grid on;
xlabel('Time [s]'); ylabel('Gain'); title('X-Axis Gains');

subplot(3,1,2)
plot(t, W_y(1,:), 'r', t, W_y(2,:), 'g', t, W_y(3,:), 'b');
legend('Kp_y', 'Ki_y', 'Kd_y'); grid on;
xlabel('Time [s]'); ylabel('Gain'); title('Y-Axis Gains');

subplot(3,1,3)
plot(t, W_t(1,:), 'r', t, W_t(2,:), 'g', t, W_t(3,:), 'b');
legend('Kp_{\theta}', 'Ki_{\theta}', 'Kd_{\theta}'); grid on;
xlabel('Time [s]'); ylabel('Gain'); title('\theta Gains');

%% ===== Subroutines =====
function [u_sat, w_new, P_new] = ekf_snpid(x1, x2, x3, w, P, Q, R, n, alpha, umax)
% ekf_snpid: Online trained Single-Neuron PID controller via EKF + anti-windup
% Inputs:
%   x1,x2,x3 : P, D, and I error inputs
%   w, P     : Current weights and EKF covariance (3x1 and 3x3)
%   Q, R     : Process and measurement covariances
%   n        : Learning rate gains vector [n1; n2; n3]
%   alpha    : Tanh scaling factor
%   umax     : Unidimensional saturation limit
% Outputs:
%   u_sat    : Saturated control signal
%   w_new, P_new : Updated weights and EKF covariance

    % 1) Compute weighted sum (v) and raw output (u)
    x = [x1; x2; x3];
    v = w' * x;
    u = alpha * tanh(v);
    
    % 2) Anti-windup back-calculation
    u_sat = max(-umax, min(umax, u));
    es    = u_sat - u;       % Saturation error
    
    if abs(es) > 0
        e_aw = x1 + es;      % Modify innovation if saturated
    else
        e_aw = x1;
    end
    
    % 3) EKF Weight Update
    %    a) Jacobian H = ∂u/∂w (3x1)
    sech2 = (1 - tanh(v)^2);
    H_jac = alpha * sech2 * x;  
    
    %    b) Kalman Gain (3x1)
    S = H_jac' * P * H_jac + R;
    K = (P * H_jac) / S;
    
    %    c) Covariance Update
    P_new = P - K * H_jac' * P + Q;
    
    % 4) Final Weight Adjustment (scaled by learning rates)
    w_new = w + (n .* K) * e_aw;
end