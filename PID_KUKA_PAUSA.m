% Baseline discrete PID controller for an omnidirectional YouBot.
% Incorporates a hard pause (trajectory hold) to evaluate anti-windup 
% and recovery behavior. Provides foundational system validation for 
% the robust adaptive control framework.
%
% Part of the "On-Demand Adaptive PID Control via Neural Networks" repository.

clear; clc; close all;

%% Robot Parameters (YouBot Mecanum)
L    = 0.2355;    % [m] Half-length
l    = 0.15;      % [m] Half-width
r    = 0.0475;    % [m] Wheel radius
dt   = 0.05;      % [s] Sampling time
Tsim = 130;       % [s] Total simulation time
N    = round(Tsim/dt);

%% Reference Trajectory (Sinusoidal with Hold Period)
t_original = (0:N-1) * dt;

% Base Sinusoidal Trajectory
xref_original  = 0.1 * t_original;
yref_original  = 0.2 * sin(0.5 * t_original);
thref_original = -(pi/8) * ones(1, N);

% Hard Pause Implementation
pause_start    = 15;   % [s]
pause_duration = 50;   % [s]

idx_start = round(pause_start/dt);
idx_pause = round(pause_duration/dt);
idx_end   = idx_start + idx_pause;

% Capture state at pause initiation
x_hold  = xref_original(idx_start);
y_hold  = yref_original(idx_start);
th_hold = thref_original(idx_start);

% Construct extended trajectory with hold period
xref  = [xref_original(1:idx_start), repmat(x_hold, 1, idx_pause), xref_original(idx_start+1:end)];
yref  = [yref_original(1:idx_start), repmat(y_hold, 1, idx_pause), yref_original(idx_start+1:end)];
thref = [thref_original(1:idx_start), repmat(th_hold, 1, idx_pause), thref_original(idx_start+1:end)];

% Recompute total time vector
t = (0:length(xref)-1) * dt;
N = length(t);

%% Baseline PID Gains
Kp = [0.3; 0.3; 0.2];   % [Kp_x; Kp_y; Kp_th]
Ki = [0.2; 0.2; 0.1];   % [Ki_x; Ki_y; Ki_th]
Kd = [0.1; 0.1; 0.05];  % [Kd_x; Kd_y; Kd_th]

%% State and Error Initialization
state = zeros(3, N);    % [x; y; theta]
state(:,1) = [0; 0; 0];

ex = zeros(1,N); ey = zeros(1,N); eth = zeros(1,N);
sum_ex = 0; sum_ey = 0; sum_eth = 0;

ux = zeros(1,N); uy = zeros(1,N); uth = zeros(1,N);
p_step = zeros(1,N); % Step magnitude array

%% Main Control and Simulation Loop
for k = 2:N
    % --- Current Pose Measurement
    xk = state(1,k-1);
    yk = state(2,k-1);
    th = state(3,k-1);
    
    % --- Tracking Errors
    ex(k)  = xref(k) - xk;
    ey(k)  = yref(k) - yk;
    eth(k) = wrapToPi_local(thref(k) - th);
    
    % --- Step Magnitude Calculation
    % Evaluates the movement from the previous iteration
    p_step(k) = norm(state(:,k-1) - state(:,max(1, k-2))); 
    
    % --- Error Integration
    sum_ex  = sum_ex  + ex(k);
    sum_ey  = sum_ey  + ey(k);
    sum_eth = sum_eth + eth(k);
    
    % --- Error Derivation
    de_x  = ex(k)  - ex(k-1);
    de_y  = ey(k)  - ey(k-1);
    de_th = eth(k) - eth(k-1);
    
    % --- Discrete PID Control Law
    ux(k)  = Kp(1)*ex(k)  + Ki(1)*sum_ex  + Kd(1)*de_x;
    uy(k)  = Kp(2)*ey(k)  + Ki(2)*sum_ey  + Kd(2)*de_y;
    uth(k) = Kp(3)*eth(k) + Ki(3)*sum_eth + Kd(3)*de_th;
    
    % --- Anti-Windup: Control Saturation and Integral Correction
    sat_limit = 0.3;
    
    if abs(ux(k)) > sat_limit
        ux(k) = sign(ux(k)) * sat_limit;
        sum_ex = sum_ex - ex(k);  
    end
    
    if abs(uy(k)) > sat_limit
        uy(k) = sign(uy(k)) * sat_limit;
        sum_ey = sum_ey - ey(k);  
    end
    
    if abs(uth(k)) > sat_limit
        uth(k) = sign(uth(k)) * sat_limit;
        sum_eth = sum_eth - eth(k);  
    end
    
    % --- Robot Kinematics Integration (Explicit Euler with Hold)
    if k >= idx_start && k < idx_end
        state(:,k) = state(:,k-1); % Robot frozen
    else
        state(:,k) = state(:,k-1) + dt * [ux(k); uy(k); uth(k)];
    end
end

%% Results Visualization
figure('Name','Conventional PID with Hold - YouBot','NumberTitle','off', 'Position', [100, 100, 1000, 800]);

subplot(2,2,1)
plot(state(1,:), state(2,:), 'b', xref, yref, '--r');
grid on;
xlabel('x [m]'); ylabel('y [m]');
legend('Real Trajectory', 'Reference', 'Location', 'best')
title('XY Plane Trajectory')

subplot(2,2,2)
plot(t, ex, t, ey, t, eth, 'LineWidth', 1);
grid on;
legend('e_x', 'e_y', 'e_{\theta}', 'Location', 'best')
xlabel('Time [s]'); title('Tracking Errors')

subplot(2,2,3)
plot(t, ux, t, uy, t, uth, 'LineWidth', 1);
grid on;
legend('u_x', 'u_y', 'u_{\theta}', 'Location', 'best')
xlabel('Time [s]'); title('PID Control Signals')

subplot(2,2,4)
plot(t, p_step, 'LineWidth', 1);
grid on;
xlabel('Time [s]'); ylabel('||\Delta p||');
title('Pose Step Magnitude')

%% ===== Subroutines =====
function ang = wrapToPi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end