% Baseline discrete PID controller for an omnidirectional YouBot.
% Provides foundational system validation and parameter benchmarking 
% for the robust adaptive control framework.
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

%% Reference Trajectory (Sinusoidal)
t     = (0:N-1) * dt;
xref  = 0.1 * t;
yref  = 0.2 * sin(0.5 * t);
thref = -(pi/8) * ones(1, N);

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
    
    % --- Robot Kinematics Integration (Explicit Euler)
    state(:,k) = state(:,k-1) + dt * [ux(k); uy(k); uth(k)];
end

%% Results Visualization
figure('Name','Conventional PID - YouBot','NumberTitle','off', 'Position', [100, 100, 1200, 400]);

subplot(1,3,1)
plot(state(1,:), state(2,:), 'b', xref, yref, '--r');
grid on;
xlabel('x [m]'); ylabel('y [m]');
legend('Real Trajectory', 'Reference', 'Location', 'best')
title('XY Plane Trajectory')

subplot(1,3,2)
plot(t, ex, t, ey, t, eth, 'LineWidth', 1);
grid on;
legend('e_x', 'e_y', 'e_{\theta}', 'Location', 'best')
xlabel('Time [s]'); title('Tracking Errors')

subplot(1,3,3)
plot(t, ux, t, uy, t, uth, 'LineWidth', 1);
grid on;
legend('u_x', 'u_y', 'u_{\theta}', 'Location', 'best')
xlabel('Time [s]'); title('PID Control Signals')

%% ===== Subroutines =====
function ang = wrapToPi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end