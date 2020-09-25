%% FEM3200 - Optimal Filtering 

% Project: Kalman Filter design for GPS receiver
%
% Authors: Alexander Berndt, Rebecka Winqvist
%
% Date: 22 September 2020

%% File Structure

% The following files are here (./main.m):

%   ./data/GPSdata.mat
%   ./NonLinearLeastSquares.m

% This script also uses Matlab2Tikz 
%   - make sure matlab2tikz/src is in your MATLAB path

% %% Initialisation
% 
% clc
% clear
% 
% load("data/GPSdata.mat");
% currentFolder = pwd;
% 
% %% Nonlinear Least Squares Algorithm
% 
% % estimate car trajectory using NL least-squares algorithm
% est = NonLinearLeastSquares(gps_data, ref_data_struct.s2r);
% 
% x_h = est.x_h;
% P = est.P;
% 
% % obtain coordinate values 
% x = x_h(1,:);
% y = x_h(2,:);
% z = x_h(3,:);
% a = x_h(4,:);
% 
% % plot x-y 2D
% gcf1 = figure(1);
% plot(x,y);
% grid on
% xlabel("x [m]");
% ylabel("y [m]");
% title("NL Estimator");
% saveas(gcf1, "./plots/nl_estimator_traj_2D.eps");
% tikzfilename = strcat(currentFolder,'/tikzfiles/nl_estimator_traj_2D.tex');
% cleanfigure; 
% matlab2tikz('filename',tikzfilename);
% 
% % plot x-y-z 3D
% gcf2 = figure(2);
% plot3(x,y,z);
% grid on
% xlabel("x [m]");
% ylabel("y [m]");
% zlabel("z [m]");
% title("NL Estimator");
% zlim([-100 100]);
% saveas(gcf2, "./plots/nl_estimator_traj_3D.eps");
% tikzfilename = strcat(currentFolder,'/tikzfiles/nl_estimator_traj_3D.tex');
% cleanfigure; 
% matlab2tikz('filename',tikzfilename);

%% Initialisation

clc
clear
% close

load("data/GPSdata.mat");
currentFolder = pwd;

%% Extended Kalman Filter

% run the EKF
est = ExtendedKalmanFilter(gps_data, ref_data_struct);

%% Get estimated and true values for plotting

% extract estimation states and covariances
x_h         = est.x_h;
P           = est.P;
e_k         = est.e_k; 

% obtain coordinate values 
x            = x_h(1,:);
y            = x_h(3,:);
z            = x_h(5,:);
delta_T      = x_h(7,:);
delta_Tdot   = x_h(8,:);

% true trajectory
x_t          = ref_data_struct.traj_ned(1,:);
y_t          = ref_data_struct.traj_ned(2,:);
z_t          = ref_data_struct.traj_ned(3,:);
delta_T_t    = ref_data_struct.x_clk(1,:);
delta_Tdot_t = ref_data_struct.x_clk(2,:);

% time vector
t   =   ref_data_struct.Ts.*(0:1:length(x));

%% Plotting

% plot errors between a range 480 to 2000
t_min = 480;
t_max = 2000;
t_vec = t(t_min:t_max);

gcf5 = figure(5);
clf;

subplot(5,1,1);
stairs(t_vec,x(t_min:t_max));
hold on
grid on
plot(t_vec,x_t(t_min:t_max));
ylabel("x [m]");
legend("EKF","true");

subplot(5,1,2);
stairs(t_vec,y(t_min:t_max));
hold on
grid on
plot(t_vec,y_t(t_min:t_max));
ylabel("y [m]");
legend("EKF","true");

subplot(5,1,3);
stairs(t_vec,z(t_min:t_max));
hold on
grid on
plot(t_vec,z_t(t_min:t_max));
ylabel("z [m]");
legend("EKF","true");

subplot(5,1,4);
stairs(t_vec,delta_T(t_min:t_max));
hold on
grid on
plot(t_vec,delta_T_t(t_min:t_max));
ylabel("$\Delta T$ [m]",'Interpreter','Latex');
legend("EKF","true");

subplot(5,1,5);
stairs(t_vec,delta_Tdot(t_min:t_max));
hold on
grid on
plot(t_vec,delta_Tdot_t(t_min:t_max));
ylabel("$\dot{\Delta} T$ [m]",'Interpreter','Latex');
xlabel("t(k)");
legend("EKF","true");

%% Plot the covariance matrix entries

t_min = 1;
t_max = 2000;
t_vec = t(t_min:t_max);

gcf6 = figure(6);
clf;

subplot(5,1,1);
stairs(t_vec,reshape(est.P(1,1,t_min:t_max),[],1));
grid on
ylabel("P(1,1) - x");

subplot(5,1,2);
stairs(t_vec,reshape(est.P(3,3,t_min:t_max),[],1));
grid on
ylabel("P(3,3) - y");

subplot(5,1,3);
stairs(t_vec,reshape(est.P(5,5,t_min:t_max),[],1));
grid on
ylabel("P(5,5) - z");

subplot(5,1,4);
stairs(t_vec,reshape(est.P(7,7,t_min:t_max),[],1));
grid on
ylabel("P(7,7) - Delta T");

subplot(5,1,5);
stairs(t_vec,reshape(est.P(8,8,t_min:t_max),[],1));
grid on
xlabel("t(k)");
ylabel("P(8,8) - Delta T dot");

%% Innovation for specific satellite

sat_number = 11;

t_min = 400;
t_max = 2000;
t_vec = t(t_min:t_max);

gcf7 = figure(7);
clf;

plot(t_vec, e_k(sat_number,t_min:t_max));
grid on
xlabel("t(k)");
ylabel("innovation e_k(k)");

%% 2D Plot

t_min = 500;
t_max = 800;

gcf3 = figure(3);
clf;
plot(x(t_min:t_max),y(t_min:t_max));
grid on
xlabel("x [m]");
ylabel("y [m]");
title("EKF - 2D");
hold on
plot(x_t(t_min:t_max),y_t(t_min:t_max),'r-.');
legend("EKF","true");
saveas(gcf3, "./plots/ekf_traj_2D.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/ekf_traj_2D.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% 3D Plot

t_min = 500;
t_max = 800;

gcf4 = figure(4);
clf;
plot3(x(t_min:t_max),y(t_min:t_max),z(t_min:t_max));
grid on
xlabel("x [m]");
ylabel("y [m]");
zlabel("z [m]");
title("EKF - 3D");
hold on
plot3(x_t(t_min:t_max),y_t(t_min:t_max),z_t(t_min:t_max),'r-.');
zlim([-35 35]);
saveas(gcf4, "./plots/ekf_traj_3D.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/ekf_traj_3D.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% Part III

























































%.

