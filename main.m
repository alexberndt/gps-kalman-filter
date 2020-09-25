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
close

load("data/GPSdata.mat");
currentFolder = pwd;

%% Extended Kalman Filter

s2r = ref_data_struct.s2r;
est = ExtendedKalmanFilter(gps_data, ref_data_struct);

x_h = est.x_h;
P = est.P;

% obtain coordinate values 
x           = x_h(1,:);
y           = x_h(3,:);
z           = x_h(5,:);
delta_T     = x_h(6,:);
delta_Tdot  = x_h(7,:);

% true trajectory
x_t = ref_data_struct.traj_ned(1,:);
y_t = ref_data_struct.traj_ned(2,:);
z_t = ref_data_struct.traj_ned(3,:);

% plot x-y 2D
gcf3 = figure(3);
clf;
plot(x,y);
grid on
xlabel("x [m]");
ylabel("y [m]");
title("EKF - 2D");
hold on
plot(x_t,y_t,'r-.');
legend("EKF","true");
saveas(gcf3, "./plots/ekf_traj_2D.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/ekf_traj_2D.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

plot_extra = true;

if plot_extra

    % plot x-y-z 3D
    gcf4 = figure(4);
    clf;
    plot3(x,y,z);
    grid on
    xlabel("x [m]");
    ylabel("y [m]");
    zlabel("z [m]");
    title("EKF - 3D");
    hold on
    plot3(x_t,y_t,z_t,'r-.');
    zlim([-100 100]);
    saveas(gcf4, "./plots/ekf_traj_3D.eps");
    tikzfilename = strcat(currentFolder,'/tikzfiles/ekf_traj_3D.tex');
    cleanfigure; 
    matlab2tikz('filename',tikzfilename);

    % plot errors
    t_range = 500; % plot only time-steps 0 to 500 

    gcf5 = figure(5);
    clf;

    subplot(3,1,1);
    stairs(x(1:t_range));
    hold on
    grid on
    plot(x_t(1:t_range));
    xlabel("k");
    ylabel("x [m]");
    legend("EKF","true");

    subplot(3,1,2);
    stairs(y(1:t_range));
    hold on
    grid on
    plot(y_t(1:t_range));
    xlabel("k");
    ylabel("y [m]");
    legend("EKF","true");

    subplot(3,1,3);
    stairs(z(1:t_range));
    hold on
    grid on
    plot(z_t(1:t_range));
    xlabel("k");
    ylabel("z [m]");
    legend("EKF","true");

    %% Plot the covariance matrix entries

    gcf6 = figure(6);
    clf;

    subplot(3,1,1);
    stairs(reshape(est.P(1,1,1:t_range),[],1));
    grid on
    xlabel("k");
    ylabel("P(1,1) - x");

    subplot(3,1,2);
    stairs(reshape(est.P(3,3,1:t_range),[],1));
    grid on
    xlabel("k");
    ylabel("P(3,3) - y");

    subplot(3,1,3);
    stairs(reshape(est.P(5,5,1:t_range),[],1));
    grid on
    xlabel("k");
    ylabel("P(5,5) - z");

    % subplot(3,1,2);
    % stairs(y(1:t_range));
    % hold on
    % grid on
    % plot(y_t(1:t_range));
    % xlabel("k");
    % ylabel("y [m]");
    % legend("EKF","true");
    % 
    % subplot(3,1,3);
    % stairs(z(1:t_range));
    % hold on
    % grid on
    % plot(z_t(1:t_range));
    % xlabel("k");
    % ylabel("z [m]");
    % legend("EKF","true");
end



%% Part III

























































%.

