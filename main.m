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

%% Initialisation

clc
clear

load("data/GPSdata.mat");
currentFolder = pwd;

% true trajectory
x_t          = ref_data_struct.traj_ned(1,:);
y_t          = ref_data_struct.traj_ned(2,:);
z_t          = ref_data_struct.traj_ned(3,:);
delta_T_t    = ref_data_struct.x_clk(1,:);
delta_Tdot_t = ref_data_struct.x_clk(2,:);

% time vector
t   =   ref_data_struct.Ts.*(0:1:length(x_t));

%% Nonlinear Least Squares Algorithm

% estimate car trajectory using NL least-squares algorithm
est_NL = NonLinearLeastSquares(gps_data, ref_data_struct.s2r);

x_h_NL = est_NL.x_h;
P_NL = est_NL.P;

% obtain coordinate values 
x_NL = x_h_NL(1,:);
y_NL = x_h_NL(2,:);
z_NL = x_h_NL(3,:);
a_NL = x_h_NL(4,:);

% plot x-y 2D
gcf1 = figure(1);
plot(x_NL,y_NL);
grid on
xlabel("x [m]");
ylabel("y [m]");
title("NL Estimator");
saveas(gcf1, "./plots/nl_estimator_traj_2D.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/nl_estimator_traj_2D.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

% plot x-y-z 3D
gcf2 = figure(2);
plot3(x_NL,y_NL,z_NL);
grid on
xlabel("x [m]");
ylabel("y [m]");
zlabel("z [m]");
title("NL Estimator");
zlim([-100 100]);
saveas(gcf2, "./plots/nl_estimator_traj_3D.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/nl_estimator_traj_3D.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% Show to number of satellites with time

% sat_mat = zeros(length(gps_data),2015);
% for satid = 1:length(gps_data)
%     sat_mat(satid,:) = gps_data(satid).PseudoRange;
% end
% 
% figure(3);
% % infimgdata = repmat(infRGB, size(sat_mat,1), size(sat_mat,2)); 
% % infimg = image(infimgdata, 'alphadata', ~isnan(sat_mat));  
% hold on
% test_image = imagesc(sat_mat,'alphadata', ~(isnan(sat_mat)|isinf(sat_mat)));
% colormap(jet());
% colorbar
% hold off
% ylabel("Satellite ID");
% xlabel("time [k]");

%% Extended Kalman Filter

% run the EKF
est         = ExtendedKalmanFilter(gps_data, ref_data_struct);

% run EKF with no measurements at timesteps 1000 to 1010
est_nomeas  = ExtendedKalmanFilterNoMeas(gps_data, ref_data_struct, 1000,1010);

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

% satellite measurements available
sat_count   = est.sat_count;

%% Plot to covariance when measurements stop

% plot errors between a range 480 to 2000
t_min = 400;
t_max = 2015;
t_vec = t(t_min:t_max);



P_nom = est_nomeas.P;

gcf11 = figure(11);
clf;
sigma_kf_x      = sqrt(reshape(P(1,1,t_min:t_max),1,[]));
sigma_kf_x_no   = sqrt(reshape(P_nom(1,1,t_min:t_max),1,[]));
e_x             = x(t_min:t_max)-x_t(t_min:t_max);
x_P_max         = 3*sigma_kf_x;
x_P_min         = -3*sigma_kf_x;
x_P_nom_max     = 3*sigma_kf_x_no;
x_P_nom_min     = -3*sigma_kf_x_no;

% plot(t_vec,e_NL_x);
hold on
grid on
stairs(t_vec,e_x);
stairs(t_vec,x_P_max,'r-');
stairs(t_vec,x_P_min,'r-');
stairs(t_vec,x_P_nom_max,'k-');
stairs(t_vec,x_P_nom_min,'k-');
% plot(t_vec,x_P_NL_max,'m--');
% plot(t_vec,x_P_NL_min,'m--');
ylabel("x [m]",'Interpreter','Latex');

%% iii) Plot the estimation errors

gcf4 = figure(4);
clf;

subplot(5,1,1);
% determine 3 sigma bound 
sigma_kf_x      = sqrt(reshape(P(1,1,t_min:t_max),1,[]));
e_x             = x(t_min:t_max)-x_t(t_min:t_max);
x_P_max         = 3*sigma_kf_x;
x_P_min         = -3*sigma_kf_x;
sigma_NL_x      = sqrt(reshape(P_NL(1,t_min:t_max),1,[]));
e_NL_x          = x_NL(t_min:t_max)-x_t(t_min:t_max);
x_P_NL_max      = 3*sigma_NL_x;
x_P_NL_min      = -3*sigma_NL_x;

% plot(t_vec,e_NL_x);
hold on
grid on
stairs(t_vec,e_x);
stairs(t_vec,x_P_max,'r-');
stairs(t_vec,x_P_min,'r-');
stairs(t_vec,x_P_NL_max,'k--');
stairs(t_vec,x_P_NL_min,'k--');
ylabel("x [m]");

subplot(5,1,2);
% determine 3 sigma bound 
sigma_kf_y     = sqrt(reshape(P(3,3,t_min:t_max),1,[]));
e_y         = y(t_min:t_max)-y_t(t_min:t_max);
y_P_max     = 3*sigma_kf_y;
y_P_min     = -3*sigma_kf_y;
sigma_NL_y      = sqrt(reshape(P_NL(2,t_min:t_max),1,[]));
e_NL_y          = y_NL(t_min:t_max)-y_t(t_min:t_max);
y_P_NL_max      = 3*sigma_NL_y;
y_P_NL_min      = -3*sigma_NL_y;

stairs(t_vec,e_y);
hold on
grid on
stairs(t_vec,y_P_max,'-r');
stairs(t_vec,y_P_min,'-r');
stairs(t_vec,y_P_NL_max,'k--');
stairs(t_vec,y_P_NL_min,'k--');
ylabel("y [m]");

subplot(5,1,3);
% determine 3 sigma bound 
sigma_kf_z     = sqrt(reshape(P(5,5,t_min:t_max),1,[]));
e_z         = z(t_min:t_max)-z_t(t_min:t_max);
z_P_max     = 3*sigma_kf_z;
z_P_min     = -3*sigma_kf_z;
sigma_NL_z      = sqrt(reshape(P_NL(3,t_min:t_max),1,[]));
e_NL_z          = z_NL(t_min:t_max)-z_t(t_min:t_max);
z_P_NL_max      = 3*sigma_NL_z;
z_P_NL_min      = -3*sigma_NL_z;

stairs(t_vec,e_z);
hold on
grid on
stairs(t_vec,z_P_max,'-r');
stairs(t_vec,z_P_min,'-r');
stairs(t_vec,z_P_NL_max,'k--');
stairs(t_vec,z_P_NL_min,'k--');
legend("innovation","+3 EKF","-3 EKF","+3 NL","-3 NL");
ylabel("z [m]");

subplot(5,1,4);
sigma_kf_t     = sqrt(reshape(P(7,7,t_min:t_max),1,[]));
e_t         = delta_T(t_min:t_max)-delta_T_t(t_min:t_max);
t_P_max     = 3*sigma_kf_t;
t_P_min     = -3*sigma_kf_t;
stairs(t_vec,e_t);
hold on
grid on
stairs(t_vec,t_P_max,'-r');
stairs(t_vec,t_P_min,'-r');
ylabel("Delta T [s]");

subplot(5,1,5);
sigma_kf_tdot     = sqrt(reshape(P(8,8,t_min:t_max),1,[]));
e_tdot         = delta_Tdot(t_min:t_max)-delta_Tdot_t(t_min:t_max);
tdot_P_max     = 3*sigma_kf_tdot;
tdot_P_min     = -3*sigma_kf_tdot;
stairs(t_vec,e_tdot);
hold on
grid on
stairs(t_vec,tdot_P_max,'-r');
stairs(t_vec,tdot_P_min,'-r');
ylabel("Delta T dot [s]");

saveas(gcf4, "./plots/gcf4.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/gcf4.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% Plot the trajectories

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
ylabel("$\Delta T$ [s]",'Interpreter','Latex');
legend("EKF","true");

subplot(5,1,5);
stairs(t_vec,delta_Tdot(t_min:t_max));
hold on
grid on
plot(t_vec,delta_Tdot_t(t_min:t_max));
ylabel("$\dot{\Delta} T$ [s]",'Interpreter','Latex');
xlabel("t(k)");
legend("EKF","true");

saveas(gcf5, "./plots/gcf5.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/gcf5.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% Plot the covariance matrix entries

t_min = 1;
t_max = 2000;
t_vec = t(t_min:t_max);

gcf6 = figure(6);
clf;

subplot(5,1,1);
stairs(t_vec,reshape(est.P(1,1,t_min:t_max),[],1));
grid on
hold on
stairs(t_vec,reshape(P_NL(1,t_min:t_max),[],1));
ylabel("P(1,1) - x");

subplot(5,1,2);
stairs(t_vec,reshape(est.P(3,3,t_min:t_max),[],1));
grid on
hold on
stairs(t_vec,reshape(P_NL(2,t_min:t_max),[],1));
ylabel("P(3,3) - y");

subplot(5,1,3);
stairs(t_vec,reshape(est.P(5,5,t_min:t_max),[],1));
grid on
hold on
stairs(t_vec,reshape(P_NL(3,t_min:t_max),[],1));
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

saveas(gcf6, "./plots/gcf6.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/gcf6.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% 2D Plot
 
t_min = 500;
t_max = 800;

gcf7 = figure(7);
clf;
plot(x(t_min:t_max),y(t_min:t_max));
grid on
xlabel("x [m]");
ylabel("y [m]");
title("EKF - 2D");
hold on
plot(x_t(t_min:t_max),y_t(t_min:t_max),'r-.');
legend("EKF","true");
saveas(gcf7, "./plots/gcf7.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/gcf7.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% 3D Plot

t_min = 500;
t_max = 800;

gcf8 = figure(8);
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

saveas(gcf8, "./plots/ekf_traj_3D.eps");
tikzfilename = strcat(currentFolder,'/tikzfiles/ekf_traj_3D.tex');
cleanfigure; 
matlab2tikz('filename',tikzfilename);

%% Innovation for specific satellite

% sat_number = 11;
% 
% t_min = 1;
% t_max = 2000;
% t_vec = t(t_min:t_max);
% 
% gcf9 = figure(9);
% clf;
% plot(t_vec, e_k(sat_number,t_min:t_max));
% grid on
% xlabel("t(k)");
% ylabel("innovation e_k(k)");

























































%.

