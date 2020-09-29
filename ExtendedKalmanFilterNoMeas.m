function est=ExtendedKalmanFilterNoMeas(gps_data,ref_data, n_min, n_max)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% est=ExtendedKalmanFilter(gps_data,s2r)
%
% Function that calculates the single point position solution from GPS 
% pseudo range measurements using an extended kalman filter. 
%
% Input
% 
% gps_data      1*M array of struct with the fields:
%               Satellite - Name of satellite
%               Satellite_Position_NED - Position of the satellite
%               PseudoRange - Measured pseudo ranges
%
% s2r		variance of range measurement error (use ref_data.s2r)
%        
% Output:
%
% est           Struct with the fields:
%               x_h - Matrix where each column holds the estimated position 
%                     and clock offset (meters) for each time instant.
%               P - Matrix where the columns holds the diagonal elements of
%               the state covariance matrix. 
% 
% Author: Alexander Berndt and Rebecka Winqvist ({alberndt,rebwin}@kth.se)
% Copyright (c) 2014 KTH, ISC License (open source)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

N       =   length(gps_data(1).PseudoRange);	% length of data 
M       =   length(gps_data);                   % number of satellites (=30)
est.x_h =   zeros(8,N);                         % estimate of states
est.P   =   zeros(8,8,N);                       % covariance matrix P 
est.ek  =   NaN(M,N);                           % innovation
est.sat_count   = zeros(1,N);    

s2r     =   ref_data.s2r;                       %
PSD_clk =   ref_data.PSD_clk;                   % 
Ts      =   ref_data.Ts;                        % sampling time

%% Construct process matrix F for model

F_x     =   [1 Ts; 0 1];
F_y     =   [1 Ts; 0 1];
F_z     =   [1 Ts; 0 1]; 
F_clk   =   [1 Ts; 0 1]; 
F       =   blkdiag(F_x,F_y,F_z,F_clk);

%% Construct process matrix F for model

xhat_k_km1  = zeros(8,1); % estimate vector

%% Determine process noise covariance Q

S_x         = 1.0; 
S_y         = 1.0; 
S_z         = 0.8e-2; 
Qtilde_2    = [[(Ts^4)/3 (Ts^3)/2];
               [(Ts^3)/2 Ts^2]];

Q_x         = S_x/Ts*Qtilde_2;
Q_y         = S_y/Ts*Qtilde_2;
Q_z         = S_z/Ts*Qtilde_2;
 
S_phi       = PSD_clk(1);
S_f         = PSD_clk(2);
Q_clk       = [[(S_phi*Ts + S_f*(Ts^3)/3) (Ts^2*S_f)];
               [(Ts^2*S_f) (S_f*Ts)]];

Q           = blkdiag(Q_x,Q_y,Q_z,Q_clk);

%% Determine measurement noise covariance R

R = s2r*eye(M);

%% Initialize estimate covariance

P_k         = zeros(8,8);

%% Run the Simulation

for n=1:N
    
    % Get measurement from satellites    
    y_i_tilde_vec   = zeros(M,1); 
    y_i_vec         = zeros(M,1);
    h_nonlinear     = zeros(M,1);
    H               = zeros(M,8);
    
    satellite_avail = zeros(M,1);
    
    for i=1:M
        % check if satellite measurement i is available (is NOT NAN)
        if ~isnan(gps_data(i).PseudoRange(n))
            
            satellite_avail(i) = 1; %1 = true

            % position (x,y,z) of satellite m
            p_i     = gps_data(i).Satellite_Position_NED(:,n);

            % partial derivative elements of  h'(x)
            h_p_1   = h_prime_func(p_i,xhat_k_km1,'x');
            h_p_2   = 0;
            h_p_3   = h_prime_func(p_i,xhat_k_km1,'y');
            h_p_4   = 0;
            h_p_5   = h_prime_func(p_i,xhat_k_km1,'z');
            h_p_6   = 0;
            h_p_7   = ref_data.c;
            h_p_8   = 0;

            % partial derivative vector h'(x) evaluated at xhat_k_km1
            h_prime     = [h_p_1 h_p_2 h_p_3 h_p_4 h_p_5 h_p_6 h_p_7 h_p_8];

            % measurement from satellite
            y_i         = gps_data(i).PseudoRange(n);

            % LHS of linearization y^i - h(xh) + h'(xh)xh     
            % where xh = \hat{x}
            y_i_tilde           = y_i - h_func(p_i, xhat_k_km1, ref_data) + h_prime*xhat_k_km1;
            
            h_nonlinear(i)      = h_func(p_i, xhat_k_km1, ref_data);
            
            y_i_vec(i)          = y_i;
            y_i_tilde_vec(i)    = y_i_tilde;
            H(i,:)              = h_prime;
        else
            y_i_tilde_vec(i)    = 0;
            H(i,:)              = zeros(1,8);
        end
    end
    
    % determine idxs of available satellite measurements
    idxs            =   find(satellite_avail); 
    sat_count       = sum(satellite_avail); 
    
    H_sub               = H(idxs,:);
    R_sub               = R(idxs,idxs);
    y_i_vec_sub         = y_i_vec(idxs);
    h_nonlinear_sub     = h_nonlinear(idxs);
    
    %% Linear Kalman Filter Equations 
    
    if n < n_min || n > n_max
        % use the NL function for the innovation!
        e_k         =   y_i_vec_sub - h_nonlinear_sub; 

        if P_k == Inf
            disp("P_k is Inf - solving DARE");
            [P_k,~,~] = idare(F',H_sub',Q,R_sub,[],[]);
        end

        R_ek        = H_sub*P_k*H_sub' + R_sub;
        K_k         = F*P_k*H_sub'/R_ek;
    
        P_kp1       = F*P_k*F' + Q - K_k*R_ek*K_k';
        xhat_kp1_k  = F*xhat_k_km1 + K_k*e_k;
    else
        P_kp1       = F*P_k*F' + Q;
        xhat_kp1_k  = F*xhat_k_km1;
    end
    
    %% Set variables for next iteration
    xhat_k_km1      = xhat_kp1_k;
    P_k             = P_kp1;

    %%  Store the estimate
    est.x_h(:,n)    = xhat_kp1_k;
    est.e_k(idxs,n) = e_k;
    est.P(:,:,n)    = P_kp1;
    est.sat_count(n)   = sat_count;

end


end






















































































%.