function est=ExtendedKalmanFilter(gps_data,ref_data)
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



N = length(gps_data(1).PseudoRange);        % length of data 
M = length(gps_data);                       % number of satellites (=30)
est.x_h = zeros(7,N);                       % estimate of states
est.P = zeros(7,7,N);                       % covariance matrix P 

% get data from ref_data_struct
s2r     = ref_data.s2r;
PSD_clk = ref_data.PSD_clk;
Ts = ref_data.Ts;

F_x =   [1 Ts; 0 1];
F_y =   [1 Ts; 0 1];
F_z =   [1]; 
F_clk = [1 Ts; 0 1]; 


F = blkdiag(F_x,F_y,F_z,F_clk);
xhat_k_km1 = zeros(7,1); % estimate vector
P_k = Inf; % 0.001*eye(7); % INIT in loop

% MEASUREMENT COVARIANCE MATRIX
R_k = s2r*eye(M);

% PROCESS COVARANCE MATRIX
S_x = 1.0e-10; % TODO: update the 
S_y = 1.0e-10;
S_z = 1.0e-10;
Qtilde_2 = [[(Ts^4)/3 (Ts^3)/2];
            [(Ts^3)/2 Ts^2]];

Q_k_x = S_x/Ts*Qtilde_2;
Q_k_y = S_y/Ts*Qtilde_2;
Q_k_z = S_z/Ts;
 
S_phi = PSD_clk(1);
S_f = PSD_clk(2);
Q_k_clk = [[(S_phi*Ts + S_f*(Ts^3)/3) (Ts^2*S_f)];
           [(Ts^2*S_f) (S_f*Ts)]];

Q_k = blkdiag(Q_k_x,Q_k_y,Q_k_z,Q_k_clk);

for n=1:N
    
    % Get measurement from satellites    
    y_i_tilde_vec = zeros(M,1); 
    y_i_vec = zeros(M,1);
    H       = zeros(M,7);
    
    satellite_avail = zeros(M,1);
    
    for i=1:M
        % check if satellite measurement i is available (is NOT NAN)
        if ~isnan(gps_data(i).PseudoRange(n))
            
            satellite_avail(i) = 1;

            % position (x,y,z) of satellite m
            p_i = gps_data(i).Satellite_Position_NED(:,n);

            % partial derivative elements of  h'(x)
            h_p_1 = h_prime_func(p_i,xhat_k_km1,'x');
            h_p_2 = 0;
            h_p_3 = h_prime_func(p_i,xhat_k_km1,'y');
            h_p_4 = 0;
            h_p_5 = h_prime_func(p_i,xhat_k_km1,'z');
            h_p_6 = ref_data.c;
            h_p_7 = 0;

            % partial derivative vector h'(x) evaluated at xhat_k_km1
            h_prime = [h_p_1 h_p_2 h_p_3 h_p_4 h_p_5 h_p_6 h_p_7];

            % measurement from satellite
            y_i = gps_data(i).PseudoRange(n);

            % LHS of linearization y^i - h(xh) + h'(xh)xh     
            % where xh = \hat{x}
            y_i_tilde = y_i - h_func(p_i, xhat_k_km1, ref_data) + h_prime*xhat_k_km1;
            y_i_vec(i) = y_i;
            y_i_tilde_vec(i) = y_i_tilde;
            H(i,:) = h_prime;
        else
            y_i_tilde_vec(i) = 0;
            H(i,:)=zeros(1,7);
        end
    end
    
    % LINEAR KF 
    idxs = find(satellite_avail);
    
    H_sub = H(idxs,:);
    R_k_sub = R_k(idxs,idxs);
    y_i_vec_sub = y_i_vec(idxs);
    
    e_k = y_i_vec_sub - H_sub*xhat_k_km1;
    
    if P_k == Inf
        disp("P_k is Inf - solving DARE");
        [P_k,~,~] = idare(F',H_sub',Q_k,R_k_sub,[],[]);
    end
    
    R_ek = H_sub*P_k*H_sub' + R_k_sub;
    
%     disp("evaluating K_K")
    K_k = F*P_k*H_sub'*inv(R_ek);
%     disp("after K_K")
    
    P_kp1 = F*P_k*F' + Q_k - K_k*R_ek*K_k';
    xhat_kp1_k = F*xhat_k_km1 + K_k*e_k;

    % update time-index for next step
    xhat_k_km1 = xhat_kp1_k;
    P_k = P_kp1;

    % Store the estimate
    est.x_h(:,n) = xhat_kp1_k;
    est.P(:,:,n) = P_kp1;

end


end






















































































%.
