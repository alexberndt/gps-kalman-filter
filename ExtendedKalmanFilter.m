function est=ExtendedKalmanFilter(gps_data,s2r)
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



N=length(gps_data(1).PseudoRange);  % length of data 
M=length(gps_data);                 % number of satellites (=31)
est.x_h=zeros(4,N);                 % estimate of states
est.P=zeros(4,N);                   % estimate of P (?)

for n=1:N
    gps_data;
    x=zeros(4,1); % estimate vector
    dx=inf(4,1);
    res=zeros(M,1);
    H=zeros(M,4);
    itr_ctr=0;
    
    %% Kalman Filter
    while norm(dx)>0.01 && itr_ctr<10;
        
        for m=1:M
            % check if satellite measurement is available (is NOT NAN)
            if ~isnan(gps_data(m).PseudoRange(n))
                dR_h=gps_data(m).Satellite_Position_NED(:,n)-x(1:3);
                res(m)=gps_data(m).PseudoRange(n)-(norm(dR_h)+x(4));
                H(m,1:3)=-dR_h'./norm(dR_h);
                H(m,4)=1;
            else
                res(m)=0;
                H(m,:)=zeros(1,4);
            end
        end
        
        
        % Calculate the correction to the state vector
        dx=(H'*H)\(H'*res);
        
        % Update the state vector
        x=x+dx;
        
        % Update the iteration counter
        itr_ctr=itr_ctr+1;
    end

    % Store the estimate
    est.x_h(:,n)=x;
    est.P(:,n)=s2r*diag(inv(H'*H));
end


end






















































































%.
