function h_prime_ = h_prime_func(p_i,xhat_k_km1,var)
        
    x_rec = xhat_k_km1(1);
    y_rec = xhat_k_km1(3);
    z_rec = xhat_k_km1(5);
    
    x_i = p_i(1);
    y_i = p_i(2);
    z_i = p_i(3);

    h_den = sqrt((x_i - x_rec)^2 + (y_i - y_rec)^2 + (z_i - z_rec)^2);

    if var == 'x'
        h_num = x_rec - x_i;         
    elseif var == 'y'
        h_num = y_rec - y_i; 
    else 
        h_num = z_rec - z_i; 
    end
    
    h_prime_ = h_num/h_den;
end