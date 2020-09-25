function val = h_func(p_i,xhat_k_km1,ref_data)

    x_rec   = xhat_k_km1(1);
    y_rec   = xhat_k_km1(3);
    z_rec   = xhat_k_km1(5);
    delta_t = xhat_k_km1(7);
    
    x_i     = p_i(1);
    y_i     = p_i(2);
    z_i     = p_i(3);
    
    val =  sqrt((x_i - x_rec)^2 + (y_i - y_rec)^2 + (z_i - z_rec)^2) + ref_data.c*delta_t;

end