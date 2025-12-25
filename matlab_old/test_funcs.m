

x_i = 1;
y_i = 2;
z_i = 3;

delta_t = 0.00001;

x_rec = 5;
y_rec = 6;
z_rec = 7;

p_i = [x_i, y_i, z_i];
xhat = [x_rec, 0, y_rec, 0, z_rec, delta_t, 0];

h = h_func(p_i, xhat, ref_data_struct);




