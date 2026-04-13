def step_heat(state, t, dt):
    pass

# import numpy as np
# import matplotlib.pyplot as plt

# def main():
#     L = 20.0       # length in x-direction (m)
#     nx = 100       # number of grid points in x
#     dx = L / nx    # grid spacing in x
    
#     H = 11.25      # length in y-direction (m)
#     ny = 75        # number of grid points in y
#     dy = H / ny    # grid spacing in y
    
#     dt = 100.0     # time step (s)
#     t_end = 200000 # total simulation time (s)
#     n_steps = int(t_end / dt)
    
#     u0 = 0.0000317     # default flow velocity (m/s)
#     lam_w = 0.6        # thermal conductivity of water
#     rho_w = 10**3      # density of water
#     c_w = 4.18 * 10**3 # specific heat capacity of water
    
#     phi = 0.2          # porosity
#     lam_s = 1.1        # thermal conductivity of solid
#     rho_s = 2.6 * 10**3# density of solid
#     c_s = 0.8 * 10**3  # specific heat solid
    
#     alpha = (lam_w + lam_s) / (c_w * rho_w * phi + c_s * rho_s * (1 - phi))
#     gamma = c_w * rho_w / (c_w * rho_w * phi + c_s * rho_s * (1 - phi))
#     alpha = alpha * 100 
    
#     T_high = 80.0
#     T_low = 15.0
    
#     T = np.ones((ny, nx)) * T_high
#     T[:, 0] = T_low
    
#     u_x = np.ones((ny, nx)) * u0
#     u_y = np.zeros((ny, nx))
    
#     T_history = [T.copy()]
#     save_interval = 200
    
#     for step in range(1, n_steps + 1):
#         T_new = T.copy()
#         for j in range(ny):
#             for i in range(nx):
#                 if i == 0:
#                     T_new[j, i] = T_low
#                     continue
#                 elif i == nx - 1:
#                     d2T_dx2 = (T[j, i-1] - 2*T[j, i] + T[j, i-1]) / dx**2
#                 else:
#                     d2T_dx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / dx**2
                
#                 if j == 0:
#                     d2T_dy2 = (T[j+1, i] - 2*T[j, i] + T[j+1, i]) / dy**2
#                 elif j == ny - 1:
#                     d2T_dy2 = (T[j-1, i] - 2*T[j, i] + T[j-1, i]) / dy**2
#                 else:
#                     d2T_dy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / dy**2
                
#                 diffusion = alpha * (d2T_dx2 + d2T_dy2)
                
#                 if u_x[j, i] >= 0:
#                     dT_dx = (T[j, i] - T[j, i-1]) / dx
#                 else:
#                     if i < nx - 1:
#                         dT_dx = (T[j, i+1] - T[j, i]) / dx
#                     else:
#                         dT_dx = 0
                        
#                 if u_y[j, i] >= 0:
#                      if j > 0:
#                         dT_dy = (T[j, i] - T[j-1, i]) / dy
#                      else:
#                         dT_dy = 0
#                 else:
#                     if j < ny - 1:
#                         dT_dy = (T[j+1, i] - T[j, i]) / dy
#                     else:
#                         dT_dy = 0
                
#                 convection = gamma * (u_x[j, i] * dT_dx + u_y[j, i] * dT_dy)
#                 T_new[j, i] = T[j, i] + dt * (diffusion - convection)
                
#         T = T_new.copy()
        
#         if step % save_interval == 0:
#             T_history.append(T.copy())
#             print(f"Step {step}/{n_steps} completed. Physical time: {step*dt} s")
            
#     plt.figure(figsize=(10, 6))
#     plt.imshow(T, origin='lower', extent=[0, L, 0, H], cmap='hot', aspect='auto')
#     plt.colorbar(label='Temperature (°C)')
#     plt.title('Final Temperature Distribution')
#     plt.xlabel('x (m)')
#     plt.ylabel('y (m)')
#     plt.savefig('heat_convection_result.png')
#     print("Execution complete. Final plot saved as 'heat_convection_result.png'")

# if __name__ == "__main__":
#     main()
