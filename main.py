from my_constants import *
from functions import *
import matplotlib.animation as animation
import json


# u : axial velocity
# v : radial velocity
# a = 0
print("Removing old timestepping folder")
# remove timestepping folder
remove_timestepping()

ss = 10
# Continuity terms
rho_r = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
rho_x = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
rhs_rho_term = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))

# Momentum X terms
pressure_x = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
visc_x = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
ux_x = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
ur_x = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
rhs_ux_term = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
# Momentum R terms
pressure_r = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
visc_r = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
ux_r = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
ur_r = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
rhs_ur_term = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
# energy terms

e_r = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
e_x = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
rhs_e_term = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))


# actual terms

q_rho = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
q_momx = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
q_momr = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))
q_energy = np.zeros((ss, Nx+1, Nr+1), dtype=(np.float64, np.float64))


# Calculate initial values
T_0, rho_0, p_0, e_0, Ut_0, u_0, v_0 = bulk_values(T_s)

# Array initialization

p1, rho1, u1, v1, Ut1, e1, T1, rho2, u2, v2, Ut2, e2, T2, p2, Tw1, Tw2, Ts1, Ts2, Tc1, Tc2, de0, de1 = initialize_grid(
    p_0, rho_0, e_0, T_0, T_s)

#  Smoothing inlet

# constant inlet
p_in, u_in, v_in, rho_in, e_in, T_in = val_in_constant()

# normal conditions
# p_in = 300.
# u_in = 10.
# v_in = 0.
# rho_in = 0.2
# T_in = p_in/rho_in/R*M_n

# Ut_in = np.sqrt(u_in**2. + v_in**2.)
# e_in = 5./2. * p_in + 1./2. * rho_in * u_in**2.


# setting wall and frost layer initial conditions
# p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(0)
print("p_in: ", p_in, "u_in: ", u_in, "v_in: ", v_in,
      "rho_in: ", rho_in, "e_in: ", e_in, "T_in: ", T_in)

# logger.info

# BC SURFACES- check deep copy
# Ts1[:] = T1[:, Nr]
# Ts2[:] = Ts1

# inlet BCs
print("Applying BCs")
u1, v1, Ut1, p1, rho1, T1, e1 = inlet_BC(
    u1, v1, Ut1, p1, rho1, T1, e1)
# negative temp check
if np.any(T1 < 0):
    print("Temp inlet_BC has at least one negative value")
    exit()

p1, rho1, T1, u1, Ut1, e1 = outlet_BC(p1, e1, rho1, u1, v1, Ut1, rho_0)

# negative temp check
if np.any(T1 < 0):
    # print("Temp outlet BC has at least one negative value")
    exit()

# PARABOLIC VELOCITY PROFILE - smoothing of parabolic velocity inlet

u1, v1, Ut1, e1 = parabolic_velocity(rho1, T1, u1, v1, Ut1, e1, u_in, v_in)

T1 = (e1 - 1./2.*rho1*Ut1**2) * 2./5. / rho1/R*M_n
# print(T1)
p1 = rho1*R/M_n*T1

# negative temp check
if np.any(T1 < 0):
    print("Temp parabolic after smoothing has at least one negative value")
    exit()

# PREPPING AREA - smoothing of internal properties
p1, rho1, T1 = smoothing_inlet(
    p1, rho1, T1, p_in, p_0, rho_in, rho_0, n_trans)

# recalculate energy
for j in range(0, Nx+1):
    u1[j, :] = exp_smooth(j + n_trans, u1[j, :]*2, 0, 0.4, n_trans)


Ut1 = np.sqrt(u1**2. + v1**2.)
e1 = 5./2. * p1 + 1./2 * rho1*Ut1**2.

# negative temp check
if np.any(T1 < 0):
    print("Temp smoothing has at least one negative value")
    exit()

# print("Applying No-slip BC")
p1, T1, u1, v1, Ut1, e1 = no_slip_no_mdot(p1, rho1, T1, u1, v1, Ut1, e1)

# negative temp check
if np.any(T1 < 0):
    print("Temp no slip has at least one negative value")
    exit()


# Calculating Peclet number in the grid points to determine differencing scheme
# Pe1 = Peclet_grid(Pe, u1, D_hyd, p1, T1)


# rho3, ux3, ur3, u3, e3, T3, p3, Pe3 = delete_r0_point(
#     rho1, ux1, ur1, u1, e1, T1, p1, Pe1)


# SAVING INITIAL MATRICES
# print("Saving initial fields")
# save initial fields
save_initial_conditions(rho1, u1, v1, Ut1, e1, T1, de0, p1, de1)

# p, rho, tg, u, v, Ut, e = continue_simulation()
# print("resuming simulation")

# i1 = 0
print("Plotting initial fields")
plot_imshow(p1, u1, T1, rho1, e1)
# save_plots(i1, p1, u1, T1, rho1, e1)

# N = n_matrix()
# r = [i for i in np.arange(Nr+1)]
# r = [dr*(1+i*2) for i in range(Nr+1)]
# r = np.array(r)
# r = np.tile(r, (Nx+1, 1))
# r = r*e-2
# r = [i for i in np.arange(Nr)]
# r = np.array(r)
# r = dr*r
# r[0] = dr/2
# # r = np.insert(r, 0, 1., axis=0)
# r = np.tile(r, (Nx+1, 1))
# r = np.transpose(r)
# print(r)

r = [i for i in np.arange(Nr+1)]
r = np.array(r)
r = dr*r
r[0] = dr/2
# r = np.insert(r, 0, 1., axis=0)
r = np.transpose(r)
# r = np.tile(r, (Nx+1, 1))

# Gradient starting matrices
# calculate initial gradients matrix:
# print("Calculating initial gradients")
d_dr, m_dx = grad_rho_matrix(v1, u1, rho1, r)
dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p1, u1, r)
dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p1, v1, r)
grad_x, grad_r = grad_e2_matrix(v1, u1, e1, r)

# rho1, p1, T1, u1, v1, Ut1, e1 = continue_simulation(Nx)
# plot_imshow(p1, u1, T1, rho1, e1)


def main_calc(dt, r, p1, rho1, T1, u1, v1, Ut1, e1, p2, rho2, T2, u2, v2, Ut2, e2):
    # i = 0.001 /dt

    # fig, axs = plt.subplots(5)
    # # fig.clear()

    # im1 = axs[0].imshow(p1.transpose())
    # im2 = axs[1].imshow(T1.transpose())
    # im3 = axs[2].imshow(rho1.transpose())
    # im4 = axs[3].imshow(u1.transpose())
    # im5 = axs[4].imshow(e1.transpose())

    # plt.colorbar(im1, ax=axs[0])
    # plt.colorbar(im2, ax=axs[1])
    # plt.colorbar(im3, ax=axs[2])
    # plt.colorbar(im4, ax=axs[3])
    # plt.colorbar(im5, ax=axs[4])
    # plot_imshow(p1, T1, rho1, u1, e1)  # make an initial plot

    # N = n_matrix()
    # NOTE: use ss for plotting terms
    for i in np.arange(np.int64(0), np.int64(Nt+1)):
        # fig.suptitle('Fields' + str(i))
        if i % 100 == 0:
            print("Timestep: #", i)
        # fig.clear()

        # fig.canvas.draw()  # draw the image
        # plt.pause(0.1)  # slow down the "animation"


# # Dynamically changing time stepping dt
#         os.chdir(
#             'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/system/')
#         # reading the data from the file
#         with open('dictionary.txt') as f:
#             data = f.read()
#         # reconstructing the data as a dictionary
#         js = json.loads(data)
#         # print("Data type after reconstruction : ", type(js))
#         # print(js)
#         Nt = js['timesteps']
#         dt = T_sim/js['timesteps']
#         print("dt: ", dt)

# simple time integration
        p2, rho2, T2, u2, v2, Ut2, e2 = simple_time(dt, r,
                                                    p1, rho1, T1, u1, v1, Ut1, e1)

# Returning result
        rho1[:, :] = rho2
        Ut1[:, :] = Ut2
        u1[:, :] = u2
        v1[:, :] = v2
        e1[:, :] = e2
        p1[:, :] = p2
        T1[:, :] = T2
        Tw1[:] = Tw2
        Ts1[:] = Ts2
        Tc1[:] = Tc2

    #    # PRESSURE DISTRIBUTION
    #     im1 = axs[0].imshow(p1.transpose())
    #     # plt.colorbar(im, ax=ax[0])
    #     axs[0].set(ylabel='Pressure [Pa]')
    #     # plt.title("Pressure smoothing")

    #     # Temperature DISTRIBUTION
    #     im2 = axs[1].imshow(T1.transpose())
    #     axs[1].set(ylabel='Tg [K]')

    #     # axs[1].colorbars(location="bottom")
    #     # axs[2].set(ylabel='temperature [K]')

    #     im3 = axs[2].imshow(rho1.transpose())
    #     axs[2].set(ylabel='Density [kg/m3]')

    #     # VELOCITY DISTRIBUTION
    #     # axs[1].imshow()
    #     im4 = axs[3].imshow(u1.transpose())
    #     # axs[1].colorbars(location="bottom")
    #     axs[3].set(ylabel='Ux [m/s]')
    #     # plt.title("velocity parabolic smoothing")

    #     im5 = axs[4].imshow(e1.transpose())
    #     axs[4].set(ylabel='energy density [J/m3]')

    #     plt.xlabel("Grid points - x direction")

    #     plt.colorbar(im1, ax=axs[0])
    #     plt.colorbar(im2, ax=axs[1])
    #     plt.colorbar(im3, ax=axs[2])
    #     plt.colorbar(im4, ax=axs[3])
    #     plt.colorbar(im5, ax=axs[4])
    #     plot_imshow(p1, T1, rho1, u1, e1)  # make another plot

        # im1.set_array(p1)  # prepare the new image
        # rho3, u3, v3, Ut3, e3, T3, p3 = delete_r0_point(
        #     rho1, u1, v1, Ut1, e1, T1, p1)

        save_data(i, dt, rho1, u1, v1, Ut1, e1, T1, Tw2, Ts2, de0, p1)

# saving last solution
        save_last(i, dt, rho1, u1, v1, Ut1, e1, T1, Tw2, Ts2, de0, p1)
        # save_last(i, dt, rho3, u3, v3, Ut3, e3, T3, Tw2, Ts2, de0, p3)


# PLOTTING FIELDS
        # if i >= 0:
        # if i == 2:
        if i % 19 == 0:
            # if i >= 0:
            # print("plotting current iteration", i)
            plot_imshow(p1, u1, T1, rho1, e1)
            # plt.imsave()
            # save_field_plot(i, p1, u1, T1, rho1, e1)
        # fig.canvas.draw()
        # plt.pause(0.1)


if __name__ == "__main__":
    main_calc(dt, r, p1, rho1, T1, u1, v1, Ut1, e1, p2, rho2, T2,
              u2, v2, Ut2, e2)

# END OF PROGRAM
