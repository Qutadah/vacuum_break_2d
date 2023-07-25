from my_constants import *
from functions import *
import matplotlib.animation as animation
# u : axial velocity
# v : radial velocity

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

# Calculate initial values
T_0, rho_0, p_0, e_0, Ut_0, u_0, v_0 = bulk_values(T_s)

# Array initialization

p1, rho1, u1, v1, Ut1, e1, T1, rho2, u2, v2, Ut2, e2, T2, p2, Tw1, Tw2, Ts1, Ts2, Tc1, Tc2, de0, de1, rho3, u3, v3, Ut3, e3, T3, p3 = initialize_grid(
    p_0, rho_0, e_0, T_0, T_s)

#  Smoothing inlet

# constant inlet
out_cons = val_in_constant()


p_in = out_cons[0]
u_in = out_cons[1]
v_in = out_cons[2]
rho_in = out_cons[3]
e_in = out_cons[4]
T_in = out_cons[5]

# setting wall and frost layer initial conditions
# p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(0)
print("p_in: ", p_in, "u_in: ", u_in, "v_in: ", v_in,
      "rho_in: ", rho_in, "e_in: ", e_in, "T_in: ", T_in)


# BC SURFACES- check deep copy
Ts1[:] = T1[:, Nr]
Ts2[:] = Ts1

# inlet BCs
print("Applying inlet BCs")
u1, v1, Ut1, p1, rho1, T1, e1 = inlet_BC(
    u1, v1, Ut1, p1, rho1, T1, e1, p_in, u_in, rho_in, T_in, e_in)
# negative temp check
if np.any(T1 < 0):
    print("Temp inlet_BC has at least one negative value")
    exit()

p1, rho1, T1, u1, Ut1, e1 = outlet_BC(p1, e1, rho1, u1, v1, Ut1, rho_0)

# negative temp check
if np.any(T1 < 0):
    print("Temp outlet BC has at least one negative value")
    exit()

# PARABOLIC VELOCITY PROFILE - smoothing of parabolic velocity inlet

u1, v1, Ut1, e1 = parabolic_velocity(rho1, T1, u1, v1, Ut1, e1)

T1 = (e1 - 1./2.*rho1*Ut1**2) * 2./5. / rho1/R*M_n

p1 = rho1*R/M_n*T1

# negative temp check
if np.any(T1 < 0):
    print("Temp parabolic after smoothing has at least one negative value")
    exit()

# PREPPING AREA - smoothing of internal properties
p1, rho1, T1 = smoothing_inlet(
    p1, rho1, T1, p_in, p_0, rho_in, rho_0, n_trans)

# recalculate energy

e1 = 5./2. * p1 + 1./2 * rho1*Ut1**2.

# negative temp check
if np.any(T1 < 0):
    print("Temp smoothing has at least one negative value")
    exit()

print("Applying No-slip BC")
# NO SLIP BC
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
print("Saving initial fields")
# save initial fields
save_initial_conditions(rho1, u1, v1, Ut1, e1, T1, de0, p1, de1)

# i1 = 0
print("Plotting initial fields")
plot_imshow(p1, u1, T1, rho1, e1)
# save_plots(i1, p1, u1, T1, rho1, e1)

# Gradient starting matrices
# calculate initial gradients matrix:
print("Calculating initial gradients")
d_dr, m_dx = grad_rho_matrix(v1, u1, rho1)
dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p1, u1)
dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p1, v1)
grad_x, grad_r = grad_e2_matrix(v1, u1, e1)

print("Saving gradients to file")
save_gradients(d_dr, m_dx, dp_dx, ux_dx, ux_dr,
               dp_dr, ur_dx, ur_dr, grad_x, grad_r)

# NOTE: This means at m=0 no mass deposition and no helium...We dont want the surface to freeze.
# Tw1[0] = 298.
# Tw2[0] = 298.
# Ts1[0] = 298.
# Ts2[0] = 298.
# print("Tw init:", Tw1)


# def main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2, ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3):
print("Main loop started")


def main_calc(p1, rho1, T1, u1, v1, Ut1, e1, p2, rho2, T2, u2, v2, Ut2, e2, de0, de1, p3, rho3, T3, u3, v3, Ut3, e3, Tw1, Ts1, Tc1, p_in, rho_in, T_in, e_in, u_in, v_in, rho_r, rho_x, rhs_rho_term, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term, e_r, e_x, rhs_e_term):

    N = n_matrix()
    # NOTE: use ss for plotting terms
    for i in np.arange(np.int64(0), np.int64(Nt+1)):
        print("Iteration: #", i)

        # variable inl et
        # p_in, q_in, ux_in, ur_in, rho_in, e_in = val_in(
        #     i, ux_in)  # define inlet values

        # p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(i)

        # constant inlet
        print("Assigning inlet values")
        p_in, u_in, v_in, rho_in, e_in, T_in = val_in_constant()

        # print("Creating empty de1 matrix to save variable mass deposition")
        # de1 matrix this is the de_variable in RK3 function
        # de1 = np.zeros((Nx+1), dtype=(np.float64))  # mass deposition rate.
        # Initialized 0 and then put in RK3 function to recalculate at all timesteps

        # print("Creating empty de_timestep matrix to save final mass deposition")
        # if i == 1:
        #     de_timestep = np.zeros((Nx+1), dtype=(np.float64))  # place holder

        # else:
        # #     # NOTE: This will take last de from RK3, i need the mass deposition rate of previous time step for the next
        # de_var = rk_out[0]
        # print("Rk3 next")

        # RK3 time integration
        # rk_out = [de_timestep, qn, uxn, urn, uun, en, tn, pn]
        # rk_out = tvdrk3(
        #     u1, v1, Ut1, p1, rho1, T1, e1, p_in, u_in, rho_in, T_in, e_in, rho_0, v_in, i)


# simple time integration
        p2, rho2, T2, u2, v2, Ut2, e2, rho_r, rho_x, rhs_rho_term, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term, e_r, e_x, rhs_e_term = simple_time(
            p1, rho1, T1, u1, v1, Ut1, e1, p_in, rho_in, T_in, e_in, u_in, v_in, rho_0, rho_r, rho_x, rhs_rho_term, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term, e_r, e_x, rhs_e_term, i)

        # out = [p2, rho2, T2, u2, v2, Ut2]

# RK4

# calculating Peclet for field, helps later for differencing scheme used
        # Pe1 = Peclet_grid(Pe, u1, D_hyd, p1, T1)

        print("NAN check next")

# perform NAN value matrix checks:
        # for x in np.arange(len(out)):
        #     assert np.isfinite(out[x]).all()

        # negative density check
        if np.any(rho2 < 0):
            print("The Density Array has at least one negative value")
            exit()

        # negative energy check
        if np.any(e2 < 0):
            print("The energy has at least one negative value")
            exit()

# Temp check
        # if np.any(T2 < 0):
        #     print("Temp returned has at least one negative value")
        #     exit()

#         print("Calculating frost layer thickness")
# # calculate frost layer thickness
#         de0, del_SN = integral_mass_delSN(de2)

#         print("Performing check on negative frost layer thickness")

#         if np.any(del_SN < 0):
#             print("negative frost layer thickness found")
#             exit()

#         print("calculating wall temperature")

# insert wall function

        # w_out = Cu_Wall_function(
        #     ur1, T1, Tw1, Tc1, Ts1, T_in, del_SN, de1, e1, u1, rho1, p1, T2, p2, e2, rho2, u2, ur2)

    # defining next values from RK3
        # Tw2 = w_out[0]
        # Ts2 = w_out[1]
        # Tc2 = w_out[2]
        # qhe = w_out[3]
        # dt2nd_w_m = w_out[4]
        # q_dep = w_out[5]


# save qhe, qdep matrices in timestep
        # print("saving qhe")
        # save_qhe(i, dt, qhe)
        # print("saving q_dep")
        # save_qdep(i, dt, q_dep)

# check Ts1 and T2 temperatures align, rebalances energies within

        # print("making sure wall Tg> Ts")
        # T2, e2, p2, rho2, u2 = gas_surface_temp_check(
        # T2, Ts2, ur2, e2, u2, rho2)
        # print("plotting returning")
        # plot_imshow(p2, u2, T2, rho2, e2)


# Returning result
        print("Returning results for the next time iteration")

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
        # de1[:] = de2[:]

# Recalculate PECLET
        # Pe2 = Peclet_grid(Pe1, u1, D_hyd, p1, T1)

# DELETE R=0 Point/Column
# The 3 index indicates matrices with no r=0, deleted column..
        print("Deleting the r=0")
        rho3, u3, v3, Ut3, e3, T3, p3 = delete_r0_point(
            rho1, u1, v1, Ut1, e1, T1, p1)

# SAVING DATA
        print("Saving data")
        save_data(i, dt, rho3, u3, v3, Ut3, e3, T3, Tw2, Ts2, de0, p3)

# point to plot terms with time
        # aa = 3
        # bb = 30

        # if i == ss-1:
        #     x = np.linspace(0, 7, ss)
        #     fig, axs = plt.subplots(5)
        #     plt.suptitle("Momentum R terms")

        # y1 = rho_x[:, aa, bb]
        # y2 = rho_r[:, aa, bb]
        # y3 = rhs_rho_term[:, aa, bb]
        # axs[0].plot(x, y1)
        # axs[1].plot(x, y2)
        # axs[2].plot(x, y3)

        # y1 = pressure_x[:, aa, bb]
        # y2 = visc_x[:, aa, bb]
        # y3 = ux_x[:, aa, bb]
        # y4 = ur_x[:, aa, bb]
        # y5 = rhs_ux_term[:, aa, bb]
        # axs[0].plot(x, y1)
        # axs[1].plot(x, y2)
        # axs[2].plot(x, y3)
        # axs[3].plot(x, y4)
        # # axs[4].plot(x, y5)

        # y1 = pressure_r[:, aa, bb]
        # y2 = visc_r[:, aa, bb]
        # y3 = ux_r[:, aa, bb]
        # y4 = ur_r[:, aa, bb]
        # y5 = rhs_ur_term[:, aa, bb]
        # axs[0].plot(x, y1)
        # axs[1].plot(x, y2)
        # axs[2].plot(x, y3)
        # axs[3].plot(x, y4)
        # axs[4].plot(x, y5)

        # y1 = e_r[:, aa, bb]
        # y2 = e_x[:, aa, bb]
        # y3 = rhs_e_term[:, aa, bb]
        # axs[0].plot(x, y1)
        # axs[1].plot(x, y2)
        # axs[2].plot(x, y3)

        # plt.title("rhs_e_term term")
        # plt.plot(x, y, color="red")
        # plt.show()

# PLOTTING FIELDS
        if i % 20 == 0:
            print("plotting current iteration", i)
            plot_imshow(p3, u3, T3, rho3, e3)
# First set up the figure, the axis, and the plot element we want to animate
        # im = plt.imshow((p3, u3, T3, rho3, e3),
        #                 interpolation='none', aspect='auto', vmin=0, vmax=1)

# save_plots(i)
# if np.any(T2[:, Nr] < Ts2):
#     if i == 0:
#         aii = 0
#     else:
#         ae += aii
#     print("Tg<Ts detected", ae)


if __name__ == "__main__":
    # main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2,
    #          ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3)
    main_calc(p1, rho1, T1, u1, v1, Ut1, e1, p2, rho2, T2, u2, v2,
              Ut2, e2, de0, de1, p3, rho3, T3, u3, v3, Ut3, e3, Tw1, Ts1, Tc1, p_in, rho_in, T_in, e_in, u_in, v_in, rho_r, rho_x, rhs_rho_term, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term, e_r, e_x, rhs_e_term
              )

# END OF PROGRAM

# Plotting values after BCs

#         # fig, axs = plt.subplots(2, 2)
# #        print("Radius", R_cyl)
#         r1 = np.linspace(0, R_cyl, Nr+1)  # r = 0 plotted
#         r = np.delete(r1, 0, axis=0)  # r = 0 point removed from array
# #        print("array", r)
#         X = np.linspace(0, L, Nx+1)

# #        print("linspace", R)
#        # print("shape r", np.shape(r))
#         # RADIAL DIRECTION
#         # a = rho1[0,:]
#         b = u3[20, :]
#         c = T3[20, :]
#         # d = Ts1[:]
#         # e = Tw1[0,:]
#         f = p3[20, :]
#         g = ur3[20, :]

    # AXIAL DIRECTION
    # a = rho3[:,Nr]
    # b = u3[:, Nr]
    # c = T1[:, Nr]
    # d = Ts1[:]
    # e = Tw1[:]
    # f = p3[:, Nr]
    # g= de1[:]
    # h= de0[:]


# ----------------------- start plot radius ---------------------------------------- #
    """
    # NOTE:
            fig, axs = plt.subplots(4)
            fig.tight_layout()
            fig.suptitle('Properties along radial axis @ m=20')
            axs[0].scatter(r, b, label="Velocity", color='red')
            axs[0].set(ylabel='U [m/s]')
            # plt.ylabel("Velocity [m/s]")
            axs[1].scatter(r, c, label="Temperature", color='blue')
            axs[1].set(ylabel='Temperature [K]')
            # plt.ylabel("Temperature [K]")
            axs[2].scatter(r, f, label="Pressure", color='green')
            axs[2].set(ylabel='Pressure [Pa]')
            # plt.ylabel("Pressure [Pa]")
            axs[3].scatter(r, g, label="Ur", color='yellow')
            axs[3].set(ylabel='Ur [m/s]')
            plt.xlabel("radius (m)")
            plt.show()

    """

# end plot radius
    # plt.figure()
    # plt.subplot(210)
    # plt.scatter(r, b, label="Velocity", color='red')
    # plt.title("Velocity - Radial axis")
    # plt.xlabel("radius (m)")
    # plt.ylabel("Velocity [m/s]")
    # # ax.set_xlabel("Radius (r)", fontsize=14)
    # # ax.set_ylabel("Velocity",
    # #       color="black",
    # #       fontsize=14)

    # plt.subplot(211)
    # plt.scatter(r, c, label="Temperature", color='blue')
    # plt.subplot(212)
    # plt.title("Tg- Radial axis")
    # plt.xlabel("radius (m)")
    # plt.ylabel("Temperature [K]")

    # plt.scatter(r, f, label="Pressure", color='green')
    # plt.title("Pressure - Radial axis")
    # plt.xlabel("radius (m)")
    # plt.ylabel("P [Pa]")

    # plt.scatter(r,d)
    # plt.scatter(r,e)
 #       plt.scatter(r,f)
    # plt.plot(r,b)
    # plt.plot(r,c)
    # plt.plot(r,d)
    # plt.plot(r,e)
#        plt.plot(X,g)
    # plt.plot(X,h)
    # plt.plot(X,b)
    # plt.title("Axial velocity along X axis")
    # plt.xlabel("Length (m)")
    # plt.ylabel("Ux (m/s)")

#        plt.ylim((0, 0.05))   # set the ylim to bottom, top
    # axs[0, 0].scatter(r, a)
    # axs[0, 0].set_title('density along R')
    # axs[0, 1].plot(r, b, 'tab:orange')
    # axs[0, 1].set_title('Velocity along R')
    # axs[1, 0].plot(r, c, 'tab:green')
    # axs[1, 0].set_title('Tg along R')
    # axs[1, 1].plot(X, d, 'tab:red')
    # axs[1, 1].set_title('Ts along R')

    # axs[0, 0].scatter(X, a)
    # axs[0, 0].set_title('density along R')
    # axs[0, 1].plot(X, b, 'tab:orange')
    # axs[0, 1].set_title('Velocity along R')
    # axs[1, 0].plot(X, c, 'tab:green')
    # axs[1, 0].set_title('Tg along R')
    # axs[1, 1].plot(X, d, 'tab:red')
    # axs[1, 1].set_title('Ts along R')
#        plt.title("Pressure along inlet in the r-direction")
 #       plt.legend()


# define global tx to save in worksheets.

#        tx = t


# VTK CONVERSION - not working - not important
    # vtk_convert(rho3, ux3, ur3, u3, e3, T3, Tw2, Ts2, de0, p3, de1, Pe3)
    # numpyToVTK(rho3)
    # numpyToVTK(ux3)
    # numpyToVTK(ur3)
