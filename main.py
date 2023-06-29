from inspect import currentframe
from my_constants import *
from functions import *
import logging
import openpyxl
import pandas as pd
# @numba.jit()


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


# @numba.jit()
def check_negative(var_in, n):  # CHECKS CALCULATIONS FOR NEGATIVE OR NAN VALUES
    # at surface
    if n == Nr:

        if var_in < 0:
            print("negative Surface", var_in)
            exit()
        if math.isnan(var_in):
            print("NAN Surface ", var_in)
            assert not math.isnan(var_in)

    # at BULK
    else:

        if var_in < 0:
            print("negative Bulk ", var_in)
            exit()
        if math.isnan(var_in):
            print("NAN Bulk ", var_in)
            assert not math.isnan(var_in)


# logging
# wb = openpyxl.Workbook()
# ws = wb.active


# Calculate initial values
T_0, rho_0, p_0, e_0, ux_0 = bulk_values(T_s)

# ----------------- Array initialization ----------------------------

p1, rho1, ux1, ur1, u1, e1, T1, rho2, ux2, ur2, u2, e2, T2, p2, Tw1, Tw2, Ts1, Ts2, Tc1, Tc2, de0, de1, qhe, rho3, ux3, ur3, u3, e3, T3, p3, Pe, Pe1 = initialize_grid(
    p_0, rho_0, e_0, T_0, T_s)

# ---------------------  Smoothing inlet --------------------------------

# constant inlet
p_in, ux_in, ur_in, rho_in, e_in, T_in = val_in_constant()


# setting wall and frost layer initial conditions
# p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(0)
print("p_in: ", p_in, "ux_in: ", ux_in, "ur_in: ", ur_in, "rho_in: ",
      rho_in, "e_in: ", e_in, "T_in: ", T_in)

# PREPPING AREA - smoothing
p1, rho1, T1, ux1, u1, e1 = smoothing_inlet(
    p1, rho1, T1, ux1, ur1, ux_in, u1, p_in, p_0, rho_in, rho_0, n_trans)

# NOTE: should i remove it?
# BC SURFACES
Ts1[:] = T1[:, Nr]
Ts2[:] = Ts1
Tw1[:] = T_s
Tw2[:] = T_s
Tc1[:] = T_s
Tc2[:] = T_s


# PARABOLIC VELOCITY PROFILE - inlet prepping area


# ux, u = parabolic_velocity(ux1, ux_in, T1)

print("Applying No-slip BC")

# NOTE: Do i need more boundary conditions?
# ---------- NO SLIP BC
ux1, u1, e1, T1, ur1, p1, rho1 = no_slip(ux1, u1, p1, rho1, T1, ur1)


# ------  inlet BCs
print("Applying inlet BCs")
ux1, ur1, u1, p1, rho1, T1, e1, Tw1, Ts1, Tc1 = inlet_BC(
    ux1, ur1, u1, p1, rho1, T1, e1, p_in, ux_in, rho_in, T_in, e_in, Tw1, Ts1, Tc1)


# initial

# Calculating Peclet number in the grid points to determine differencing scheme
Pe1 = Peclet_grid(Pe, u1, D_hyd, p1, T1)


# rho3, ux3, ur3, u3, e3, T3, p3, Pe3 = delete_r0_point(
#     rho1, ux1, ur1, u1, e1, T1, p1, Pe1)

print("Removing old timestepping folder")

# remove timestepping folder
remove_timestepping()

# SAVING INITIAL MATRICES
print("Saving initial fields")
# save initial fields
save_initial_conditions(rho1, ux1, ur1, u1, e1, T1,
                        Tw1, Ts1, de0, p1, de1, Pe1, Tc1)


print("Plotting initial fields")
plot_imshow(p1, ux1, T1, rho1, e1)

# Gradient starting matrices
# calculate initial gradients matrix:
print("Calculating initial gradients")
d_dr, m_dx = grad_rho_matrix(ux_in, rho_in, ur1, ux1, rho1)
dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p_in, p1, ux_in, ux1)
dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p1, ur1, ur_in)
grad_x, grad_r = grad_e2_matrix(ur1, ux1, ux_in, e_in, e1)


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


def main_cal(p1, rho1, T1, ux1, ur1, u1, e1, p2, rho2, T2, ux2, ur2, u2, e2, de0, de1, p3, rho3, T3, ux3, ur3, u3, e3, pe, Tw1, Ts1, Tc1):

    N = n_matrix()

    for i in np.arange(np.int64(1), np.int64(Nt+1)):
        print("Iteration: #", i)

        # variable inl et
        # p_in, q_in, ux_in, ur_in, rho_in, e_in = val_in(
        #     i, ux_in)  # define inlet values

        # p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(i)

        # constant inlet
        print("Assigning inlet values")
        p_in, ux_in, ur_in, rho_in, e_in, T_in = val_in_constant()

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
        print("Rk3 next")

        # RK3 time integration
        # rk_out = [de_timestep, qn, uxn, urn, uun, en, tn, pn]
        rk_out = tvdrk3(
            ux1, ur1, u1, p1, rho1, T1, e1, p_in, ux_in, rho_in, T_in, e_in, rho_0, ur_in, de1, Tw1, Ts1, Tc1, i)

        print("Rk3 complete")

# defining next values from RK3
        de2 = rk_out[0]
        rho2 = rk_out[1]
        ux2 = rk_out[2]
        ur2 = rk_out[3]
        u2 = rk_out[4]
        e2 = rk_out[5]
        T2 = rk_out[6]
        p2 = rk_out[7]
        visc_matrix = rk_out[8]

# NOTE: RECALCULATE ENERGIES IMPORTANT
        e, T = balance_energy(p2, rho2, u2)
        e, p = balance_energy2(rho2, T2, u2)

# calculating Peclet for field, helps later for differencing scheme used
        Pe1 = Peclet_grid(Pe, u1, D_hyd, p1, T1)

        print("NAN check next")

# perform NAN value matrix checks:
        for x in np.arange(len(rk_out)):
            assert np.isfinite(rk_out[x]).all()

        print("Negative densities and energy next")

        # negative density check
        if np.any(rk_out[1] < 0):
            print("The Density Array has at least one negative value")
            exit()

        # negative energy check
        if np.any(rk_out[5] < 0):
            print("The energy has at least one negative value")
            exit()

        print("Calculating frost layer thickness")
# calculate frost layer thickness
        de0, del_SN = integral_mass_delSN(de2)

        print("Performing check on negative frost layer thickness")

        if np.any(del_SN < 0):
            print("negative frost layer thickness found")
            exit()

        print("calculating wall temperature")

# insert wall function

        w_out = Cu_Wall_function(
            ur1, T1, Tw1, Tc1, Ts1, T_in, del_SN, de1, e1, u1, rho1, p1, T2, p2, e2, rho2, u2, ur2)

    # defining next values from RK3
        Tw2 = w_out[0]
        Ts2 = w_out[1]
        Tc2 = w_out[2]
        qhe = w_out[3]
        dt2nd_w_m = w_out[4]
        q_dep = w_out[5]


# save qhe, qdep matrices in timestep
        # print("saving qhe")
        # save_qhe(i, dt, qhe)
        # print("saving q_dep")
        # save_qdep(i, dt, q_dep)

# NOTE: Perform energy checks throughout the program
# check Ts1 and T2 temperatures align, rebalances energies within

        # print("making sure wall Tg> Ts")
        # T2, e2, p2, rho2, u2 = gas_surface_temp_check(
        # T2, Ts2, ur2, e2, u2, rho2)

# find difference in energies across timesteps
        # d_e = energy_difference_dt(e1, e2)

# Returning results of current time step for i++
        print("Returning results for the next time iteration")

        rho1[:, :] = rho2
        u1[:, :] = u2
        ux1[:, :] = ux2
        ur1[:, :] = ur2
        e1[:, :] = e2
        p1[:, :] = p2
        T1[:, :] = T2
        Tw1[:] = Tw2
        Ts1[:] = Ts2
        Tc1[:] = Tc2
        de1[:] = de2[:]

# Recalculate PECLET
        Pe2 = Peclet_grid(Pe1, u1, D_hyd, p1, T1)

# DELETE R=0 Point/Column
# The 3 index indicates matrices with no r=0, deleted column..
        print("Deleting the r=0 for plotting and saving purposes")
        rho3, ux3, ur3, u3, e3, T3, p3, Pe3, visc_matrix3 = delete_r0_point(
            rho1, ux1, ur1, u1, e1, T1, p1, Pe2, visc_matrix)

        # rho3, ux3, ur3, u3, e3, T3, p3, Pe3 = delete_surface_inviscid(
        #     rho3, ux3, ur3, u3, e3, T3, p3, Pe3)

# SAVING DATA
        save_data(i, dt, rho3, ux3, ur3, u3, e3,
                  T3, Tw2, Ts2, de0, p3, de2, Pe3, q_dep, qhe, visc_matrix3)

# PLOTTING FIELDS
        plot_imshow(p3, ux3, T3, rho3, e3)

        # if np.any(T2[:, Nr] < Ts2):
        #     if i == 0:
        #         aii = 0
        #     else:
        #         ae += aii
        #     print("Tg<Ts detected", ae)


if __name__ == "__main__":
    # main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2,
    #          ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3)
    main_cal(p1, rho1, T1, ux1, ur1, u1, e1, p2, rho2, T2, ux2, ur2,
             u2, e2, de0, de1, p3, rho3, T3, ux3, ur3, u3, e3, Pe1, Tw1, Ts1, Tc1)


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
