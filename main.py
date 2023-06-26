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


## ----------------------------------------- logging ----------------------------------------- ##
# wb = openpyxl.Workbook()
# ws = wb.active


#### -----------------------------------------   Calculate initial values ----------------------------------------- #
T_0, rho_0, p_0, e_0, ux_0 = bulk_values()

# ----------------- Array initialization ----------------------------

p1, rho1, ux1, ur1, u1, e1, T1, rho2, ux2, ur2, u2, e2, T2, p2, Tw1, Tw2, Ts1, Ts2, Tc1, Tc2, de0, de1, qhe, rho3, ux3, ur3, u3, e3, T3, p3, Pe, Pe1 = initialize_grid(
    p_0, rho_0, e_0, T_0, T_s)

# ---------------------  Smoothing inlet --------------------------------

# constant inlet
p_in, ux_in, ur_in, rho_in, e_in, T_in = val_in_constant()

# p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(0)
print("p_in: ", p_in, "ux_in: ", ux_in, "ur_in: ", ur_in, "rho_in: ",
      rho_in, "e_in: ", e_in, "T_in: ", T_in)
### ------------------------------------- PREPPING AREA - smoothing ------------------------------------------------- ########

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


####### ---------------------------- PARABOLIC VELOCITY PROFILE - inlet prepping area-------------------------------------------------------- ######


# ux, u = parabolic_velocity(ux1, ux_in, T1)

# ---------------------------------------------------------- NO SLIP BC

ux1, u1, e1 = no_slip(ux, u)

# ------  inlet BCs

ux1, ur1, u1, p1, rho1, T1, e1 = inlet_BC(
    ux1, ur1, u1, p1, rho1, T1, e1, p_in, ux_in, rho_in, T_in, e_in)


# Calculating Peclet number in the grid points to determine differencing scheme
Pe1 = Peclet_grid(Pe, u1, D_hyd, p1, T1)

## ------------------------------------------------------------- SAVING INITIAL MATRICES ---------------------------------------------------------------- #####


# rho3, ux3, ur3, u3, e3, T3, p3, Pe3 = delete_r0_point(
#     rho1, ux1, ur1, u1, e1, T1, p1, Pe1)

# remove timestepping folder
remove_timestepping()

# save initial fields
save_initial_conditions(rho1, ux1, ur1, u1, e1, T1,
                        Tw1, Ts1, de0, p1, de1, Pe1)

plot_imshow(p1, ux1, T1, rho1, e1)
# plt.imsave('result.png', )

## ------------------------------------------------ BC INLET starting matrices  ------------------------------------------------- #
# calculate initial gradients matrix:
a, d_dr, m_dx = grad_rho_matrix(ux_in, rho_in, ur1, ux1, rho1)
dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p_in, p1, ux_in, ux1)
dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p, ur1, ur_in)
grad_x, grad_r = grad_e2_matrix(ur1, ux1, ux_in, e_in, e1)

save_gradients(a, d_dr, m_dx, dp_dx, ux_dx, ux_dr,
               dp_dr, ur_dx, ur_dr, grad_x, grad_r)


# NOTE: This means at m=0 no mass deposition and no helium...We dont want the surface to freeze.
# Tw1[0] = 298.
# Tw2[0] = 298.
# Ts1[0] = 298.
# Ts2[0] = 298.
# print("Tw init:", Tw1)


# def main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2, ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3):

def main_cal(p1, rho1, T1, ux1, ur1, e1, p2, rho2, T2, ux2, ur2, u2, e2, de0, de1, p3, rho3, T3, ux3, ur3, u3, e3, pe):

    # ------------------------------------- Time iteration  --------------------------------------------- #

    # create N matrix: needed once only
    N = n_matrix()
    assert np.isfinite(N).all()

    for i in np.arange(np.int64(0), np.int64(Nt+1)):

        # variable inl et
        # p_in, q_in, ux_in, ur_in, rho_in, e_in = val_in(
        #     i, ux_in)  # define inlet values

        # p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(i)

        # constant inlet
        p_in, ux_in, ur_in, rho_in, e_in, T_in = val_in_constant()

        # RK3 time integration
        ux2, ur2, u2, p2, rho2, T2, e2 = tvdrk3(
            ux1, ur1, u1, p1, rho1, T1, e1, p_in, ux_in, rho_in, T_in, e_in, rho_0, ur_in)


# calculating Peclet for field, helps later for differencing scheme used
        Pe1 = Peclet_grid(Pe, u1, D_hyd, p1, T1)

# # Calculating gradients

#         # calculate first derivative matrix:
#         d_dr, m_dx = grad_rho_matrix(ux_in, rho_in, ur1, ux1, rho1)
#         dp_dx, ux_dr, ux_dr = grad_ux2_matrix(p_in, p1, ux_in, ux1)
#         dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p, ur1, ur_in)
#         grad_x, grad_r = grad_e2_matrix(ur1, ux1, ux_in, e_in, e1)

# # calculating second derivative
#         dt2x_ux, dt2x_ur = dt2x_matrix(ux_in, ur_in, ux1, ur1)
#         dt2r_ux, dt2r_ur = dt2r_matrix(ux1, ur1)
#         assert np.isfinite(a).all()
#         assert np.isfinite(d_dr).all()
#         assert np.isfinite(m_dx).all()
#         assert np.isfinite(dp_dx).all()
#         assert np.isfinite(ux_dr).all()
#         assert np.isfinite(ur_dx).all()
#         assert np.isfinite(ur_dr).all()
#         assert np.isfinite(dp_dr).all()
#         assert np.isfinite(grad_x).all()
#         assert np.isfinite(grad_r).all()

#         # viscosity calculations
#         visc_matrix = viscous_matrix(T1, p1)
#         assert np.isfinite(visc_matrix).all()

        # mass deposition rate matrix and source term
        # S = 0  # no mass deposition
        # de i mass deposion rate per unit area, de1 or de0

        # use RK3

        if np.any(rho2 < 0):
            print("The Density Array has at least one negative value")
            exit()
        assert np.isfinite(rho2).all()

        # ensure no division by zero
        rho2 = no_division_zero(rho2)


# print("T1 bulk: ", T1[m, n], "T2 bulk:", T2[m, n])
#                     check_negative(T1[m, n], n)
#                     check_negative(T2[m, n], n)

        # print("P2 surface: ", p2[m, n])
        # check_negative(p2[m, n], n)

        # calculate RHS
        # rhs_rho1 = rhs_rho(dr, dx, ur1[m, n], ux1[m, n],
        #                    rho1[m, n], a, ux_in, rho_in, d_dr[m, n], m_dx[m, n])
        # rhs_ux1, rhs_ur_1 = rhs_ma(dr, dx, ux_in, p,
        #                            p_in, ux1, ur_in, ur[m, n], rho1)
        # rhs_e1 = rhs_energy(m, n, dr, dx, ur1,
        #                     ux1, ux_in, e_in, e1)

# --------------------------- Equations ------------------------------------------- #

    # Only consider mass deposition at a large enough density, otherwise the program will arise negative density

        # # print("mass calculated de1[m]", de1[m], "m", m, "n", n)
        # # No convective heat flux. q2
        # else:
        #     de1[m] = 0
        #     print("NO MASS DEPOSITION: ")

        # Integrate deposition mass
        # de0[m] += dt*np.pi*D*de1[m]
        # print("deposition mass surface", de0[m])
        # check_negative(de0[m], n)

        # Calculate the SN2 layer thickness
        # del_SN = de0[m]/np.pi/D/rho_sn
        # print("del_SN: ", del_SN)
        # check_negative(del_SN, n)

#                     de1[m] = 0

#                     # density calculation
#                     rho2[m, n] = rho1[m, n] - dt/(n*dr*dr)*(rho1[m, n]*(n)*dr*ur1[m, n] - rho1[m, n-1]*(
#                         n-1)*dr*ur1[m, n-1]) - 4*dt/D * de1[m]
#                     # rho2[m, n] = rho1[m, n] - dt/(n*dr*dr)*(rho1[m, n]*(n)*dr*ur1[m, n] - rho1[m, n-1]*(
#                     #     n-1)*dr*ur1[m, n-1]) - dt/dx*(dm) - 4*dt/D * de1[m]   ## New assumption
#                     print("rho2 surface", rho2[m, n])
#                     check_negative(rho2[m, n], n)
#                     #     print("dr term:", -dt/(n*dr*dr)*(rho1[m, n]*(n)*dr*ur1[m, n] - rho1[m, n-1]*(
#                     #         n-1)*dr*ur1[m, n-1]), "rho1[m,n]:", rho1[m, n], "rho1[m,n-1]:", rho1[m, n-1])
#                     #     print("rho1", rho1[m, n])

#                     # ensure no division by zero
#                     if rho2[m, n] == 0:
#                         rho2[m, n] = 0.0001
#                         print("Density went to zero")

#                     ur2[m, n] = de1[m]/rho2[m, n]
#                     ux2[m, n] = 0.  # no slip boundary condition.
#                     u2[m, n] = np.sqrt(ux1[m, n]**2 + ur1[m, n]**2)

#                     print("ur1 surface", ur1[m, n], "u1 surface", u1[m, n],
#                           "ur2 surface", ur2[m, n], "u2 surface", u2[m, n])
#                     # check_negative(ur2[m, n], n)
#                     check_negative(u2[m, n], n)

#                     # internal energy current timestep
#                     eps = 5./2.*p1[m, n]
#                     e1[m, n] = eps + 1./2. * rho1[m, n] * ur1[m, n]**2
#                     print("rho1 ", rho1[m, n])
#                     check_negative(rho1[m, n], n)

#                     # define wall second derivative
#                     # dt2nd = dt2nd_wall((m, Tw1))

#                 # Radial heat transfer within Copper section

# # Only consider the thermal resistance through SN2 layer when thickness is larger than a small preset value (taking average value)

# # NOTE: CHECK THIS LOGIC TREE

#                     #     # q deposited into frost layer. Nusselt convection neglected
#                     # q_dep = de1[m]*(1/2*(ur1[m, n])**2 +
#                     #                 delta_h(T1[m, n], Ts1[m]))

#                     # if del_SN > 1e-5:
#                     #     print(
#                     #         "This is del_SN > 1e-5 condition, conduction across SN2 layer considered")

#                     #     # heatflux into copper wall from frost layer
#                     #     qi = k_sn*(Ts1[m]-Tw1[m])/del_SN
#                     #     print("qi: ", qi)
#                     #     check_negative(qi, n)

        #    # pipe wall equation
        #     Tw2[m] = Tw1[m] + dt/(w_coe*c_c(Tw1[m]))*(
        #         qi-q_h(Tw1[m], BW_coe)*Do/D)+dt/(rho_cu*c_c(Tw1[m]))*k_cu(Tw1[m])*dt2nd
        #     print("Tw2: ", Tw2[m])
        #     check_negative(Tw2[m], n)

        #     # SN2 Center layer Tc equation
        #     Tc2[m] = Tc1[m] + dt * \
        #         (q_dep-qi) / (rho_sn * c_n(Ts1[m, n]*del_SN))
        #     print("Tc2: ", Tc2[m, n])
        #     check_negative(Tc2[m], n)

        # else:
        #     # heatflux into copper wall from frost layer
        #     qi = 0
        #     print("qi: ", qi)
        #     check_negative(qi, n)

        # pipe wall equation
        # Tw2[m] = Tw1[m] + dt/(w_coe*c_c(Tw1[m]))*(
        #     qi-q_h(Tw1[m], BW_coe)*Do/D)+dt/(rho_cu*c_c(Tw1[m]))*k_cu(Tw1[m])*dt2nd
        # print("Tw2: ", Tw2[m])
        # check_negative(Tw2[m], n)

        # SN2 Center layer Tc equation
        # NOTE: Is this te wall temperature?
        # Tc2[m] = Tw2[m]
        # print("Tc2: ", Tc2[m])

        # Calculate SN2 surface temp
        # Ts2[m] = 2*Tc2[m] - Tw2[m]
        # print("Ts2: ", Ts2[m])
        # check_negative(Ts2[m], n)

        # NOTE: CHECK THIS SURFACE TEMPERATURE BC with Yolanda
        # if T2[m, n] < Ts2[m]: # or T2[m, n] > Ts2[m]
        #     e2[m, n] = 5./2.*rho2[m, n]*R*Ts2[m] / \
        #         M_n + 1./2.*rho2[m, n]*ur2[m, n]**2

        #     print("THIS IS T2 < Ts")
        #     print("e2 surface", e2[m, n])
        #     check_negative(e2[m, n], n)

        #     T2[m, n] = 2./5.*(e2[m, n] - 1./2.*rho2[m, n]
        #                       * ur2[m, n]**2.)*M_n/rho2[m, n]/R
        #     print(
        #         "T2 surface recalculated to make it equal to wall temperature (BC)", T2[m, n])
        #     check_negative(T2[m, n], n)

        # Heat transfer rate helium
        # qhe[m] = q_h(Tw1[m], BW_coe)*np.pi*Do

        # print("line 759", "Ts1", Ts1[m], "Ts2", Ts2[m], "Tc2", Tc2[m], "c_c(Ts1[m])", c_c(Ts1[m]), "qh", q_h(Ts1[m], BW_coe), "k_cu(Ts1[m])", k_cu(Ts1[m]), "dt2nd", dt2nd)
        # print("qhe: ", qhe[m])
        # check_negative(qhe[m], n)

#                    p2[m, n] = rho2[m, n] * R * T2[m, n]/M_n

        # append [m,n]
        # my_values = [de1[m], rho1[m, n], - dt/(n*dr*dr)*(rho1[m, n]*(n)*dr*ur1[m, n] - rho1[m, n-1]*(
        #         n-1)*dr*ur1[m, n-1]), - 4*dt/D * de1[m], rho2[m,n], ur2[m,n], u2[m,n], eps, p1[m,n], e1[m,n],-dt / \
        #         (n*dr)*(e2_dr),  e2_dr, e2, T2, - dt*4 / \
        #         D*de1[m]*(e1[m, n]/rho1[m, n]),p2[m,n]]    # Create list of values
        # pd.DataFrame({"[m,n]":[[m,n]])
        # df1.append(my_values)
        # my_data.loc[l] = my_values      # Append values as new row

        # # logging results

        # ws["A"+str(m*n)] = [m,n]
        # ws["B1"] =
        # ws["C1"] =
        # ws["D1"] =
        # ws["E1"] =
        # ws["F1"] =
        # ws["G1"] =


#### ------------------------------------- Case 2: no mass deposition (within flow field,away from wall in radial direction)--------------------------------- ####

        # eps = 5./2.*p1[m, n]
        # print("eps bulk:", eps)
        # if eps < 0:
        #     print("negative eps Bulk ", eps)
        #     exit()
        # if math.isnan(eps):
        #     print("NAN EPS Bulk ", eps)
        #     assert not math.isnan(eps)

        # # NOTE: added check artificial limit speed of sound
        #         if ux2[m, n] > np.sqrt(7./5.*R*T2[m, n]/M_n)*1.0:
        #             # Inlet velocity, m/s (gamma*RT)
        #             ux2[m, n] = np.sqrt(7./5.*R*T2[m, n]/M_n)*1.0
        #             print("sound velocity limited reached")


############################################## Updating timesteps finished ############################################################


# ------------------------------------- surface boundary conditions --------------------------------------------- # Not appliccable inviscid flow.

        # ux1 = surface_BC(ux1)
        # ux2 = surface_BC(ux1)

# ------------------------------------- Inlet boundary conditions --------------------------------------------- #

        ux2, ur2, u2, p2, rho2, T2, e2 = inlet_BC(
            ux2, ur2, u2, p2, rho2, T2, e2, p_in, ux_in, rho_in, T_in, e_in)

# ------------------------------------ Outlet boundary conditions ------------------------------------------- # NOTE: Check later
        # p2, rho2, ux2, u2, e2 = outlet_BC(p2, e2, rho2, ux2, ur2, u2, rho_0)


# ------------------------------ Returning results of current time step to next iteration ------------------------- #

        rho1[:, :] = rho2
        u1[:, :] = u2
        ux1[:, :] = ux2
        ur1[:, :] = ur2
        e1[:, :] = e2
        p1[:, :] = p2
        T1[:, :] = T2
        # Tw1[:] = Tw2
        # Ts1[:] = Ts2
        # Tc1[:] = Tc2

# -------------------------------------- Recalculate Peclet number  ---------------------------------------------------
        Pe2 = Peclet_grid(Pe1, u1, D_hyd, p1, T1)

# -------------------------------------- DELETING R=0 Point/Column  ---------------------------------------------------
        # The 3 index indicates matrices with no r=0, deleted column..
        rho3, ux3, ur3, u3, e3, T3, p3, Pe3 = delete_r0_point(
            rho1, ux1, ur1, u1, e1, T1, p1, Pe2)

        # rho3, ux3, ur3, u3, e3, T3, p3, Pe3 = delete_surface_inviscid(
        #     rho3, ux3, ur3, u3, e3, T3, p3, Pe3)

# --------------------------------------- PLOTTING FIELDS ---------------------------------------  #

        plot_imshow(p3, ux3, T3, rho3, e3)

# --------------------------------------- Saving data ---------------------------------------  #

        save_data(i, dt, rho3, ux3, ur3, u3, e3,
                  T3, Tw2, Ts2, de0, p3, de1, Pe3)

        # vtk_convert(rho3, ux3, ur3, u3, e3, T3, Tw2, Ts2, de0, p3, de1, Pe3)
        # numpyToVTK(rho3)
        # numpyToVTK(ux3)
        # numpyToVTK(ur3)


if __name__ == "__main__":
    # main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2,
    #          ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3)
    main_cal(p1, rho1, T1, ux1, ur1, e1, p2, rho2, T2, ux2, ur2,
             u2, e2, de0, de1, p3, rho3, T3, ux3, ur3, u3, e3, Pe1)


## -------------------------------- EXTRAS ----------------------------------------#


## -------------------------------- value checks ----------------------------------------#

# checking Ts = Tg
    # print(Ts1[:])
    # print(T1[:, Nr])

    # if np.array_equiv(Ts1[:], T1[:, Nr]) == True:
    #     # if (Ts1[:] == T1[:, Nr]):
    #     print("first check complete, Tg = Ts")
    # else:
    #     print("check false")
    #     exit()
## -------------------------------------------- Plotting values after BCs-------------------------------------------- ##

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

# ----------------------- end plot radius ---------------------------------------- #
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
