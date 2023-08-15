# ----------------- Helper Functions --------------------------------#

import warnings
import copy
from my_constants import *
import os
import shutil
import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


def smoothing_inlet(p, rho, T, p_in, p_0, rho_in, rho_0, n_trans):
    for i in range(0, Nx+1):
        p[i, :] = exp_smooth(i+n_trans, p_in*2.-p_0, p_0, 0.4, n_trans)
    # print("P1 smoothing values", p1[i,:])
        rho[i, :] = exp_smooth(i + n_trans, rho_in*2 -
                               rho_0, rho_0, 0.4, n_trans)
    #    T1[i, :] = T_neck(i)
        # if i<51: T1[i]=T_in
        T[i, :] = p[i, :]/rho[i, :]/R*M_n
        # v_max = np.sqrt(7./5.*R*T/M_n)  # diatomic gas gamma = 7/5
    #    u1[i, :] = exp_smooth(i + n_trans, ux_in*2, 0, 0.4, n_trans)

        # if i < n_trans+1:
        #     e1[i, :] = 5./2.*p1[i, :]+1./2.*rho1[i, :]*u1[i, :]**2

    #        rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]  # IDEAL GAS LAW

        # print("p1 matrix after smoothing", p1)
        # else:
        #     e1[i, :] = 5/2*rho1[i, :]/M_n*R*T_in+1/2**rho1[i, :]*u1[i, :]**2
    # for i in range(0, Nx+1):
    out = p, rho, T
    return out


def exp_smooth(grid, hv, lv, order, tran):
    coe = ((hv+lv)/2-lv)/(np.exp(order*tran)-1)
    c = lv-coe
    if grid < 0:
        s_result = hv
    elif grid < (tran+1):
        s_result = -coe*np.exp(order*(grid))-c+hv+lv
    elif grid < (2*tran+1):
        s_result = coe*np.exp(order*(2*tran-grid))+c
    else:
        s_result = lv
    return s_result


def initialize_grid(p_0, rho_0, e_0, T_0, T_s):

    p1 = np.full((Nx+1, Nr+1), p_0,
                 dtype=(np.longdouble, np.longdouble))  # Pressure
    rho1 = np.full((Nx+1, Nr+1), rho_0,
                   dtype=(np.longdouble, np.longdouble))  # Density
    ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble,
                   np.longdouble))  # velocity -x
    ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble,
                   np.longdouble))  # velocity -r
    u1 = np.sqrt(np.square(ux1) + np.square(ur1))  # total velocity
    e1 = np.full((Nx+1, Nr+1), e_0, dtype=(np.longdouble, np.longdouble))

    T1 = np.full((Nx+1, Nr+1), T_0,
                 dtype=(np.longdouble, np.longdouble))  # Temperature

    rho2 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.longdouble, np.longdouble))
    ux2 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    ur2 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    u2 = np.sqrt(np.square(ux2) + np.square(ur2))  # total velocity
    e2 = np.full((Nx+1, Nr+1), e_0, dtype=(np.longdouble, np.longdouble))
    T2 = np.full((Nx+1, Nr+1), T_0, dtype=(np.longdouble, np.longdouble))
    p2 = np.full((Nx+1, Nr+1), p_0,
                 dtype=(np.longdouble, np.longdouble))  # Pressure

    Tw1 = np.full((Nx+1), T_s, dtype=(np.longdouble))  # Wall temperature
    Tw2 = np.full((Nx+1), T_s, dtype=(np.longdouble))
    # Temperature of SN2 surface
    Ts1 = np.full((Nx+1), T_0, dtype=(np.longdouble))
    Ts2 = np.full((Nx+1), T_0, dtype=(np.longdouble))

    # Average temperature of SN2 layer
    Tc1 = np.full((Nx+1), T_s, dtype=(np.longdouble))
    Tc2 = np.full((Nx+1), T_s, dtype=(np.longdouble))
    de0 = np.zeros((Nx+1), dtype=(np.longdouble))  # Deposition mass, kg/m
    de1 = np.full((Nx+1), 0., dtype=(np.longdouble))  # Deposition rate
    # de2 = np.full((Nx+1), 0., dtype=(np.float64))  # Deposition rate
    qhe = np.zeros_like(de0, dtype=np.longdouble)  # heat transfer

    # These matrices are just place holder. These will be overwritten and saved. (remove r=0)
    # rho3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    # ux3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    # ur3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    # u3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    # e3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    # T3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    # p3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))

    out = [p1, rho1, ux1, ur1, u1, e1, T1, rho2, ux2, ur2, u2, e2, T2, p2,
           Tw1, Tw2, Ts1, Ts2, Tc1, Tc2, de0, de1]
    return out


def n_matrix():
    n = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for i in np.arange(np.int64(0), np.int64(Nx+1)):
        for j in np.arange(np.int64(1), np.int64(Nr+1)):
            n[i, j] = j
    n[:, 0] = 1
    return n


def viscous_matrix(T, P):
    visc_matrix = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            visc_matrix[m, n] = mu_n(T[m, n], P[m, n])
    for x in np.arange(len(visc_matrix)):
        assert np.isfinite(visc_matrix).all()

# negative viscosity check
    if np.any(visc_matrix < 0):
        print("The viscous matrix has at least one negative value")
        exit()
    # visc_matrix[:, :] = 0.
    return visc_matrix


def rhs_rho(d_dr, m_dx, r):  # include i to calculate terms
    rhs_rho = - 1/r*d_dr - m_dx
    A = - 1/r*d_dr
    B = -m_dx

    return rhs_rho


def rhs_ma(dp_dx, rho, dt2r_ux, r, ux_dr, dt2x_ux, ux, ux_dx, ur, dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix):
    A = -dp_dx/rho
    B = visc_matrix/rho * (
        dt2r_ux + 1/r*ux_dr + dt2x_ux)
    C = -ux * ux_dx
    D = - ur*ux_dr

# RHS Ux
    rhs_ux = -dp_dx/rho + visc_matrix/rho * (
        dt2r_ux + 1/r*ux_dr + dt2x_ux) - ux * ux_dx - ur*ux_dr

    E = - dp_dr/rho
    F = visc_matrix/rho * \
        (- ur/(r**2) + 1/r*ur_dr +
         dt2r_ur + dt2x_ur)
    G = - ux * ur_dx
    H = - ur*ur_dr

# RHS Ur
    rhs_ur = - dp_dr/rho + visc_matrix/rho * \
        (- ur/r**2 + 1/r*ur_dr + dt2r_ur + dt2x_ur) - ux * ur_dx - ur*ur_dr

    return rhs_ux, rhs_ur


def no_division_zero(array):
    # ensure no division by zero
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if array[m, n] == 0:
                array[m, n] = 0.0001
    return array


def rhs_energy(grad_r, grad_x, r):
    rhs_e = - 1/r*grad_r - grad_x
    A = -  1/r*grad_r
    B = -grad_x
    return rhs_e


def simple_time(dt, r, p, q, tg, u, v, Ut, e):
    # N = n_matrix()
    T_0, rho_0, p_0, e_0, Ut_0, u_0, v_0 = bulk_values(4.2)
    p_in, u_in, v_in, rho_in, e_in, T_in = val_in_constant()

    q_0 = np.zeros((Nx+1, Nr+1), dtype=(np.float64))
    qq = copy.deepcopy(q_0)  # density
    uxx = copy.deepcopy(q_0)  # velocity axial
    urr = copy.deepcopy(q_0)  # velocity radial
    uu = copy.deepcopy(q_0)  # total velocity
    ee = copy.deepcopy(q_0)  # energy
    tt = copy.deepcopy(q_0)  # temperature

    # print("start simple time")
    # plot_imshow(p, u, tg, q, e)

    # plot_imshow(p, u, tg, q, e)
    u, v, Ut, p, q, tg, e = inlet_BC(
        u, v, Ut, p, q, tg, e)
    # negative temp check
    # if np.any(tg < 0):
    #     print("Temp inlet_BC has at least one negative value")
    #     exit()
    # plot_imshow(p, u, tg, q, e)
    # print("inlet BC here")
    # plot_imshow(p, u, tg, q, e)
    p, q, tg, u, Ut, e = outlet_BC(p, e, q, u, v, Ut, rho_0)
# negative temp check
    # if np.any(tg < 0):
    #     print("Temp outlet_BC has at least one negative value")
    #     exit()

    # u, v, Ut, e = parabolic_velocity(q, tg, u, v, Ut, e, u_in, v_in)

# # energy balance
#     tg = (e - 1./2.*q*Ut**2) * 2./5. / q/R*M_n
#     p = q*R/M_n*tg

    p, tg, u, v, Ut, e = no_slip_no_mdot(p, q, tg, u, v, Ut, e)

# negative temp check
    # if np.any(tg < 0):
    #     print("Temp no slip has at least one negative value")
    #     exit()

    # print(tg)
    # print("plotting")
    # plot_imshow(p, Ut, tg, q, e)


# Calculating gradients (first and second)

    d_dr, m_dx = grad_rho_matrix(v, u, q, r)
    dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p, u, r)
    dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p, v, r)
    grad_x, grad_r = grad_e2_matrix(v, u, e, r)

    dt2x_ux, dt2x_ur = dt2x_matrix(u, v)
    dt2r_ux, dt2r_ur = dt2r_matrix(u, v, r)

    visc_matrix = viscous_matrix(tg, p)
    # print("zero viscosity assumed")
    # visc_matrix[:, :] = 0.

    assert np.isfinite(visc_matrix).all()

    r_rho = rhs_rho(d_dr, m_dx, r)
    r_ux, r_ur = rhs_ma(dp_dx, q, dt2r_ux, r, ux_dr, dt2x_ux, u,
                        ux_dx, v, dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix)

    # p, tg, u, v, Ut, e = no_slip_no_mdot(p, q, tg, u, v, Ut, e)
    # grad_x, grad_r = grad_e2_matrix(v, u, e, N)

    r_e = rhs_energy(grad_r, grad_x, r)

# first LHS calculations
    qq = q + dt*r_rho
    # qq[0, :] = q[0, :]
    uxx = u + dt*r_ux
    # uxx[0, :] = u[0, :]
    urr = v + dt*r_ur
    # urr[0, :] = v[0, :]
    ee = e + dt*r_e
    # ee[0, :] = e[0, :]

    qq = no_division_zero(qq)

# Velocity
    uu = np.sqrt(uxx**2. + urr**2.)
# pressure and temp defining
    tt = (ee - 1./2.*qq*uu**2.) * 2./5. / qq/R*M_n
    pp = qq*R/M_n*tt

    uxx, urr, uu, pp, qq, tt, ee = inlet_BC(
        uxx, urr, uu, pp, qq, tt, ee)
    pp, qq, tt, uxx, uu, ee = outlet_BC(pp, ee, qq, uxx, urr, uu, rho_0)
    pp, tt, uxx, urr, uu, ee = no_slip_no_mdot(pp, qq, tt, uxx, urr, uu, ee)

# negative density check
    if np.any(qq < 0):
        print("The Density Array has at least one negative value")
        exit()

# negative energy check
    if np.any(ee < 0):
        print("The energy has at least one negative value")
        exit()

    return pp, qq, tt, uxx, urr, uu, ee


# adaptive timestep
# def calc_dt(cfl, gamma_n, q, nx, nr, dx, dr):
#     a = 30.0
#     a = np.max([a, 0.0])
#     for j in np.arange(nr):
#         for i in np.arange(nx):
#             rho, ma_x, ma_r, ma_energy = q[:, i, j]
#             ux, ur, e = ma_x/rho, ma_r/rho, ma_energy/rho
#             p = rho*(gamma_n-1)*(e-0.5*(ux ^ 2+ur ^ 2))
#             c = np.sqrt(gamma_n*p/rho)
#             a = np.max([a, abs(ux), abs(ux+c), abs(ux-c),
#                        abs(ur), abs(ur+c), abs(ur-c)])
#     dt = cfl*np.min([dx, dr])/a
#     return dt

def plot_imshow(p, ux, T, rho, e):
    fig, axs = plt.subplots(5)
    fig.suptitle('Fields along tube for all R')

    # PRESSURE DISTRIBUTION
    im = axs[0].imshow(p.transpose())
    plt.colorbar(im, ax=axs[0])
    # plt.colorbar(im, ax=ax[0])
    axs[0].set(ylabel='Pressure [Pa]')
    # plt.title("Pressure smoothing")

    # Temperature DISTRIBUTION
    im = axs[1].imshow(T.transpose())
    plt.colorbar(im, ax=axs[1])
    axs[1].set(ylabel='Tg [K]')

    # axs[1].colorbars(location="bottom")
    # axs[2].set(ylabel='temperature [K]')

    im = axs[2].imshow(rho.transpose())
    plt.colorbar(im, ax=axs[2])
    axs[2].set(ylabel='Density [kg/m3]')

    # VELOCITY DISTRIBUTION
    # axs[1].imshow()
    im = axs[3].imshow(ux.transpose())
    plt.colorbar(im, ax=axs[3])
    # axs[1].colorbars(location="bottom")
    axs[3].set(ylabel='Ux [m/s]')
    # plt.title("velocity parabolic smoothing")

    im = axs[4].imshow(e.transpose())
    plt.colorbar(im, ax=axs[4])
    axs[4].set(ylabel='energy density [J/m3]')

    plt.xlabel("Grid points - x direction")
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    # x = str(i) + '.png'


def save_field_plot(i, p, ux, T, rho, e):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/field_plots/'
    if i == 0:
        if os.path.exists(pathname):
            location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
            dir = "field_plots"
            path = os.path.join(location, dir)
            shutil.rmtree(path)

        if not os.path.exists(pathname):
            os.makedirs(pathname)

    if i % 5000 == 0 and i > 0:
        if os.path.exists(pathname):
            location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
            dir = "field_plots"
            path = os.path.join(location, dir)
            shutil.rmtree(path)
        if not os.path.exists(pathname):
            os.makedirs(pathname)

    # The size of the figure is specified as (width, height) in inches
    fig2 = plt.figure(figsize=(10.0, 15.0))

    fig2, axs = plt.subplots(5)
    fig2.suptitle('Fields along tube for all R')

    # PRESSURE DISTRIBUTION
    im = axs[0].imshow(p.transpose())
    plt.colorbar(im, ax=axs[0])
    # plt.colorbar(im, ax=ax[0])
    axs[0].set(ylabel='Pressure [Pa]')
    # plt.title("Pressure smoothing")

    # Temperature DISTRIBUTION
    im = axs[1].imshow(T.transpose())
    plt.colorbar(im, ax=axs[1])
    axs[1].set(ylabel='Tg [K]')

    # axs[1].colorbars(location="bottom")
    # axs[2].set(ylabel='temperature [K]')

    im = axs[2].imshow(rho.transpose())
    plt.colorbar(im, ax=axs[2])
    axs[2].set(ylabel='Density [kg/m3]')

    # VELOCITY DISTRIBUTION
    # axs[1].imshow()
    im = axs[3].imshow(ux.transpose())
    plt.colorbar(im, ax=axs[3])
    # axs[1].colorbars(location="bottom")
    axs[3].set(ylabel='Ux [m/s]')
    # plt.title("velocity parabolic smoothing")

    im = axs[4].imshow(e.transpose())
    plt.colorbar(im, ax=axs[4])
    axs[4].set(ylabel='energy density [J/m3]')

    plt.xlabel("Grid points - x direction")
    # plt.show()
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/field_plots/'
    os.chdir(pathname)
    fname = "step_" + str(i) + ".png"
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    fig2.savefig(fname, dpi=100)


def grad_rho_matrix(ur, ux, rho, r):
    # create gradients arrays.
    m_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    d_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    for i in np.arange(Nx+1):
        for j in np.arange(Nr+1):
            # if i == 0:
            #     # m_dx[i, j] = (rho[i, j]*ux[i, j]-rho_in*ux_in)/dx
            #     m_dx[i, j] = (rho[i+1, j]*ux[i+1, j]-rho[i, j]*ux[i, j])/dx
            # axial
            if i == 0:
                m_dx[i, j] = (rho[i+1, j]*ux[i+1, j]-rho[i, j]*ux[i, j])/dx
                # m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]
                #   * ux[i-1, j])/dx  # BWD

            else:
                # m_dx[i, j] = (rho[i+1, j]*ux[i+1, j]-rho[i, j]*ux[i, j])/dx
                m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]
                              * ux[i-1, j])/dx  # BWD

# radial
            if j == 0:
                # NOTE: SYMMETRY BC
                # d_dr[i, j] = (rho[i, j+2]*(N[i, j+2]*dr)*ur[i, j+2] +
                #               rho[i, j] * (N[i, j]*dr) * ur[i, j]) / (4*dr)
                d_dr[i, j] = 2*(rho[i, j]*r[j]*ur[i, j])/dr

            else:
                d_dr[i, j] = (rho[i, j]*r[j]*ur[i, j] -
                              rho[i, j-1] * r[j-1]*ur[i, j-1])/(r[j]-r[j-1])

            # else:
            #     d_dr[i, j] = (rho[i, j+1]*(N[i, j+1]*dr)*ur[i, j+1] -
            #                   rho[i, j] * (N[i, j]*dr) * ur[i, j])/dr  # upwind
    # d_dr[:, :] = 0.
    # m_dx[:, :] = 0.
    return d_dr, m_dx


def grad_ux2_matrix(p, ux, r):  # bulk
    ux_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    dp_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    ux_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))

    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            # axial
            if m == 0:  # this doesnt matter since its a boundary
                dp_dx[m, n] = (p[m+1, n] - p[m, n])/dx
                ux_dx[m, n] = (ux[m+1, n] - ux[m, n])/dx

            # if m == Nx:
            #     dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx
            #     ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx

            # elif m == Nx:
            #     dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx  # BWD
            #     ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx  # BWD

            # else:
                # upwind 1st order  - positive flow - advection
                # dp_dx[m, n] = (p[m+1, n] - p[m, n])/dx  # upwind
                # ux_dx[m, n] = (ux[m+1, n] - ux[m, n])/dx  # upwind
            # elif m >= 1:
            #     dp_dx[m, n] = (p[m+1, n] - p[m-1, n])/(2*dx)  # upwind
            #     ux_dx[m, n] = (ux[m+1, n] - ux[m-1, n])/(2*dx)  # upwind
            else:
                dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx  # BWD
                ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx  # upwind

# radial
            if n == 0:
                # NOTE: SYMMETRY CONDITION
                # ux_dr[m, n] = (ux[m, n+2] - ux[m, n])/(4*dr)
                ux_dr[m, n] = 0.
            # elif n == Nr:
            #     ux_dr[m, n] = (ux[m, n] - ux[m, n-1])/dr  # BWD
            else:
                # upwind 1st order  - positive flow - advection
                ux_dr[m, n] = (ux[m, n] - ux[m, n-1])/(r[n]-r[n-1])  # BWD

                # ux_dr[m, n] = (ux[m, n+1] - ux[m, n])/dr  # upwind
    # ux_dr[:, :] = 0.
    # ux_dx[:, :] = 0.
    # dp_dx[:, :] = 0.
    # print(ux_dx)
    return dp_dx, ux_dx, ux_dr


def grad_ur2_matrix(p, ur, r):  # first derivatives BULK
    dp_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    ur_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    ur_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))

    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):

            # radial
            if n == 0:
                # NOTE: Symmetry BC
                # dp_dr[m, n] = (p[m, n+2] - p[m, n])/(4*dr)
                # ur_dr[m, n] = (ur[m, n+2] + ur[m, n])/(4*dr)
                dp_dr[m, n] = 0.
                ur_dr[m, n] = 2*ur[m, n]/dr

            else:
                # elif n == Nr:
                dp_dr[m, n] = (p[m, n] - p[m, n-1])/(r[n]-r[n-1])  # BWD
                ur_dr[m, n] = (ur[m, n] - ur[m, n-1])/(r[n]-r[n-1])  # BWD

            # else:
            #     dp_dr[m, n] = (p[m, n+1] - p[m, n])/dr  # upwind
            #     ur_dr[m, n] = (ur[m, n+1] - ur[m, n])/dr  # upwind

            # elif (n != 1 and n != Nr-1):
            #     dp_dr = (p[m, n+1] - p[m, n-1])/(2*dr)  # CD
            #     ur_dr = (ur[m, n+1] - ur[m, n-1])/(2*dr)

            # if m == 0:
            #     ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

# axial
            if m == 0:
                ur_dx[m, n] = (ur[m+1, n] - ur[m, n])/dx  # upwind
            else:
                ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/dx  # BWD
                # ur_dx[m, n] = (ur[m+1, n] - ur[m, n])/(dx)  # upwind

            # elif (m <= n_trans+2 and m >= n_trans-2):
            #     ur_dx = (ur1[m-2, n] - 8*ur1[m-1, n] + 8 *
            #              ur1[m+1, n] - ur1[m+2, n])/(12*dx)  # 4 point CD

            # elif (m > 1 and m <= Nx - 2):
            #     # upwind 1st order  - positive flow - advection
            #     ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/dx

            # else:
            #     # upwind 1st order  - positive flow - advection
            #     ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/(dx)  # CD
            # dp_dr[:, :] = 0.
            # ur_dx[:, :] = 0.
            # ur_dr[:, :] = 0.
            return dp_dr, ur_dx, ur_dr


def grad_e2_matrix(v, u, e, r):     # use upwind for Pe > 2
    grad_r = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    grad_x = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if n == 0:
                # NOTE: Symmetry BC
                # grad_r[m, n] = ((N[m, n+2]*dr)*v[m, n+2]*e[m, n+2] +
                #                 (N[m, n]*dr)*v[m, n]*e[m, n])/(4*dr)
                grad_r[m, n] = 2*r[n]*v[m, n]*e[m, n]/dr

# surface case
            else:
                # if n == Nr:
                grad_r[m, n] = (r[n]*v[m, n]*e[m, n] -
                                r[n-1]*v[m, n-1]*e[m, n-1])/(r[n]-r[n-1])  # BWD

# n == Nr-1:
            # else:
            #     grad_r[m, n] = ((N[m, n+1]*dr)*v[m, n+1]*e[m, n+1] -
            #                     (N[m, n]*dr)*v[m, n]*e[m, n])/dr  # upwind

            # if m == 0:
            #     grad_x = (e1[m+1, n]*ux1[m+1, n]-e_in*ux_in)/(dx)
            if m == 0:
                # if m == Nx:
                # print("e1[m, n]*ux1[m, n]: ", e1[m, n]*ux1[m, n],
                #       "-e1[m-1, n]*ux1[m-1, n]: ", -e1[m-1, n]*ux1[m-1, n])
                grad_x[m, n] = (e[m+1, n]*u[m+1, n]-e[m, n]
                                * u[m, n])/dx  # FD

            else:  # 0 < m < Nx,  1 < n < Nr
                grad_x[m, n] = (e[m, n]*u[m, n]-e[m-1, n]
                                * u[m-1, n])/dx  # upwind

            # elif (m <= n_trans+2 and m >= n_trans-2):
            #     grad_x = (e1[m-2, n]*ux1[m-2, n] - 8*e1[m-1, n]*ux1[m-1, n] + 8 *
            #               e1[m+1, n]*ux1[m+1, n] - e1[m+2, n]*ux1[m+2, n])/(12*dx)
            # elif (m >= 1 and m <= Nx - 2):
            #     # upwind 1st order  - positive flow - advection
            #     grad_x[m, n] = (e[m, n]*u[m, n]-e[m-1, n]*u[m-1, n])/dx
                # grad_x = 3*(e1[m, n]*ux1[m, n]) - 4*(e1[m-1, n]
                #                                      * ux1[m-1, n]) + (e1[m-2, n]
                #                                                        * ux1[m-2, n]) / dx  # upwind, captures shocks
    # grad_x[:, :] = 0.
    # grad_r[:, :] = 0.
    # print(grad_r)
    return grad_x, grad_r


def dt2x_matrix(u, v):
    dt2x_ux = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    dt2x_ur = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if m == 0:
                # dt2nd axial ux1
                dt2x_ux[m, n] = (u[m+2, n] - 2.*u[m+1, n] + u[m, n]) / dx**2.

# dt2nd axial ur1
                dt2x_ur[m, n] = (v[m+2, n] - 2.*v[m+1, n] + v[m, n])/dx**2.

            else:
                # elif m == Nx:
                # dt2nd axial ux1
                dt2x_ux[m, n] = (u[m-2, n] - 2.*u[m-1, n] +
                                 u[m, n])/(dx**2.)  # BWD
# dt2nd axial ur1
# Three-point BWD
                dt2x_ur[m, n] = (
                    v[m-2, n] - 2.*v[m-1, n] + v[m, n])/(dx**2.)

#             else:
#                 # dt2nd axial ux1
#                 dt2x_ux[m, n] = (u[m+1, n] + u[m-1, n] -
#                                  2.*u[m, n])/(dx**2.)  # CD

# # dt2nd axial ur1
#                 dt2x_ur[m, n] = (v[m+1, n] + v[m-1, n] -
#                                  2.*v[m, n])/(dx**2.)  # CD
    # dt2x_ux[:, :] = 0.
    # dt2x_ur[:, :] = 0.
    # save_dt2x_matrix(dt2x_ux, dt2x_ur)
    return dt2x_ux, dt2x_ur


def dt2r_matrix(u, v, r):
    dt2r_ux = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    dt2r_ur = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):

            if n == 0:
                # NOTE: Symmetry Boundary Condition
                # dt2nd radial ux1
                # dt2r_ux[m, n] = (u[m, n+2] - u[m, n]) / (4.*dr**2.)
                dt2r_ux[m, n] = (-u[m, n] + u[m, n+1])/(dr**2.)  # BWD
# dt2nd axial ur
# dt2nd radial ur1
                # dt2r_ur[m, n] = (v[m, n+2] - 3.*v[m, n]) / (4.*dr**2.)
                dt2r_ur[m, n] = (-u[m, n+1] + 3*u[m, n])/(dr**2.)  # BWD

                # print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
                # print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

            # elif n == Nr:
            else:
                # dt2nd radial u1
                # NOTE: CHECK
                dt2r_ux[m, n] = (u[m, n] - 2*u[m, n-1] +
                                 u[m, n-2]) / (r[n]-r[n-1])**2.

                # dt2nd radial ur1
                dt2r_ur[m, n] = (v[m, n] - 2*v[m, n-1] +
                                 v[m, n-2]) / (r[n]-r[n-1])**2.

            # else:  # (n is between 1 and Nr)

            #     # dt2nd radial ux1
            #     dt2r_ux[m, n] = (u[m, n+1] + u[m, n-1] -
            #                      2.*u[m, n])/(dr**2.)  # CD
            # # dt2nd radial ur1
            #     dt2r_ur[m, n] = (v[m, n+1] + v[m, n-1] -
            #                      2.*v[m, n])/(dr**2.)  # CD
    # dt2r_ux[:, :] = 0.
    # dt2r_ur[:, :] = 0.
    return dt2r_ux, dt2r_ur


# def dt2nd_radial(ux1, ur1, m, n):
#     if n == 1:
#         # NOTE: Symmetry Boundary Condition

#         # dt2nd radial ux1
#         dt2nd_radial_ux1 = (ux1[m, n+2] - ux1[m, n]) / (4*dr**2)

# # dt2nd radial ur1
#         dt2nd_radial_ur1 = (ur1[m, n+2] - 3 * ur1[m, n]) / (4*dr**2)

# # print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
# # print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

#     else:  # (n is between 1 and Nr)

#         # dt2nd radial ux1
#         dt2nd_radial_ux1 = (ux1[m, n+1] + ux1[m, n-1] -
#                             2*ux1[m, n])/(dr**2)  # CD
# # dt2nd radial ur1
#         dt2nd_radial_ur1 = (ur1[m, n+1] + ur1[m, n-1] -
#                             2*ur1[m, n])/(dr**2)  # CD
# # print("dt2nd_radial_ur1:", dt2nd_radial_ur1)
#     dt2nd_radial_ux1[:, :] = 0.
#     dt2nd_radial_ur1[:, :] = 0.
#     return dt2nd_radial_ux1, dt2nd_radial_ur1


# # @jit(nopython=True)
# def dt2nd_axial(ux_in, ur_in, ux1, ur1, m, n):
#     if m == 0:
#         # --------------------------- dt2nd axial ux1 ---------------------------------#
#         dt2nd_axial_ux1 = (ux_in - 2*ux1[m, n] + ux1[m+1, n]) / (dx**2)
#         # dt2nd_axial_ux1 = (ux1[m+2,n] -2*ux1[m+1,n] + ux1[m,n])/(dx**2) #FWD

#     # --------------------------- dt2nd axial ur1 ---------------------------------#
#         #                        dt2nd_axial_ur1 = (ur1[m+2,n] -2*ur1[m+1,n] + ur1[m,n])/(dx**2) #FWD
#         # FWD
#         dt2nd_axial_ur1 = (-ur_in + ur_in - 30 *
#                            ur1[m, n] + 16*ur1[m+1, n] - ur1[m+2, n])/(12*dx**2)
#         # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)
#  #                        dt2nd_axial_ur1 = (2*ur1[m,n] - 5*ur1[m+1,n] + 4*ur1[m+2,n] -ur1[m+3,n])/(dx**3)  # FWD

#     elif m == Nx:
#         # --------------------------- dt2nd axial ux1 ---------------------------------#

#         dt2nd_axial_ux1 = (ux1[m-2, n] - 2*ux1[m-1, n] +
#                            ux1[m, n])/(dx**2)  # BWD
#     # dt2nd_axial_ux1 = (2*ux1[m,n] - 5*ux1[m-1,n] + 4*ux1[m-2,n] -ux1[m-3,n])/(dx**3) # BWD
#         # --------------------------- dt2nd axial ur1 ---------------------------------#
#     # Three-point BWD
#         dt2nd_axial_ur1 = (ur1[m-2, n] - 2*ur1[m-1, n] + ur1[m, n])/(dx**2)
#         # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

#     else:
#         # --------------------------- dt2nd axial ux1 ---------------------------------#
#         dt2nd_axial_ux1 = (ux1[m+1, n] + ux1[m-1, n] -
#                            2*ux1[m, n])/(dx**2)  # CD

#     # --------------------------- dt2nd axial ur1 ---------------------------------#
#         dt2nd_axial_ur1 = (ur1[m+1, n] + ur1[m-1, n] -
#                            2*ur1[m, n])/(dx**2)  # CD
#         # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

#     return dt2nd_axial_ux1, dt2nd_axial_ur1

#
def continue_simulation(Nx):
    # if os.path.exists(pathname):
    # change Working directory
    rho = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
    p = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
    T = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
    u = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
    v = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
    Ut = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
    e = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))

    rho1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    p1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    T1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    u1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    v1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    Ut1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    e1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))

# define field variables
    rho = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\rho.csv",
                     delimiter=",", dtype=np.longdouble)
    p = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\p.csv",
                   delimiter=",", dtype=np.longdouble)
    T = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\Tg.csv",
                   delimiter=",", dtype=np.longdouble)
    u = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\ux.csv",
                   delimiter=",", dtype=np.longdouble)
    v = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\ur.csv",
                   delimiter=",", dtype=np.longdouble)
    Ut[:, :] = np.sqrt(u[:, :]**2. + v[:, :]**2.)
    e = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\e.csv",
                   delimiter=",", dtype=np.longdouble)

    plot_imshow(p, u, T, rho, e)

# append N =0
    b = np.full((Nx+1, 1), 100, dtype=np.longdouble)  # Density

    rho1 = np.hstack((b, rho))
    p1 = np.hstack((b, p))
    T1 = np.hstack((b, T))
    u1 = np.hstack((b, u))
    v1 = np.hstack((b, v))
    Ut1 = np.hstack((b, Ut))
    e1 = np.hstack((b, e))

    if np.any(T1 < 0):
        print("Temp Array has at least one negative value")
        exit()
    if np.any(e1 < 0):
        print("Energy has at least one negative value")
        exit()
    if np.any(rho1 < 0):
        print("Density has at least one negative value")
        exit()

    return rho1, p1, T1, u1, v1, Ut1, e1


def mu_n(T, P):
    #   Calculate viscosity of nitrogen (Pa*s)
    # print("viscosity temp and pressure", T, P)
    if T == 0:
        T = 0.0001
    tao = 126.192/T
    if tao == 0:
        tao = 0.0001
    if P == 0:
        P = 0.0001
    delta = 1/(3395800/P/tao)
    if T == 98.94:
        T = 98.95
#    print("T value in viscosity function", T)
    omega = np.exp(0.431*np.log(T/98.94)**0-0.4623*np.log(T/98.94)**1 +
                   0.08406*np.log(T/98.94)**2+0.005341*np.log(T/98.94)**3 -
                   0.00331*np.log(T/98.94)**4)
#    print("omega", omega)
    mu_n_2 = 0.0266958*np.sqrt(28.01348*T)/0.3656**2/omega
    mu_n_1 = 10.72*tao**0.1*delta**2*np.exp(-0*delta**0) +\
        0.03989*tao**0.25*delta**10*np.exp(-1*delta**1) +\
        0.001208*tao**3.2*delta**12*np.exp(-1*delta**1) -\
        7.402*tao**0.9*delta**2*np.exp(-1*delta**2) +\
        4.620*tao**0.3*delta**1*np.exp(-1*delta**3)
#    print("viscosity from function:", (mu_n_1+mu_n_2)/1e6)
    # mu_n_2 = 0
    # mu_n_1 = 0
    # print("viscosity from function:", (mu_n_1+mu_n_2)/1e6)

    return (mu_n_1+mu_n_2)/1e6


def exp_smooth(grid, hv, lv, order, tran):  # Q: from where did we get this?
    #   Exponential smooth from hv to lv
    coe = ((hv+lv)/2-lv)/(np.exp(order*tran)-1)
    c = lv-coe
    if grid < 0:
        s_result = hv
    elif grid < (tran+1):
        s_result = -coe*np.exp(order*(grid))-c+hv+lv
    elif grid < (2*tran+1):
        s_result = coe*np.exp(order*(2*tran-grid))+c
    else:
        s_result = lv
    return s_result

# This is Nr, delete R =0 point done


def bulk_values(T_s):
    # T_0 = 100.
    T_0 = 4.2  # K
    rho_0 = 1e-2  # An arbitrary small initial density in pipe, kg/m3
    # T_0 = p_0/rho_0/R*M_n
    p_0 = rho_0*R/M_n*T_0
    u_0 = 0
    v_0 = 0

    Ut_0 = np.sqrt(u_0**2. + v_0**2.)
# energy bulk
    e_0 = 5./2.*p_0 + np.sqrt(u_0**2. + v_0**2.)  # Initial internal energy

    # print("p_0: ", p_0, "T_0:", T_0, "rho_0: ", rho_0, "e_0: ", e_0)
    return T_0, rho_0, p_0, e_0, Ut_0, u_0, v_0


def parabolic_velocity(rho, tg, u, v, Ut, e, u_in, v_in):
    # for i in np.arange(n_trans):
    # diatomic gas gamma = 7/5   WE USED ANY POINT, since this preparation area is constant along R direction.
    # any temperature works, they are equl in the radial direction
    # v_max = np.sqrt(7./5.*R*tg[0, 1]/M_n)
    v_max = np.sqrt(u_in**2. + v_in**2.)
    for i in np.arange(n_trans+1):
        for y in np.arange(Nr+1):

            # a = v_max
            # a = u_in
            u[i, y] = v_max*(1.0 - ((y*dr)/R_cyl)**2)
            # print("parabolic y", y)
            Ut[i, y] = u[i, y]

# No slip
    # u[:, Nr] = 0
    # Ut[:, Nr] = 0

    u = smooth_parabolic(u, n_trans)

    e = 5./2.*rho*R/M_n*tg + 1./2. * rho*Ut**2.

    out = u, v, Ut, e
    return out


def smooth_parabolic(ux, n_trans):
    for j in range(0, Nx+1):
        ux[j, :] = exp_smooth(j + n_trans, ux[j, :]*2, 0, 0.4, n_trans)
        # v_max = np.sqrt(7./5.*R*T/M_n)  # diatomic gas gamma = 7/5

        # if i < n_trans+1:
        #     e1[i, :] = 5./2.*p1[i, :]+1./2.*rho1[i, :]*u1[i, :]**2

    #        rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]  # IDEAL GAS LAW

        # print("p1 matrix after smoothing", p1)
        # else:
        #     e1[i, :] = 5/2*rho1[i, :]/M_n*R*T_in+1/2**rho1[i, :]*u1[i, :]**2
    # for i in range(0, Nx+1):
    return ux


# def smoothing_inlet(p, rho, T, p_in, p_0, rho_in, rho_0, n_trans):
#     for i in range(0, Nx+1):
#         p[i, :] = exp_smooth(i+n_trans, p_in*2.-p_0, p_0, 0.4, n_trans)
#     # print("P1 smoothing values", p1[i,:])
#         rho[i, :] = exp_smooth(i + n_trans, rho_in*2 -
#                                rho_0, rho_0, 0.4, n_trans)
#     #    T1[i, :] = T_neck(i)
#         # if i<51: T1[i]=T_in
#         T[i, :] = p[i, :]/rho[i, :]/R*M_n
#         # v_max = np.sqrt(7./5.*R*T/M_n)  # diatomic gas gamma = 7/5
#     #    u1[i, :] = exp_smooth(i + n_trans, ux_in*2, 0, 0.4, n_trans)

#         # if i < n_trans+1:
#         #     e1[i, :] = 5./2.*p1[i, :]+1./2.*rho1[i, :]*u1[i, :]**2

#     #        rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]  # IDEAL GAS LAW

#         # print("p1 matrix after smoothing", p1)
#         # else:
#         #     e1[i, :] = 5/2*rho1[i, :]/M_n*R*T_in+1/2**rho1[i, :]*u1[i, :]**2
#     # for i in range(0, Nx+1):
#     out = p, rho, T
#     return out

def remove_timestepping():
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "timestepping"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/Rk3/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "RK3"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/second_gradients/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "second_gradients"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/gradients/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "gradients"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "initial_conditions"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/m_dot/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "m_dot"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/fields/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "fields"
        path = os.path.join(location, dir)
        shutil.rmtree(path)


def no_slip_no_mdot(p, rho, tg, u, v, Ut, e):
    # no mass deposition
    u[:, Nr] = 0
    v[:, Nr] = 0
    Ut[:, Nr] = 0
    # energy assumed constant
    tg = (e - 1./2.*rho*Ut**2) * 2./5. / rho/R*M_n
    p = rho*R/M_n*tg
    return p, tg, u, v, Ut, e


def inlet_BC(u, v, Ut, p, rho, T, e):
    p_in, u_in, v_in, rho_in, e_in, T_in = val_in_constant()

    p[0, :] = p_in
    rho[0, :] = rho_in
    T[0, :] = T_in
    e[0, :] = e_in
    u[0, :] = u_in
    v[0, :] = v_in


# no slip
    u[:, Nr] = 0
    v[:, Nr] = 0
    Ut[:, Nr] = 0
    Ut = np.sqrt(u**2. + v**2.)
    e = 5./2. * p + 1./2 * rho*Ut**2
    return [u, v, Ut, p, rho, T, e]


def outlet_BC(p, e, rho, u, v, Ut, rho_0):

    for n in np.arange(Nr+1):
        p[Nx, n] = 2/5*(e[Nx, n]-1/2*rho[Nx, n]
                        * Ut[Nx, n]**2)  # Pressure

        rho[Nx, n] = max(2*rho[Nx-1, n]-rho[Nx-2, n], rho_0)  # Free outflow
        u[Nx, n] = max(2*rho[Nx-1, n]*u[Nx-1, n] -
                       rho[Nx-2, n]*u[Nx-2, n], 0) / rho[Nx, n]
        u = np.sqrt(u**2. + v**2.)
    # u[:, Nr] = 0  # no slip
    # v[:, Nr] = 0
    Ut = np.sqrt(u**2. + v**2.)
    # e[Nx, n] = 2*e[Nx-1, n]-e[Nx-2, n]
    e = 5./2. * p + 1./2 * rho*Ut**2
    tg = p/rho/R*M_n
    bc = [p, rho, tg,  u, Ut, e]
    return bc


def val_in_constant():
    #   Calculate instant flow rate (kg/s)
    p_in = 6000  # Pa
    # rho_in = 0.5
    T_in = 298.
    rho_in = p_in/R*M_n / T_in
    u_in = np.sqrt(gamma_n2*R/M_n*T_in)
    # u_in = 0.
    v_in = 0.
    Ut_in = np.sqrt(u_in**2. + v_in**2.)
    # Ut_in = 50.
    # Ut_in = np.sqrt(u_in**2 + v_in**2)
    e_in = 5./2.*rho_in/M_n*R*T_in + 1./2.*rho_in*Ut_in**2
    return p_in, u_in, v_in, rho_in, e_in, T_in


def save_initial_conditions(rho1, ux1, ur1, u1, e1, T1, de0, p1, de1):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "initial_conditions"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    os.chdir(pathname)
    np.savetxt("rho.csv", rho1, delimiter=",")
    np.savetxt("Tg.csv", T1, delimiter=",")
    np.savetxt("u.csv", u1, delimiter=",")
    np.savetxt("ux.csv", ux1, delimiter=",")
    np.savetxt("ur.csv", ur1, delimiter=",")
    np.savetxt("e.csv", e1, delimiter=",")
    np.savetxt("de.csv", de0, delimiter=",")
    np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")


def save_data(tx, dt, rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1):
    if tx % 50 == 0 and tx > 1:
        pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/'
        if os.path.exists(pathname):
            location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
            dir = "timestepping"
            path = os.path.join(location, dir)
            shutil.rmtree(path)

    increment = (tx+1)*dt

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.9f}".format(increment) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("rho.csv", rho1, delimiter=",")
    np.savetxt("Tg.csv", T1, delimiter=",")
    np.savetxt("u.csv", u1, delimiter=",")
    np.savetxt("ux.csv", ux1, delimiter=",")
    np.savetxt("ur.csv", ur1, delimiter=",")
    np.savetxt("e.csv", e1, delimiter=",")
    np.savetxt("tw.csv", Tw1, delimiter=",")
    np.savetxt("ts.csv", Ts1, delimiter=",")
    np.savetxt("de_mass.csv", de0, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")


def save_last(tx, dt, rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/last_timestep/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("rho.csv", rho1, delimiter=",")
    np.savetxt("Tg.csv", T1, delimiter=",")
    np.savetxt("u.csv", u1, delimiter=",")
    np.savetxt("ux.csv", ux1, delimiter=",")
    np.savetxt("ur.csv", ur1, delimiter=",")
    np.savetxt("e.csv", e1, delimiter=",")
    np.savetxt("tw.csv", Tw1, delimiter=",")
    np.savetxt("ts.csv", Ts1, delimiter=",")
    np.savetxt("de_mass.csv", de0, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")


# def delete_r0_point(rho, u, v, Ut, e, T, p):
#     rho = np.delete(rho, 0, axis=1)
#     u = np.delete(u, 0, axis=1)
#     v = np.delete(v, 0, axis=1)
#     Ut = np.delete(Ut, 0, axis=1)
#     e = np.delete(e, 0, axis=1)
#     T = np.delete(T, 0, axis=1)
#     p = np.delete(p, 0, axis=1)
#     return [rho, u, v, Ut, e, T, p]
