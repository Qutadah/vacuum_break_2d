# ----------------- Helper Functions --------------------------------#

import warnings
import copy
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import re
import inspect
from my_constants import *
import os
import shutil
import numba
from numba import jit
# import vtk
import numpy as np
# import vtk.util.numpy_support as numpy_support
# from scipy.ndimage.filters import laplace
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def dt2nd_wall(m, Tw1, T_in):

    if m == 0:
        dt2nd = (T_in - 2 * Tw1[m] +
                 Tw1[m+1])/(dx**2)  # 3-point CD
#       dt2nd = Tw1[m+1]-Tw1[m]-Tw1[m-1]+T_in
    elif m == Nx:
        # print("m=Nx", m)
        dt2nd = (-Tw1[m-3] + 4*Tw1[m-2] - 5*Tw1[m-1] +
                 2*Tw1[m]) / (dx**2)  # Four point BWD
    else:
        dt2nd = Tw1[m-1]-2*Tw1[m]+Tw1[m+1]/(dx**2)
    return dt2nd


# @jit(nopython=True)
def dt2nd_w_matrix(Tw, T_in):
    dt2nd_w_m = np.zeros((Nx+1), dtype=(np.float64))

    for m in np.arange(Nx+1):
        if m == 0:
            dt2nd_w_m[m] = (T_in - 2 * Tw[m] +
                            Tw[m+1])/(dx**2)  # 3-point CD
    #       dt2nd = Tw1[m+1]-Tw1[m]-Tw1[m-1]+T_in
        elif m == Nx:
            # print("m=Nx", m)
            dt2nd_w_m[m] = (-Tw[m-3] + 4*Tw[m-2] - 5*Tw[m-1] +
                            2*Tw[m]) / (dx**2)  # Four point BWD
        else:
            dt2nd_w_m[m] = Tw[m-1]-2*Tw[m]+Tw[m+1]/(dx**2)
    return dt2nd_w_m


# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8)')
def initialize_grid(p_0, rho_0, e_0, T_0, T_s):

    # rho12 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
    p1 = np.full((Nx+1, Nr+1), p_0,
                 dtype=(np.longdouble, np.longdouble))  # Pressure
    rho1 = np.full((Nx+1, Nr+1), rho_0,
                   dtype=(np.longdouble, np.longdouble))  # Density
    ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble,
                   np.longdouble))  # velocity -x
    ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble,
                   np.longdouble))  # velocity -r
    u1 = np.sqrt(np.square(ux1) + np.square(ur1))  # total velocity
    # Internal energy
    e1 = np.full((Nx+1, Nr+1), e_0, dtype=(np.longdouble, np.longdouble))
    # CHECK TODO: calculate using equation velocity.
    # TODO: calculate using equation velocity.

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
    rho3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    ux3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    ur3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    u3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    e3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    T3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))
    p3 = np.full((Nx+1, Nr), T_s, dtype=(np.longdouble, np.longdouble))

    # Dimensionless number in grid:
    Pe = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                  np.float64))  # Peclet number
    Pe1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                   np.float64))  # Peclet number
    out = [p1, rho1, ux1, ur1, u1, e1, T1, rho2, ux2, ur2, u2, e2, T2, p2,
           Tw1, Tw2, Ts1, Ts2, Tc1, Tc2, de0, de1, rho3, ux3, ur3, u3, e3, T3, p3]
    return out


# def rhs_rho(m, n, dr, dx, ur, ux, rho, a, ux_in, rho_in):

def n_matrix():
    # Initialized once when starting main
    n = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for i in np.arange(np.int64(0), np.int64(Nx+1)):
        for j in np.arange(np.int64(1), np.int64(Nr+1)):
            n[i, j] = j
    # print("Removing N=0 in matrix")
    n[:, 0] = 1
    # n[0, :] = 1
    return n


# def m_matrix():
#     # Initialized once when starting main
#     mm = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
#     for i in np.arange(np.int64(1), np.int64(Nx+1)):
#         for j in np.arange(np.int64(1), np.int64(Nr+1)):
#             mm[i, j] = j
#     return mm

def viscous_matrix_water():
    visc_matrix = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    visc_matrix[:, :] = 8.90e-4
    return visc_matrix


# returns viscosity matrix
def viscous_matrix(T, P):
    visc_matrix = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            visc_matrix[m, n] = mu_n(T[m, n], P[m, n])
    # save_visc(i, dt, visc_matrix)

# perform NAN value matrix checks:
    # print("performing finite check on visc_matrix")
    # print(visc_matrix)
    for x in np.arange(len(visc_matrix)):
        assert np.isfinite(visc_matrix).all()

# negative viscosity check
    if np.any(visc_matrix < 0):
        print("The viscous matrix has at least one negative value")
        exit()
    # visc_matrix[:, :] = 0.
    return visc_matrix


# return S term matrix, returns next mdot. input previous de
def source_mass_depo_matrix(rho_0, T, P, Ts1, rho, ux, ur, de, N):  # -4/D* mdot
    dm = np.zeros((Nx+1), dtype=(np.float64))
    dm_r = np.zeros((Nx+1), dtype=(np.float64))
    de3 = np.zeros((Nx+1), dtype=(np.float64))
    S = np.zeros((Nx+1), dtype=(np.float64))
    # chosen at gridpoint Nr-1 because at Nr, ux =0
    for m in np.arange(Nx+1):
        if m == Nx:
            dm[m] = rho[m, Nr-1]*ux[m, Nr-1] - rho[m-1, Nr-1]*ux[m-1, Nr-1]
        else:
            dm[m] = rho[m+1, Nr-1]*ux[m+1, Nr-1] - rho[m, Nr-1]*ux[m, Nr-1]
    for m in np.arange(Nx+1):
        if m == Nx:
            dm_r[m] = rho[m, Nr] * Nr*dr*ur[m, Nr] - \
                rho[m, Nr-1]*(Nr-1)*dr*ur[m, Nr-1]
        else:
            dm_r[m] = rho[m, Nr] * Nr*dr*ur[m, Nr] - \
                rho[m, Nr-1]*(Nr-1)*dr*ur[m, Nr-1]
# skip m=0, not needed
    de3 = m_de(T, P, ur,  Ts1, de, dm)  # dm_r[m], ur[m, Nr], N)  # used BWD
    for m in np.arange(np.int64(1), np.int64(Nx+1)):
        if rho[m, Nr] > 2.*rho_0:
            de3[m] = 0.
    S = -4./D * de3     # 1d array
    S_out = [de3, S]
    print("de3", de3)
    return S_out

# returns continuity RHS matrix, including source term S


def rhs_rho(i, d_dr, m_dx, N, rho_r, rho_x, rhs_rho_term):  # include i to calculate terms

    # calculate source term
    rhs_rho = - 1/N/dr*d_dr - m_dx
    A = - 1/N/dr*d_dr
    B = -m_dx

    # np.concatenate(([rho_r], [A]), axis=0)
    # np.concatenate(([rho_x], [B]), axis=0)
    # np.concatenate(([rhs_rho_term], [rhs_rho]), axis=0)

    # rho_r[i, :, :] = A

    rho_r[1, :, :] = A
    rho_x[1, :, :] = B
    rhs_rho_term[1, :, :] = rhs_rho
    # np.append(rho_r, A, axis=0)
    # save_stack(rho_r)

    return rhs_rho, rho_r, rho_x, rhs_rho_term

# returns MOMENTUM RHS matrix


def rhs_ma(i, dp_dx, rho, dt2r_ux, N, ux_dr, dt2x_ux, ux, ux_dx, ur, dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term):

    # plotting constants
    A = -dp_dx/rho
    B = visc_matrix/rho * (
        dt2r_ux + 1/N/dr*ux_dr + dt2x_ux)
    C = -ux * ux_dx
    D = - ur*ux_dr

    rhs_ux = -dp_dx/rho + visc_matrix/rho * (
        dt2r_ux + 1/N/dr*ux_dr + dt2x_ux) - ux * ux_dx - ur*ux_dr

# plotting constants
    E = - dp_dr/rho
    F = visc_matrix/rho * \
        (- ur/(dr**2*N**2) + 1/N/dr*ur_dr +
         dt2r_ur + dt2x_ur)
    G = - ux * ur_dx
    H = - ur*ur_dr

    rhs_ur = - dp_dr/rho + visc_matrix/rho * \
        (- ur/(dr**2.*N**2.) + 1/N/dr*ur_dr +
         dt2r_ur + dt2x_ur) - ux * ur_dx - ur*ur_dr
    # surface equations
    # no momentum equations radial velocity 0 will be applied in the BCs after solving
    # np.concatenate(([pressure_x], [A]), axis=0)
    # np.concatenate(([visc_x], [B]), axis=0)
    # np.concatenate(([ux_x], [C]), axis=0)
    # np.concatenate(([ur_x], [D]), axis=0)
    # np.concatenate(([rhs_ux_term], [rhs_ux]), axis=0)

    # np.concatenate(([pressure_r], [E]), axis=0)
    # np.concatenate(([visc_r], [F]), axis=0)
    # np.concatenate(([ux_r], [G]), axis=0)
    # np.concatenate(([ur_r], [H]), axis=0)
    # np.concatenate(([rhs_ur_term], [rhs_ur]), axis=0)


# Concatenation into global matrix
    pressure_x[1, :, :] = A
    visc_x[1, :, :] = B
    ux_x[1, :, :] = C
    ur_x[1, :, :] = D
    rhs_ux_term[1, :, :] = rhs_ux

    pressure_r[1, :, :] = E
    visc_r[1, :, :] = F
    ux_r[1, :, :] = G
    ur_r[1, :, :] = H
    rhs_ur_term[1, :, :] = rhs_ur

    return rhs_ux, rhs_ur, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term

# assures no division by zero


def no_division_zero(array):
    # ensure no division by zero
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if array[m, n] == 0:
                array[m, n] = 0.0001
    return array

# returns ENERGY RHS matrix including source terms


def rhs_energy(i, grad_r, grad_x, N, e_r, e_x, rhs_e_term):
    # S_e = np.zeros((Nx+1), dtype=(np.float64))
    rhs_e = - 1/N/dr*grad_r - grad_x
    A = -  1/N/dr*grad_r
    B = -grad_x
    # S_e[:] = S[:]*(5./2.*p[:, Nr]/rho[:, Nr] + 1./2.*u[:, Nr]**2)
    # rhs_e[:, Nr] = - 1/N[:, Nr]/dr*grad_r[:, Nr]
    # np.concatenate(([e_r], [A]), axis=0)
    # np.concatenate(([e_x], [B]), axis=0)
    # np.concatenate(([rhs_e_term], [rhs_e]), axis=0)

    e_r[1, :, :] = A
    e_x[1, :, :] = B
    rhs_e_term[1, :, :] = rhs_e
    return rhs_e, e_r, e_x, rhs_e_term
    # ri = rhsInv(nx,ny,nz,dx,dy,dz,q,iflx)
    # if (ivis==1)
    #     rv = rhsVis(nx,ny,nz,dx,dy,dz,q,Re)
    #     r  = ri + rv
    # else
    #     r  = ri
    # end
# nonconservative
    # Continuity equation

    # Momentum x
    # Momentum R
    # Energy


# Conservative

    # Continuity equation
    # Momentum x
    # Momentum R
    # Energy


# NOTE: NEXT
# simple time integration

# def euler_backward_time(q):
#     # RHS empty matrix initialization
#     rhs_rho1, rhs_ux1, rhs_ur1, rhs_e1 = rhs_matrix_initialization()
#     # nonconservative form - source term S included
#     rhs_rho1 = rhs_rho(d_dr, m_dx, N, S)

#     rhs_ux1, rhs_ur1 = rhs_ma(dp_dx, rho1, dt2r_ux, N, ux_dr, dt2x_ux,
#                               ux1, ux_dx, ur1, dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix)

#     rhs_e1 = rhs_energy(grad_r, grad_x, N, S)
#     rho2 = rho1 + dt * rhs_rho1

#     # calculate ux2
#     ux2 = ux1 + dt*rhs_ux1
#     # apply surface BC
#     ux2[:, Nr] = 0

# # calculate ur2
#     ur2 = ur1 + dt * rhs_ur1

#     de1 = -1*S*D/4.
# # apply surface ur2 calculation:
#     ur2[:, Nr] = de1[:] / rho2[:, Nr]

#     e2 = e1 + dt*rhs_e1

#     if np.any(e2 < 0):
#         print("The energy Array has at least one negative value")
#         exit()

#     # Update pressure
#     p2 = 2./5.*(e2 - 1./2.*rho2*ur2**2)

#     return q2

# This iterates RK3 for all equations

def rk4(ux, ur, u, p, q, tg, e, p_in, ux_in, rho_in, T_in, e_in, rho_0, ur_in, Rks):
    # create N matrix:
    N = n_matrix()

    # print("Deep copying initial matrices")

    qq = copy.deepcopy(q)  # density
    qn = copy.deepcopy(q)

    uxx = copy.deepcopy(q)  # velocity axial
    uxn = copy.deepcopy(q)

    urr = copy.deepcopy(q)  # velocity radial
    urn = copy.deepcopy(q)

    uu = copy.deepcopy(q)  # total velocity
    un = copy.deepcopy(q)

    ee = copy.deepcopy(q)  # energy
    en = copy.deepcopy(q)

    tt = copy.deepcopy(q)  # temperature
    tn = copy.deepcopy(q)

# # First step
# apply BCs
# l = [ux, ur, u, p, rho, T, e]
    p, tg, ux, ur, u, e = no_slip_no_mdot(p, q, tg, ux, ur, u, e)
    # plot_imshow(p, u, tg, q, e)
    ux, ur, u, p, q, tg, e = inlet_BC(
        ux, ur, u, p, q, tg, e, p_in, ux_in, rho_in, T_in, e_in)
    # plot_imshow(p, u, tg, q, e)
    p, q, tg, ux, u, e = outlet_BC(p, e, q, ux, ur, u, rho_0)
    # plot_imshow(p, u, tg, q, e)
    ux, u, e = parabolic_velocity(rho, tg, u, u_in, Ut, e)


# Calculating gradients (first and second)
    print("Calculating gradients for RK3 loop #", n)

# def grad_rho_matrix(ux_in, rho_in, ur, ux, rho):
    d_dr, m_dx = grad_rho_matrix(ur, ux, q)
# def grad_ux2_matrix(p_in, p, ux_in, ux):  # bulk
    dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p_in, p, ux_in, ux)
# def grad_ur2_matrix(p, ur, ur_in):  # first derivatives
    dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p, ur, ur_in)
# def grad_e2_matrix(ur1, ux1, ux_in, e_in, e1):     # use
    grad_x, grad_r = grad_e2_matrix(ur, ux, e)

# def dt2x_matrix(ux_in, ur_in, ux1, ur1):
    dt2x_ux, dt2x_ur = dt2x_matrix(ux_in, ur_in, ux, ur)
# def dt2r_matrix(ux1, ur1):
    dt2r_ux, dt2r_ur = dt2r_matrix(ux, ur)


# # Plot gradients with X
#     abb = [dp_dx, ux_dx, ur_dx, grad_x, dt2x_ux, dt2r_ux]
#     dpdx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
#     uxdx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
#     urdx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
#     gradx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
#     dt2xux = np.zeros((Nx+1), dtype=(np.float64))  # place holder
#     dt2rux = np.zeros((Nx+1), dtype=(np.float64))  # place holder

# # plotting and saving gradients
#         dpdx[:] = abb[0][:, Nr]
#         uxdx[:] = abb[1][:, Nr]
#         urdx[:] = abb[2][:, Nr]
#         gradx[:] = abb[3][:, Nr]
#         dt2xux[:] = abb[4][:, Nr]
#         dt2rux[:] = abb[5][:, Nr]

    # aa = 40
    # plt.figure()
    # x = np.linspace(0, aa, aa+1)
    # y1 = dpdx[0:aa+1]
    # y2 = uxdx[0:aa+1]
    # y3 = urdx[0:aa+1]
    # y4 = gradx[0:aa+1]
    # y5 = dt2xux[0:aa+1]
    # y6 = dt2rux[0:aa+1]

    # plt.plot(x, y1, color="black", label="dp_dx")
    # plt.plot(x, y2, color="blue", label="ux_dx")
    # plt.plot(x, y3, color="brown", label="ur_dx")
    # plt.plot(x, y4, color="yellow", label="grad_x")
    # plt.plot(x, y5, color="green", label="dt2x_ux")
    # plt.plot(x, y6, color="red", label="dt2r_ux")
    # plt.legend()
    # legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # plt.legend(["dp_dx", "ux_dx", "ur_dx", "grad_x",
    #            "dt2x_ux", "dt2r_ux"], loc="lower right")
    # plt.show()

# viscosity calculations
    print("Calculating viscosity for RK3 loop #", n)
    visc_matrix = viscous_matrix(tg, p)
    assert np.isfinite(visc_matrix).all()

    # de_variable (de1) matrix input.
    # This function takes into account density large enough to have mass deposition case.
    # print("Calculating Source term matrix")
    # if n == 0:
    # de_var = de
    # print("de_variable: ", de_variable)
    # S_out = [de3, S]
    # l = [ux, ur, u, p, rho, T, e]
    # S_out = source_mass_depo_matrix(
    #     rho_0, l[5], l[3], l[8], l[4], l[0], l[1], de_var, N)
    # This de1 is returned again, first calculation is the right one for this iteration. it takes last values.
    # de_var = S_out[0]
    # CALCULATING RHS USING LOOP VALUES
    print("Calculating RHS terms matrices")

    r = rhs_rho(d_dr, m_dx, N)
    r_ux, r_ur = rhs_ma(dp_dx, q, dt2r_ux, N, ux_dr, dt2x_ux, ux,
                        ux_dx, ur, dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix)
    r_e = rhs_energy(grad_r, grad_x, N)
    # print(r_u)
    return r, r_u, r_v, r_e


# def RK4():

# simple time integration

def simple_time(p, q, tg, u, v, Ut, e, p_in, rho_in, T_in, e_in, u_in, v_in, rho_0, rho_r, rho_x, rhs_rho_term, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term, e_r, e_x, rhs_e_term, i):
    N = n_matrix()
    q_0 = np.zeros((Nx+1, Nr+1), dtype=(np.float64))
    qq = copy.deepcopy(q_0)  # density
    uxx = copy.deepcopy(q_0)  # velocity axial
    urr = copy.deepcopy(q_0)  # velocity radial
    uu = copy.deepcopy(q_0)  # total velocity
    ee = copy.deepcopy(q_0)  # energy
    tt = copy.deepcopy(q_0)  # temperature
    # if np.any(tg < 0):
    #     print("Temp before no slip simple_time has at least one negative value")
    #     exit()

    # plot_imshow(p, u, tg, q, e)
    u, v, Ut, p, q, tg, e = inlet_BC(
        u, v, Ut, p, q, tg, e, p_in, u_in, rho_in, T_in, e_in)
    # negative temp check
    # if np.any(tg < 0):
    #     print("Temp inlet_BC has at least one negative value")
    #     exit()
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

# def grad_rho_matrix(ux_in, rho_in, ur, ux, rho):
    d_dr, m_dx = grad_rho_matrix(v, u, q, N)
# def grad_ux2_matrix(p_in, p, ux_in, ux):  # bulk
    dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p, u)
# def grad_ur2_matrix(p, ur, ur_in):  # first derivatives
    dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p, v)
# def grad_e2_matrix(ur1, ux1, ux_in, e_in, e1):     # use
    grad_x, grad_r = grad_e2_matrix(v, u, e, N)

# def dt2x_matrix(ux_in, ur_in, ux1, ur1):
    dt2x_ux, dt2x_ur = dt2x_matrix(u, v)
# def dt2r_matrix(ux1, ur1):
    dt2r_ux, dt2r_ur = dt2r_matrix(u, v)

    # print("Calculating viscosity")
    visc_matrix = viscous_matrix(tg, p)
    # print("zero viscosity assumed")
    # visc_matrix[:, :] = 0.

    assert np.isfinite(visc_matrix).all()

    r, rho_r, rho_x, rhs_rho_term = rhs_rho(i,
                                            d_dr, m_dx, N, rho_r, rho_x, rhs_rho_term)
    r_ux, r_ur, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term = rhs_ma(
        i, dp_dx, q, dt2r_ux, N, ux_dr, dt2x_ux, u, ux_dx, v, dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term)

    # p, tg, u, v, Ut, e = no_slip_no_mdot(p, q, tg, u, v, Ut, e)
    # grad_x, grad_r = grad_e2_matrix(v, u, e, N)

    r_e, e_r, e_x, rhs_e_term = rhs_energy(i,
                                           grad_r, grad_x, N, e_r, e_x, rhs_e_term)

    # apply inlet BCs

    # print("r", r, "r_ux", r_ux, "r_ur", r_ur, "r_e", r_e)
# first LHS calculations

# calculating ratios

    # print("ratio conti", dt*r/q)
    # print("ratio momentum X, iteration #", i)
    # print(dt*r_ux/u)
    # print("ratio momentum R", dt*r_ur/v)
    # print("ratio energy", dt*r_e/e)

    qq = q + dt*r
    qq[0, :] = q[0, :]
    uxx = u + dt*r_ux
    uxx[0, :] = u[0, :]
    urr = v + dt*r_ur
    urr[0, :] = v[0, :]
    ee = e + dt*r_e
    ee[0, :] = e[0, :]

# ensure no division by zero
    qq = no_division_zero(qq)

# Velocity
    uu = np.sqrt(uxx**2. + urr**2.)
# pressure defining
    tt = (ee - 1./2.*qq*uu**2.) * 2./5. / qq/R*M_n
    pp = qq*R/M_n*tt

    uxx, urr, uu, pp, qq, tt, ee = inlet_BC(
        uxx, urr, uu, pp, qq, tt, ee, p_in, u_in, rho_in, T_in, e_in)
# negative temp check
    # if np.any(tg < 0):
    #     print("Temp inlet_BC has at least one negative value")
    #     exit()
    # plot_imshow(p, u, tg, q, e)
    pp, qq, tt, uxx, uu, ee = outlet_BC(pp, ee, qq, uxx, urr, uu, rho_0)

    # uxx, urr, uu, ee = parabolic_velocity(qq, tg, uxx, urr, Ut, ee, u_in, v_in)

# # energy balance
#     tt = (ee - 1./2.*qq*uu**2) * 2./5. / qq/R*M_n
#     pp = qq*R/M_n*tt

# no slip condition - pressure and temp recalculated within
    pp, tt, uxx, urr, uu, ee = no_slip_no_mdot(pp, qq, tt, uxx, urr, uu, ee)
    # print("plotting no slip")
    # plot_imshow(pp, uu, tt, qq, ee)

    return pp, qq, tt, uxx, urr, uu, ee, rho_r, rho_x, rhs_rho_term, pressure_x, visc_x, ux_x, ur_x, rhs_ux_term, pressure_r, visc_r, ux_r, ur_r, rhs_ur_term, e_r, e_x, rhs_e_term


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


# def vtk_convert(rho3, ux3, ur3, u3, e3, T3, Tw3, Ts2, de0, p3, de1, Pe3):

# def numpyToVTK(data):
#     data_type = vtk.VTK_FLOAT
#     shape = data.shape

#     flat_data_array = data.flatten()
#     vtk_data = numpy_support.numpy_to_vtk(
#         num_array=flat_data_array, deep=True, array_type=data_type)

#     img = vtk.vtkImageData()
#     img.GetPointData().SetScalars(vtk_data)
#     img.SetDimensions(shape[0], shape[1], shape[2])
#     return img


def save_plots(i, p, ux, T, rho, e):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/fields/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    plot_imshow(p, ux, T, rho, e)
    # if os.path.exists(pathname):
    #     location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
    #     dir = "fields"
    #     path = os.path.join(location, dir)


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
    plt.show()
    # x = str(i) + '.png'
    # plt.savefig(x)


def init(p, ux, T, rho, e):
    line.imshow(p, ux, T, rho, e)
    return line


def animate_func(p, ux, T, rho, e):
    line.imshow(p, ux, T, rho, e)
    return line


def grad_rho_matrix(ur, ux, rho, N):
    # create gradients arrays.
    m_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    d_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    for i in np.arange(Nx+1):
        for j in np.arange(1, Nr+1):
            # if i == 0:
            #     # m_dx[i, j] = (rho[i, j]*ux[i, j]-rho_in*ux_in)/dx
            #     m_dx[i, j] = (rho[i+1, j]*ux[i+1, j]-rho[i, j]*ux[i, j])/dx
            # axial
            if i == Nx:
                m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]
                              * ux[i-1, j])/dx  # BWD

            else:
                m_dx[i, j] = (rho[i+1, j]*ux[i+1, j]-rho[i, j]*ux[i, j])/dx

# radial
            if j == 1:
                # NOTE: SYMMETRY BC
                d_dr[i, j] = (rho[i, j+2]*(N[i, j+2]*dr)*ur[i, j+2] +
                              rho[i, j] * (N[i, j]*dr) * ur[i, j]) / (4*dr)

            elif j == Nr:
                d_dr[i, j] = (rho[i, j]*(N[i, j]*dr)*ur[i, j] -
                              rho[i, j-1] * (N[i, j-1]*dr)*ur[i, j-1])/dr

            else:
                d_dr[i, j] = (rho[i, j+1]*(N[i, j+1]*dr)*ur[i, j+1] -
                              rho[i, j] * (N[i, j]*dr) * ur[i, j])/dr  # upwind
    # d_dr[:, :] = 0.
    # m_dx[:, :] = 0.
    return d_dr, m_dx


# @numba.jit('f8(f8,f8,f8,f8,f8,f8)')


# @jit(nopython=True)
def grad_ux2_matrix(p, ux):  # bulk
    ux_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    dp_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    ux_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))

    for m in np.arange(Nx+1):
        for n in np.arange(1, Nr+1):
            # axial
            if m == Nx:
                dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx
                ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx

            # elif m == Nx:
            #     dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx  # BWD
            #     ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx  # BWD

            # else:
                # upwind 1st order  - positive flow - advection
                # dp_dx[m, n] = (p[m+1, n] - p[m, n])/dx  # upwind
                # ux_dx[m, n] = (ux[m+1, n] - ux[m, n])/dx  # upwind
            elif m >= 1:
                dp_dx[m, n] = (p[m+1, n] - p[m-1, n])/(2*dx)  # upwind
                ux_dx[m, n] = (ux[m+1, n] - ux[m-1, n])/(2*dx)  # upwind
            else:
                dp_dx[m, n] = (p[m+1, n] - p[m, n])/dx  # upwind
                ux_dx[m, n] = (ux[m+1, n] - ux[m, n])/dx  # upwind

# radial
            if n == 1:
                # NOTE: SYMMETRY CONDITION
                ux_dr[m, n] = (ux[m, n+2] - ux[m, n])/(4*dr)

            elif n == Nr:
                ux_dr[m, n] = (ux[m, n] - ux[m, n-1])/dr  # BWD
            else:
                # upwind 1st order  - positive flow - advection
                ux_dr[m, n] = (ux[m, n+1] - ux[m, n])/dr  # upwind
    # ux_dr[:, :] = 0.
    # ux_dx[:, :] = 0.
    # dp_dx[:, :] = 0.
    # print(ux_dx)
    return dp_dx, ux_dx, ux_dr

# @numba.jit('f8(f8,f8,f8,f8,f8)')


def grad_ur2_matrix(p, ur):  # first derivatives BULK
    dp_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    ur_dr = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    ur_dx = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))

    for m in np.arange(Nx+1):
        for n in np.arange(1, Nr+1):

            # radial
            if n == 1:
                # NOTE: Symmetry BC
                dp_dr[m, n] = (p[m, n+2] - p[m, n])/(4*dr)
                ur_dr[m, n] = (ur[m, n+2] + ur[m, n])/(4*dr)

            elif n == Nr:
                dp_dr[m, n] = (p[m, n] - p[m, n-1])/dr  # BWD
                ur_dr[m, n] = (ur[m, n] - ur[m, n-1])/dr  # BWD

            else:
                dp_dr[m, n] = (p[m, n+1] - p[m, n])/dr  # upwind
                ur_dr[m, n] = (ur[m, n+1] - ur[m, n])/dr  # upwind

            # elif (n != 1 and n != Nr-1):
            #     dp_dr = (p[m, n+1] - p[m, n-1])/(2*dr)  # CD
            #     ur_dr = (ur[m, n+1] - ur[m, n-1])/(2*dr)

            # if m == 0:
            #     ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

# axial
            if m == Nx:
                ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/dx  # BWD
            else:
                ur_dx[m, n] = (ur[m+1, n] - ur[m, n])/(dx)  # upwind

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


def grad_e2_matrix(v, u, e, N):     # use upwind for Pe > 2
    grad_r = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    grad_x = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    for m in np.arange(Nx+1):
        for n in np.arange(1, Nr+1):
            if n == 1:
                # NOTE: Symmetry BC
                grad_r[m, n] = ((N[m, n+2]*dr)*v[m, n+2]*e[m, n+2] +
                                (N[m, n]*dr)*v[m, n]*e[m, n])/(4*dr)

# surface case
            if n == Nr:
                grad_r[m, n] = ((N[m, n]*dr)*v[m, n]*e[m, n] -
                                (N[m, n-1]*dr)*v[m, n-1]*e[m, n-1])/dr  # BWD

# n == Nr-1:
            else:
                grad_r[m, n] = ((N[m, n+1]*dr)*v[m, n+1]*e[m, n+1] -
                                (N[m, n]*dr)*v[m, n]*e[m, n])/dr  # upwind

            # if m == 0:
            #     grad_x = (e1[m+1, n]*ux1[m+1, n]-e_in*ux_in)/(dx)
            if m == Nx:
                # print("e1[m, n]*ux1[m, n]: ", e1[m, n]*ux1[m, n],
                #       "-e1[m-1, n]*ux1[m-1, n]: ", -e1[m-1, n]*ux1[m-1, n])
                grad_x[m, n] = (e[m, n]*u[m, n]-e[m-1, n]
                                * u[m-1, n])/dx  # BWD

            else:  # 0 < m < Nx,  1 < n < Nr
                grad_x[m, n] = (e[m+1, n]*u[m+1, n]-e[m, n]
                                * u[m, n])/dx  # upwind

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
        for n in np.arange(1, Nr+1):
            if m == 0:
                # dt2nd axial ux1
                dt2x_ux[m, n] = (u[m+2, n] - 2.*u[m+1, n] + u[m, n]) / dx**2.

# dt2nd axial ur1
                dt2x_ur[m, n] = (v[m+2, n] - 2.*v[m+1, n] + v[m, n])/dx**2.

            elif m == Nx:
                # dt2nd axial ux1

                dt2x_ux[m, n] = (u[m-2, n] - 2.*u[m-1, n] +
                                 u[m, n])/(dx**2.)  # BWD
# dt2nd axial ur1
# Three-point BWD
                dt2x_ur[m, n] = (
                    v[m-2, n] - 2.*v[m-1, n] + v[m, n])/(dx**2.)

            else:
                # dt2nd axial ux1
                dt2x_ux[m, n] = (u[m+1, n] + u[m-1, n] -
                                 2.*u[m, n])/(dx**2.)  # CD

# dt2nd axial ur1
                dt2x_ur[m, n] = (v[m+1, n] + v[m-1, n] -
                                 2.*v[m, n])/(dx**2.)  # CD
    # dt2x_ux[:, :] = 0.
    # dt2x_ur[:, :] = 0.
    save_dt2x_matrix(dt2x_ux, dt2x_ur)
    return dt2x_ux, dt2x_ur


def dt2r_matrix(u, v):
    dt2r_ux = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    dt2r_ur = np.zeros((Nx+1, Nr+1), dtype=(np.longdouble, np.longdouble))
    for m in np.arange(Nx+1):
        for n in np.arange(1, Nr+1):

            if n == 1:
                # NOTE: Symmetry Boundary Condition
                # dt2nd radial ux1
                dt2r_ux[m, n] = (u[m, n+2] - u[m, n]) / (4.*dr**2.)

# dt2nd radial ur1
                dt2r_ur[m, n] = (v[m, n+2] - 3.*v[m, n]) / (4.*dr**2.)

                # print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
                # print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

            elif n == Nr:
                # dt2nd radial u1
                # NOTE: CHECK
                dt2r_ux[m, n] = (u[m, n] - 2*u[m, n-1] + u[m, n-2]) / dr**2.

                # dt2nd radial ur1
                dt2r_ur[m, n] = (v[m, n] - 2*v[m, n-1] + v[m, n-2]) / dr**2.

            else:  # (n is between 1 and Nr)

                # dt2nd radial ux1
                dt2r_ux[m, n] = (u[m, n+1] + u[m, n-1] -
                                 2.*u[m, n])/(dr**2.)  # CD
            # dt2nd radial ur1
                dt2r_ur[m, n] = (v[m, n+1] + v[m, n-1] -
                                 2.*v[m, n])/(dr**2.)  # CD
    # dt2r_ux[:, :] = 0.
    # dt2r_ur[:, :] = 0.
    save_dt2r_matrix(dt2r_ux, dt2r_ur)
    return dt2r_ux, dt2r_ur


# @jit(nopython=True)
def dt2nd_radial(ux1, ur1, m, n):
    if n == 1:
        # NOTE: Symmetry Boundary Condition

        # dt2nd radial ux1
        dt2nd_radial_ux1 = (ux1[m, n+2] - ux1[m, n]) / (4*dr**2)

# dt2nd radial ur1
        dt2nd_radial_ur1 = (ur1[m, n+2] - 3 * ur1[m, n]) / (4*dr**2)

# print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
# print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

    else:  # (n is between 1 and Nr)

        # dt2nd radial ux1
        dt2nd_radial_ux1 = (ux1[m, n+1] + ux1[m, n-1] -
                            2*ux1[m, n])/(dr**2)  # CD
# dt2nd radial ur1
        dt2nd_radial_ur1 = (ur1[m, n+1] + ur1[m, n-1] -
                            2*ur1[m, n])/(dr**2)  # CD
# print("dt2nd_radial_ur1:", dt2nd_radial_ur1)
    dt2nd_radial_ux1[:, :] = 0.
    dt2nd_radial_ur1[:, :] = 0.
    return dt2nd_radial_ux1, dt2nd_radial_ur1


def save_dt2x_matrix(array1, array2):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/second_gradients/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("dt2x_ux1.csv", array1, delimiter=",")
    np.savetxt("dt2x_ur1.csv", array2, delimiter=",")
    return


def save_dt2r_matrix(array1, array2):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/second_gradients/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("dt2r_ux1.csv", array1, delimiter=",")
    np.savetxt("dt2r_ur1.csv", array2, delimiter=",")
    return


def save_gradients(array2, array3, array4, array5, array6, array7, array8, array9, array10, array11):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/gradients/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    # np.savetxt("a.csv", array1, delimiter=",")
    np.savetxt("d_dr.csv", array2, delimiter=",")
    np.savetxt("m_dx.csv", array3, delimiter=",")
    np.savetxt("dp_dx.csv", array4, delimiter=",")
    np.savetxt("ux_dx.csv", array5, delimiter=",")
    np.savetxt("ux_dr.csv", array6, delimiter=",")
    np.savetxt("dp_dr.csv", array7, delimiter=",")
    np.savetxt("ur_dx.csv", array8, delimiter=",")
    np.savetxt("ur_dr.csv", array9, delimiter=",")
    np.savetxt("grad_x.csv", array10, delimiter=",")
    np.savetxt("grad_r.csv", array11, delimiter=",")

# @jit(nopython=True)

# @jit(nopython=True)


def dt2nd_axial(ux_in, ur_in, ux1, ur1, m, n):
    if m == 0:
        # --------------------------- dt2nd axial ux1 ---------------------------------#
        dt2nd_axial_ux1 = (ux_in - 2*ux1[m, n] + ux1[m+1, n]) / (dx**2)
        # dt2nd_axial_ux1 = (ux1[m+2,n] -2*ux1[m+1,n] + ux1[m,n])/(dx**2) #FWD

    # --------------------------- dt2nd axial ur1 ---------------------------------#
        #                        dt2nd_axial_ur1 = (ur1[m+2,n] -2*ur1[m+1,n] + ur1[m,n])/(dx**2) #FWD
        # FWD
        dt2nd_axial_ur1 = (-ur_in + ur_in - 30 *
                           ur1[m, n] + 16*ur1[m+1, n] - ur1[m+2, n])/(12*dx**2)
        # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)
 #                        dt2nd_axial_ur1 = (2*ur1[m,n] - 5*ur1[m+1,n] + 4*ur1[m+2,n] -ur1[m+3,n])/(dx**3)  # FWD

    elif m == Nx:
        # --------------------------- dt2nd axial ux1 ---------------------------------#

        dt2nd_axial_ux1 = (ux1[m-2, n] - 2*ux1[m-1, n] +
                           ux1[m, n])/(dx**2)  # BWD
    # dt2nd_axial_ux1 = (2*ux1[m,n] - 5*ux1[m-1,n] + 4*ux1[m-2,n] -ux1[m-3,n])/(dx**3) # BWD
        # --------------------------- dt2nd axial ur1 ---------------------------------#
    # Three-point BWD
        dt2nd_axial_ur1 = (ur1[m-2, n] - 2*ur1[m-1, n] + ur1[m, n])/(dx**2)
        # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

    else:
        # --------------------------- dt2nd axial ux1 ---------------------------------#
        dt2nd_axial_ux1 = (ux1[m+1, n] + ux1[m-1, n] -
                           2*ux1[m, n])/(dx**2)  # CD

    # --------------------------- dt2nd axial ur1 ---------------------------------#
        dt2nd_axial_ur1 = (ur1[m+1, n] + ur1[m-1, n] -
                           2*ur1[m, n])/(dx**2)  # CD
        # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

    return dt2nd_axial_ux1, dt2nd_axial_ur1

#
# @jit(nopython=True)


# @jit(nopython=True)
def f_ps(ts):
    # Calculate saturated vapor pressure (Pa)
    ts = float(ts)
    if ts < 10.:
        p_sat = 12.4-807.4*10**(-1)-3926.*10**(-2)+62970. * \
            10**(-3)-463300.*10**(-4)+1325000.*10**(-5)
    elif ts < 35.6:
        p_sat = 12.4-807.4*ts**(-1)-3926.*ts**(-2)+62970. * \
            ts**(-3)-463300.*ts**(-4)+1325000.*ts**(-5)
    else:
        p_sat = 8.514-458.4*ts**(-1)-19870.*ts**(-2) + \
            480000.*ts**(-3)-4524000.*ts**(-4)
    p_sat = np.exp(p_sat)*100000.
    return p_sat


# @jit(nopython=True)
def f_ts(ps):
    # Calculate saturated vapor temperature (K)
    print("Ps for f_ts calc: ", ps)
    ps1 = np.log(ps/100000.0)
    t_sat = 74.87701+6.47033*ps1+0.45695*ps1**2+0.02276*ps1**3+7.72942E-4*ps1**4+1.77899E-5 * \
        ps1**5+2.72918E-7*ps1**6+2.67042E-9*ps1**7+1.50555E-11*ps1**8+3.71554E-14*ps1**9
    return t_sat


# @jit(nopython=True)
def delta_h(tg, ts):

    # Calculate sublimation heat of nitrogen (J/kg)  ## needed for thermal resistance of SN2 layer when thickness is larger than reset value.
    delta_h62 = 6775.0/0.028
    if ts > 35.6:
        h_s = 4696.25245*62.0-393.92323*62.0**2/2+17.11194*62.0**3/3-0.35784*62.0**4/4+0.00371*62.0**5/5-1.52168E-5*62.0**6/6 -\
            (4696.25245*ts-393.92323*ts**2/2+17.11194*ts**3/3 -
             0.35784*ts**4/4+0.00371*ts**5/5-1.52168E-5*ts**6/6)
    else:
        h_s = 4696.25245*62.0-393.92323*62.0**2/2+17.11194*62.0**3/3-0.35784*62.0**4/4+0.00371*62.0**5/5-1.52168E-5*62.0**6/6 -\
            (4696.25245*35.6-393.92323*35.6**2/2+17.11194*35.6**3/3-0.35784*35.6**4/4+0.00371*35.6**5/5-1.52168E-5*35.6**6/6) +\
            (-0.02633*35.6+4.72107*35.6**2/2-5.13485*35.6**3/3+1.53391*35.6**4/4-0.13279*35.6**5/5+0.00557*35.6**6/6-1.16225E-4*35.6**7/7+9.67937E-7*35.6**8/8) -\
            (-0.02633*ts+4.72107*ts**2/2-5.13485*ts**3/3+1.53391*ts**4/4 -
             0.13279*ts**5/5+0.00557*ts**6/6-1.16225E-4*ts**7/7+9.67937E-7*ts**8/8)
    h_g = (tg-62.0)*7.0/2.0*R/M_n
    dH = delta_h62+h_s+h_g
    return dH


# @jit(nopython=True)
def c_n(ts):
    #   Calculate specific heat of solid nitrogen (J/(kg*K))
    # print("Ts for c_n specific heat SN2 calc: ", ts)
    if ts > 35.6:
        cn = (4696.25245-393.92323*ts+17.11194*ts**2 -
              0.35784*ts**3+0.00371*ts**4-1.52168E-5*ts**5)
    else:
        cn = (-0.02633+4.72107*ts-5.13485*ts**2+1.53391*ts**3-0.13279 *
              ts**4+0.00557*ts**5-1.16225E-4*ts**6+9.67937E-7*ts**7)
    return cn


# @jit(nopython=True)
def v_m(tg):
    # Calculate arithmetic mean speed of gas molecules (m/s)
    print("Tg for v_m gas: ", tg)
    v_mean = np.sqrt(8.*R*tg/np.pi/M_n)
    # ipdb.set_trace()
    return v_mean


# def save_qhe(tx, dt, qhe):
#     incrementx = (tx+1)*dt
#     pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
#         "{:.4f}".format(incrementx) + '/'
#     newpath = pathname
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     os.chdir(pathname)
#     np.savetxt("qhe.csv", qhe, delimiter=",")


# def save_visc(tx, dt, array):
#     increment = (tx+1)*dt
#     pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
#         "{:.4f}".format(increment) + '/'
#     newpath = pathname
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     os.chdir(pathname)
#     np.savetxt("visc.csv", array, delimiter=",")


# def save_qdep(tx, dt, qdep):
#     incrementy = (tx+1)*dt
#     pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
#         "{:.4f}".format(incrementy) + '/'
#     newpath = pathname
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     os.chdir(pathname)
#     np.savetxt("q_dep1.csv", qdep, delimiter=",")


# def save_mdot():
#     pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
#         "{:.4f}".format(increment) + '/'
#     newpath = pathname
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     os.chdir(pathname)
#     np.savetxt("de1.csv", de1, delimiter=",")
#     return


# @jit(nopython=True)
def c_c(ts):
    #   Calculate the heat capacity of copper (J/(kg*K))
    # print("Ts for c_c (specific heat copper) calc: ", ts)
    #  print("ts",ts)
    c_copper = 1.22717-10.74168*np.log10(ts)**1+15.07169*np.log10(
        ts)**2-6.69438*np.log10(ts)**3+1.00666*np.log10(ts)**4-0.00673*np.log10(ts)**5
    c_copper = 10.**c_copper
    return c_copper


# @jit(nopython=True)
def k_cu(T):
    #   Calculate the coefficient of thermal conductivity of copper (RRR=10) (W/(m*K)) (for pde governing copper wall, heat conducted in the x axis.)
    # print("Tw for k_cu copper: ", T)
    k1 = 3.00849+11.34338*T+1.20937*T**2-0.044*T**3+3.81667E-4 * \
        T**4+2.98945E-6*T**5-6.47042E-8*T**6+2.80913E-10*T**7
    k2 = 1217.49161-13.76657*T-0.01295*T**2+0.00188*T**3-1.77578E-5 * \
        T**4+7.58474E-8*T**5-1.58409E-10*T**6+1.31219E-13*T**7
    k3 = k2+(k1-k2)/(1+np.exp((T-70)/1))
    return k3


# @jit(nopython=True)
def D_nn(T_g, P_g):
    #   Calculate self mass diffusivity of nitrogen (m^2/s)
    if T_g > 63:
        D_n_1atm = -0.01675+4.51061e-5*T_g**1.5
    else:
        D_n_1atm = (-0.01675+4.51061e-5*63**1.5)/63**1.5*T_g**1.5
    D_n_p = D_n_1atm*101325/P_g
    D_n_p = D_n_p/1e4
    return D_n_p


# @jit(nopython=True)
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

# @numba.jit('f8(f8)')
# @jit(nopython=True)


def gamma(a):
    #   Calculate the correction factor of mass flux
    gam1 = np.exp(-np.power(a, 2.))+a*np.sqrt(np.pi)*(1+math.erf(a))
    return gam1


# @numba.jit('f8(f8,f8,f8,f8,f8)')
# @jit(nopython=True)
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

# @jit(nopython=True)

# This is Nr, delete R =0 point done


def continue_simulation():
    # if "path" exists
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/last_timestep/'
    if os.path.exists(pathname):
        # change Working directory
        rho = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
        p = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
        tg = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
        u = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
        v = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
        Ut = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))
        e = np.zeros((Nx+1, Nr), dtype=(np.longdouble, np.longdouble))

# define field variables
        rho = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\rho.csv",
                         delimiter=",", dtype=np.longdouble)
        p = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\p.csv",
                       delimiter=",", dtype=np.longdouble)
        tg = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\Tg.csv",
                        delimiter=",", dtype=np.longdouble)
        u = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\ux.csv",
                       delimiter=",", dtype=np.longdouble)
        v = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\ur.csv",
                       delimiter=",", dtype=np.longdouble)
        Ut[:, :] = np.sqrt(u[:, :]**2. + v[:, :]**2.)
        e = np.loadtxt("C:\\Users\\rababqjt\\Documents\\programming\\git-repos\\2d-vacuumbreak-explicit-V1-func-calc\\last_timestep\\e.csv",
                       delimiter=",", dtype=np.longdouble)
    return p, rho, tg, u, v, Ut, e


def bulk_values(T_s):
    # T_0 = 100.
    T_0 = 4.2  # K
    rho_0 = 1e-3  # An arbitrary small initial density in pipe, kg/m3
    # T_0 = p_0/rho_0/R*M_n
    p_0 = rho_0*R/M_n*T_0
    u_0 = 0
    v_0 = 0

    Ut_0 = np.sqrt(u_0**2. + v_0**2.)
# energy bulk
    e_0 = 5./2.*p_0 + np.sqrt(u_0**2. + v_0**2.)  # Initial internal energy

    bulk = [T_0, rho_0, p_0, e_0, Ut_0, u_0, v_0]
    print("p_0: ", p_0, "T_0:", T_0, "rho_0: ", rho_0, "e_0: ", e_0)
    return bulk


# @jit(nopython=True)
# def val_in_bulk_constant_T_P():
#     #   Calculate instant flow rate (kg/s)
#     T_0, rho_0, p_0, e_0, ux_0 = bulk_values()
#     p_in = p_0
#     T_in = T_0
#     rho_in = p_in / T_in/R*M_n
#     ux_in = 30.
#     ur_in = 0.
#     e_in = 5./2.*rho_in/M_n*R*T_in + 1./2.*rho_in*ux_in**2
#     e_in_x = e_in
#     out = np.array([p_in, ux_in, ur_in, rho_in, e_in, e_in_x, T_in])
#     return out


# @jit(nopython=True)
def integral_mass_delSN(de):
    # del_SN = np.zeros((Nx+1), dtype=(np.float64))
    de0 = np.zeros((Nx+1), dtype=(np.float64))
    # Integrate deposited mass
    for m in np.arange(Nx+1):
        de0[m] += dt*np.pi*D*de[m]

# Calculate the SN2 layer thickness
    del_SN = de0/np.pi/D/rho_sn
    # print("del_SN: ", del_SN)

    return de0, del_SN  # the de0 is incremented and never restarted

# recalculates Tg to be equal to Ts.
# NOTE: does this affect the velocities? does mde change? and if yes, does it mean ur changes?


# @jit(nopython=True)
def gas_surface_temp_check(T, Ts, ur, e, u, rho):
    # print("starting Tg> Ts check")
    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        if T[m, Nr] < Ts[m]:
            e[m, Nr] = 5./2.*rho[m, Nr]*R*Ts[m] / \
                M_n + 1./2.*rho[m, Nr]*ur[m, Nr]**2

            # # print("THIS IS T2 < Ts")
            # # print("e2 surface", e2[m, n])
            # check_negative(e2[m, n], n)

    T = 2./5.*(e - 1./2.*rho * ur**2.)*M_n/rho/R
    #     print(
    #         "T2 surface recalculated to make it equal to wall temperature (BC)", T2[m, n])
    #     check_negative(T2[m, n], n)

    # NOTE: Energy is changed assuming density and radial velocity constant. Is this correct?
    p = rho*R*T/M_n
    e = 5./2. * rho*R*T/M_n + 1./2 * rho*u**2

    return T, e, p, rho, u


# @jit(nopython=True)
def Cu_Wall_function(urx, Tx, Twx, Tcx, Tsx, T_in, delSN, de1, ex, ux, rhox, px, T2, p2, e2, rho2, u2, ur2):
    # define wall second derivative
    dt2nd_w_m = dt2nd_w_matrix(Twx, T_in)
    qi = np.zeros((Nx+1), dtype=(np.float64))
    q_dep = np.zeros((Nx+1), dtype=(np.float64))
    Tw2 = np.zeros((Nx+1), dtype=(np.float64))
    Tc2 = np.zeros((Nx+1), dtype=(np.float64))
    Ts2 = np.zeros((Nx+1), dtype=(np.float64))
    qhe = np.zeros((Nx+1), dtype=(np.float64))

# Initial calculations:

# Only consider thermal resistance in SN2 layer when del_SN > 1e-5:
# # NOTE: CHECK THIS LOGIC TREE

    print("calculating Tw, Tc, Ts, qdep")

    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        if delSN[m] > 1e-5:
            print(
                "This is del_SN > 1e-5 condition, conduction across SN2 layer considered")

            # heatflux into copper wall from frost layer
            qi[m] = k_sn*(Tsx[m]-Twx[m])/delSN[m]
            # print("qi: ", qi)
            # check_negative(qi, n)
        else:
            # no heatflux into copper wall
            qi[m] = 0
            # print("qi: ", qi)
            # check_negative(qi, n)

# pipe wall equation
        Tw2[m] = Twx[m] + dt/(w_coe*c_c(Twx[m]))*(qi[m]-q_h(Twx[m])
                                                  * Do/D) + dt/(rho_cu*c_c(Twx[m]))*k_cu(Twx[m])*dt2nd_w_m[m]

# q deposited into frost layer. Nusselt convection neglected
        q_dep[m] = de1[m]*(1/2*(urx[m, Nr])**2 + delta_h(Tx[m, Nr], Tsx[m]))

# SN2 Tc equation
# SN2 surface temperature calculation
        if delSN[m] < 1e-5:
            Tc2[m] = Tw2[m]
            Ts2[m] = 2*Tc2[m] - Tw2[m]
        else:
            Tc2[m] = Tcx[m] + dt * (q_dep[m]-qi[m]) / \
                (rho_sn * c_n(Tsx[m])*delSN[m])
            Ts2[m] = 2*Tc2[m] - Tw2[m]

    # NOTE: delta_h will change if T =Ts and will be zero.
    # NOTE: Check this logic, very important
    print("Forcing Tg >= Ts")
    for j in np.arange(np.int64(0), np.int64(Nx+1)):
        if T2[j, Nr] < Ts2[j]:
            # NOTE: Should i use the old mass deposition? import mdot of last matrix?
            # NOTE: Is this the current or updated velocity?
            T2[j, Nr] = Ts2[j]

# q deposited into frost layer. Nusselt convection neglected

        print("recalculating energies")
        T2, e2, p2, rho2, u2 = gas_surface_temp_check(
            T2, Ts2, ur2, e2, u2, rho2)

    print("calculating qhe")
    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        qhe[m] = q_h(Twx[m])

# NOTE: Check qhe values larger than e2 of Tg
# NOTE: Put a limiting factor for qhe
    # s = ex[:, Nr]
    # ID = s < qhe
    # print("Checking qhe > e1", np.any(ID))

    # print("saving qhe")
    # save_qhe(i, dt, qhe)
    # print("saving q_dep")
    # save_qdep(i,dt,q_dep)
    print("Wall function complete")
    w_out = [Tw2, Ts2, Tc2, qhe, dt2nd_w_m, q_dep]
    return w_out


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

# @jit(nopython=True)


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

# remove timestepping folder


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

# Radial velocity assumed zero


def no_slip_no_mdot(p, rho, tg, u, v, Ut, e):
    # no mass deposition
    u[:, Nr] = 0
    v[:, Nr] = 0
    Ut[:, Nr] = 0
    # energy assumed constant
    tg = (e - 1./2.*rho*Ut**2) * 2./5. / rho/R*M_n
    p = rho*R/M_n*tg
    return p, tg, u, v, Ut, e


# @jit(nopython=True)
def inlet_BC(u, v, Ut, p, rho, T, e, p_inl, u_inl, rho_inl, T_inl, e_inl):
    p[0, :] = p_inl
    rho[0, :] = rho_inl
    T[0, :] = T_inl
    e[0, :] = e_inl
    u[0, :] = u_inl

# parabolic velocity profile
    # for y in np.arange(Nr+1):
    #     u[0, y] = u_inl*(1.0 - ((y*dr)/R_cyl)**2)
    #     # print("parabolic y", y)
    #     Ut[0, y] = u[0, y]


# no slip
    u[:, Nr] = 0
    v[:, Nr] = 0
    Ut[:, Nr] = 0
    Ut = np.sqrt(u**2. + v**2.)
    e = 5./2. * p + 1./2 * rho*Ut**2
    return [u, v, Ut, p, rho, T, e]

# @jit(nopython=True)


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

# recalculating velocity from energy, T, rho


# @jit(nopython=True)
def val_in_constant(p_0, T_0, u_0):
    #   Calculate instant flow rate (kg/s)
    p_in = 6000  # Pa
    # rho_in = 0.5
    T_in = 298.
    rho_in = p_in/R*M_n / T_in
    u_in = np.sqrt(gamma_n2*R/M_n*T_in)
    # u_in = u_0
    v_in = 0.
    Ut_in = np.sqrt(u_in**2. + v_in**2.)
    # Ut_in = 50.
    # Ut_in = np.sqrt(u_in**2 + v_in**2)
    e_in = 5./2.*rho_in/M_n*R*T_in + 1./2.*rho_in*Ut_in**2
    out = np.array([p_in, u_in, v_in, rho_in, e_in, T_in])
    return out

# @jit(nopython=True)


def val_in(n):
    #   Calculate instant flow rate (kg/s)
    # Fitting results
    #    A1 = 0.00277; C = 49995.15263  # 50 kPa
    T_in = 298.
    A1 = 0.00261
    C = 100902.5175  # 100 kPa
   # A1 = 0.00277; C = 10000.15263  # 50 kPa
    ux_in = 10.
    P_in_fit = np.power(A1*n*dt+np.power(C, -1./7.), -7.)
    # P_in_fit = 1000.
    # P_in_fit = 1./2.*P_in_fit
    # print("pressure is halved P_in_fit", P_in_fit)

    dP_in_fit = -7.*A1*np.power(A1*n*dt+np.power(C, -1./7.), -8.)

    q_in = -(np.power(C, 2./7.)*0.230735318/1.4/297./T_in) * \
        (np.power(P_in_fit, -2./7.)*dP_in_fit)
    ma_in_x = q_in/A
    rho_in = ma_in_x/u_in_x
    p_in = rho_in/M_n*R*T_in
    ur_in = 0.
    e_in = 5./2.*rho_in/M_n*R*T_in + 1./2.*rho_in*u_in_x**2
    # print("u_in_x", u_in_x)
    out = np.array([p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in])
    # print(
    #     "val_in from function [q_in, ux_in, ur_in, rho_in, p_in, e_in]: ", out)
    return out

# @numba.jit('f8(f8,f8,f8,f8)')
# @jit(nopython=True)


def DN(T, P, u, T_w, rho):
    #   Calculate dimensionless numbers
    rho = P*M_n/R/T
    # rho_w=f_ps(T_w)*M_n/R/T #_w
    mu = mu_n(T, P)
    neu = mu/rho
    # print("mu", mu)
    Re = rho*(u)*D/mu  # Reynolds number
    D_n = D_nn(T, P)
    Sc = mu/rho/D_n  # Schmidt number
    Kn = 2*mu/P/np.sqrt(8*M_n/np.pi/R/T)/D
    mu_w = mu_n(T_w, f_ps(T_w))
    Sh = 0.027*Re**0.8*Sc**(1/3)*(mu/mu_w)**0.14  # Sherwood number
    Nu = 0.027*Re**0.8*Pr_n**(1/3)*(mu/mu_w)**0.14  # Nusselt number
    Cou = u*dt/dx  # Courant Number
    Pe = u * L/neu
    DN_all = np.array([Re, Sc, Kn, Sh, Nu, Cou, Pe])
    # print(DN_all)
    # print("Courant Number is: ", Cou)
    return DN_all


def Peclet_grid(pe, u, D_hyd, p, T):
    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        for n in np.arange(np.int64(1), np.int64(Nr+1)):
            pe[m, n] = u[m, n]*D_hyd / mu_n(T[m, n], p[m, n])
    return pe

# de1[m] = m_de(T1[m, n], p1[m, n], Tw1[m], de1[m], rho1[m, n]*ur1[m, n]-rho1[m, n-1]*ur1[m, n-1])

# NOTE: I am getting wrong mass deposition values... from 1d it is in the order of e-6


# returns mass deposition rate to put in de1 matrix
# @numba.jit('f8(f8,f8,f8,f8,f8)')
# @jit(nopython=True)
def m_de(T, P, ur, Ts1, de, dm):  # dm_r, ur, N):
    print("T,P,ur, Ts1, de, dm", [T, P, ur, Ts1, de, dm])
    rho = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                   np.float64))  # place holder
    v_m1 = np.zeros((Nx+1), dtype=(np.float64))
    u_mean1 = np.zeros((Nx+1), dtype=(np.float64))
    gam1 = np.zeros((Nx+1), dtype=(np.float64))
    P_s = np.zeros((Nx+1), dtype=(np.float64))
    rho_min = np.zeros((Nx+1), dtype=(np.float64))
    beta = np.zeros((Nx+1), dtype=(np.float64))
    m_out = np.zeros((Nx+1), dtype=(np.float64))
    m_max = np.zeros((Nx+1), dtype=(np.float64))

    p_0 = bulk_values(T_s)[2]
    # print("mdot calc: ", "Tg: ", T, " P: ",
    #       P, "Ts: ", T_s, "de: ", de, "dm: ", dm)
    #   Calculate deposition rate (kg/(m^2*s))

    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        if T[m, Nr] == 0:
            T[m, Nr] = 0.00001
        rho[m, Nr] = P[m, Nr]*M_n/R/T[m, Nr]
        rho_min[m] = p_0*M_n/R/T[m, Nr]
    # # no division by zero
    #     if rho[m,Nr] == 0:
    #         rho[m,Nr] = 0.00001
        v_m1[m] = np.sqrt(2*R*T[m, Nr]/M_n)  # thermal velocity of molecules
        u_mean1[m] = de[m]/rho[m, Nr]  # mean flow velocity towards the wall.
        beta[m] = u_mean1[m]/v_m1[m]  # this is Beta from Hertz Knudson
        gam1[m] = gamma(beta[m])  # deviation from Maxwellian velocity.
        P_s[m] = f_ps(Ts1[m])

        if (P[m, Nr] > P_s[m] and P[m, Nr] > p_0):
            # Correlated Hertz-Knudsen Relation #####
            m_out[m] = np.sqrt(M_n/2/np.pi/R)*Sc_PP * \
                (gam1[m]*P[m, Nr]/np.sqrt(T[m, Nr])-P_s[m]/np.sqrt(Ts1[m]))
            print("m_out calc", m_out)

            if Ts1[m] > 25:
                print("P>P0, P>Ps")
                # Arbitrary smooth the transition to steady deposition
                # NOTE: Check this smoothing function.
                m_out[m] = m_out[m]*exp_smooth(Ts1[m]-25., 1., 0.05,
                                               0.03, (f_ts(P[m, Nr]*np.sqrt(Ts1[m]/T[m, Nr]))-25.)/2.)

            # Speed of sound limit for the condensation flux
            # sqrt(7./5.*R*T/M_n)*rho
            # Used Conti in X-direction, since its absolute flux.
            # m_max[:] = D/4./dt*(rho[:, Nr-1]-rho_min)-D/4./dx*dm - D/4. * \
            #     1/N[:, Nr-1]/dr*(rho[:, Nr-1]*N[:, Nr-1]*dr*ur[:,
            #                      Nr-1] - rho[:, Nr-2]*N[:, Nr-2]*dr*ur[:, Nr-2])

            # using conti surface
            # m_max = D/4./dt*(rho-rho_min)-D/4. * (1/Nr/dr*dm_r)
            # dm is a matrix
            m_max[m] = D/4./dt*(rho[m, Nr]-rho_min[m]) - \
                D/4./dx*dm[m]  # sqrt(7./5.*R*T/M_n)*rho
            m_max[m] = 2.0e-30

            if m_out[m] > m_max[m]:
                m_out[m] = m_max[m]
                # print("mout = mmax")
        else:
            m_out[m] = 0
        # m_out = 0  # NO HEAT TRANSFER/ MASS DEPOSITION CASE
        # print("de2: ", m_out)

    print("m_out", m_out)

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/m_dot/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "m_dot"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
        # os.rmdir('C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/')
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    os.chdir(pathname)
    np.savetxt("m_out.csv", m_out, delimiter=",")
    np.savetxt("m_max.csv", m_max, delimiter=",")

    return m_out  # Output: mass deposition flux, no convective heat flux MATRIX

# def m_de(T, P, Ts1, de, dm, dm_r, ur, N):
#     p_0 = bulk_values(T_s)[2]
#     # print("mdot calc: ", "Tg: ", T, " P: ",
#     #       P, "Ts: ", T_s, "de: ", de, "dm: ", dm)
#     #   Calculate deposition rate (kg/(m^2*s))
#     if T == 0:
#         T = 0.00001
#     rho = P*M_n/R/T
# # no division by zero
#     if rho == 0:
#         rho = 0.00001
#     v_m1 = np.sqrt(2*R*T/M_n)  # thermal velocity of molecules
#     u_mean1 = de/rho  # mean flow velocity towards the wall.
#     beta = u_mean1/v_m1  # this is Beta from Hertz Knudson
#     gam1 = gamma(beta)  # deviation from Maxwellian velocity.
#     P_s = f_ps(Ts1)

#     if P > P_s and P > p_0:
#         # Correlated Hertz-Knudsen Relation #####
#         m_out = np.sqrt(M_n/2/np.pi/R)*Sc_PP * \
#             (gam1*P/np.sqrt(T)-P_s/np.sqrt(Ts1))
#         print("m_out calc", m_out)

#         if Ts1 > 25:
#             print("P>P0, P>Ps")
#             # Arbitrary smooth the transition to steady deposition
#             # NOTE: Check this smoothing function.
#             m_out = m_out*exp_smooth(Ts1-25., 1., 0.05,
#                                      0.03, (f_ts(P*np.sqrt(Ts1/T))-25.)/2.)

#         # Speed of sound limit for the condensation flux
#         rho_min = p_0*M_n/R/T
#         # sqrt(7./5.*R*T/M_n)*rho
#         # Used Conti in X-direction, since its absolute flux.
#         m_max[:] = D/4./dt*(rho[:, Nr-1]-rho_min)-D/4./dx*dm - D/4. * \
#             1/N[:, Nr-1]/dr*(rho[:, Nr-1]*N[:, Nr-1]*dr*ur[:,
#                              Nr-1] - rho[:, Nr-2]*N[:, Nr-2]*dr*ur[:, Nr-2])

#         # using conti surface
#         # m_max = D/4./dt*(rho-rho_min)-D/4. * (1/Nr/dr*dm_r)
#         print("m_max", m_max, "dm", dm)
#         if m_out > m_max:
#             m_out = m_max
#             # print("mout = mmax")
#     else:
#         m_out = 0
#     rho_min = p_0*M_n/R/T
#     # m_out = 0  # NO HEAT TRANSFER/ MASS DEPOSITION CASE
#     # print("de2: ", m_out)
#     return m_out  # Output: mass deposition flux, no convective heat flux

# @numba.jit('f8(f8,f8)')


# @jit(nopython=True)
def q_h(tw, BW_coe=0.017):  # (W/(m^2*K)
    # Boiling heat transfer rate of helium (W/(m^2*K))
    # delT = ts-4.2
    delT = tw-4.2

    q_con = 0.375*1000.*delT  # Convection
    q_nu = 58.*1000.*(delT**2.5)  # Nucleate boiling
    q_tr = 7500.  # Transition to film boiling
    # print("qcond: ", q_con, "q_nu: ", q_nu, "q_tr: ", q_tr)
    tt = np.power(q_tr/10000./BW_coe, 1./1.25)
    b2 = tt-1.
    # Breen and Westwater Correlation with tuning parameter
    # print("delT", delT)
    q_fi = (BW_coe*np.power(delT, 1.25))*10000.  # Film boiling
    if q_con > q_nu or delT < 0.01:
        q_he = q_con
    elif q_nu <= q_tr:
        q_he = q_nu
    elif q_nu > q_tr and q_fi < q_tr:
        q_he = q_tr
    else:
        q_he = q_fi
    # Smooth the turning point
    if delT >= tt-b2 and delT <= tt+b2:
        q_min = 1.25/2/b2*10000*BW_coe * \
            (1/2.25*np.power(tt-b2, 2.25)-(tt-b2)/1.25*np.power(tt-b2, 1.25))
        q_max = 1.25/2/b2*10000*BW_coe * \
            (1/2.25*np.power(tt+b2, 2.25)-(tt-b2)/1.25*np.power(tt+b2, 1.25))
        q_max1 = (BW_coe*np.power(tt+b2, 1.25))*10000
        q_he = (1.25/2/b2*10000*BW_coe*(1/2.25*np.power(delT, 2.25)-(tt-b2) /
                1.25*np.power(delT, 1.25))-q_min)/(q_max-q_min)*(q_max1-q_tr)+q_tr
    # print("rate of heat transfer to helium:", q_he)
    # print("q_h calc: ", q_he, "Tw: ", tw)
    # q_he = 0  # NO HEAT TRANSFER CASE
    return q_he

# Initialization


def save_initial_conditions(rho1, ux1, ur1, u1, e1, T1, de0, p1, de1):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "initial_conditions"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
        # os.rmdir('C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/')
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    os.chdir(pathname)
    np.savetxt("rho.csv", rho1, delimiter=",")
    np.savetxt("Tg.csv", T1, delimiter=",")
    np.savetxt("u.csv", u1, delimiter=",")
    np.savetxt("ux.csv", ux1, delimiter=",")
    np.savetxt("ur.csv", ur1, delimiter=",")
    np.savetxt("e.csv", e1, delimiter=",")
    # np.savetxt("tw.csv", Tw1, delimiter=",")
    # np.savetxt("ts.csv", Ts1, delimiter=",")
    # np.savetxt("tc.csv", Tc1, delimiter=",")
    np.savetxt("de.csv", de0, delimiter=",")
    np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    # np.savetxt("pe.csv", pe, delimiter=",")


def save_data(tx, dt, rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1):
    # a = a+1
    # if a >= 10:
    # increment2 = (tx-10)*dt
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
    # np.savetxt("de_rate.csv", de2, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    # np.savetxt("peclet.csv", pe, delimiter=",")
    # np.savetxt("qhe.csv", qhe, delimiter=",")
    # np.savetxt("qdep.csv", qdep, delimiter=",")
    # np.savetxt("visc.csv", visc, delimiter=",")


def save_last(tx, dt, rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1):
    # increment = (tx+1)*dt
    # previous_step = tx*dt
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/last_timestep/'
    # +  "{:.6f}".format(increment) + '/'
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
    # np.savetxt("de_rate.csv", de2, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    # np.savetxt("peclet.csv", pe, delimiter=",")
    # np.savetxt("qhe.csv", qhe, delimiter=",")
    # np.savetxt("qdep.csv", qdep, delimiter=",")
    # np.savetxt("visc.csv", visc, delimiter=",")
    # pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/last_timestep/' + \
    #     "{:.6f}".format(previous_step) + '/'
    # if os.path.exists(pathname):
    #     location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/last_timestep"
    #     dir = str(previous_step)
    #     path = os.path.join(location, dir)
    #     shutil.rmtree(path)


def save_RK3(x, tx, dt, rho1, ux1, ur1, u1, e1, T1, p1):
    increment = (tx+1)*dt

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/RK3/' + \
        "{:.4f}".format(increment) + '/' + "{:.1f}".format(x) + '/'
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
    # np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")


def delete_r0_point(rho, u, v, Ut, e, T, p):
    rho = np.delete(rho, 0, axis=1)
    u = np.delete(u, 0, axis=1)
    v = np.delete(v, 0, axis=1)
    Ut = np.delete(Ut, 0, axis=1)
    e = np.delete(e, 0, axis=1)
    T = np.delete(T, 0, axis=1)
    p = np.delete(p, 0, axis=1)
    # pe3 = np.delete(pe, 0, axis=1)
    # visc3 = np.delete(visc, 0, axis=1)
    return [rho, u, v, Ut, e, T, p]


if __name__ == '__main__':
    t_sat = 70
    # p_test = f_ps(t_sat)
    # print("p_test", p_test)

    p_sat = 10000
    # t_test = f_ts(p_sat)
    # print("t_test", t_test)

    l = np.linspace(1, 300, 1000)
    k = k_cu(l)

    # plt.figure()
    # plt.plot(l, k)
    # plt.show()

#    p_before = np.log(-2)

    # Nx = 20; Nr =20
    # rho1 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
    # T1 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
    # T1[2,5] = -2
    # eps = 5./2.*rho1[2, 2]/M_n*R * \
    #                     T1[2, 5]
    # check_negative(eps, eps, 1)

    # p_before = -3
    # check_negative(p_before, p_before, 1)

#   Nx = 10; Nr =10
#    rho12 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
 #    check_negative(rho12[1,2],rho12, 1)

# ------------------------- Sublimation heat ------------------------------- #

    tg = 4.2
    ts = 4.2
    print("delta_h ", delta_h(tg, ts))

# ------------------------- specific heat of solid nitrogen (J/(kg*K)) ------------------------------- #

    # print("c_n ", c_n(ts))
# ------------------------- thermal velocity ------------------------------- #

    # print("vm ", v_m(tg))

# ------------------------- heat capacity of copper (J/(kg*K)) ------------------------------- #

    # print("c_c ", c_c(ts))

# thermal conductivity of copper (RRR=10) (W/(m*K)) ------------------------------- #
    T = 4.2
    print("k_cu", k_cu(4))

# ------------------------- self mass diffusivity of nitrogen (m^2/s) ------------------------------- #

    T_g = 298
    P_g = 1000
    # print("D_nn", D_nn(T_g, P_g))

# ------------------------- Viscosity ------------------------------- #

    T = 273.15
    P = 9806649
    # print("mu_n ", mu_n(T, P))

# ------------------------- Error function ------------------------------- #

#    a = umean/vm1
    a = 0.5
    # print("gamma ", gamma(a))


# ------------------------- Inlet values ------------------------------- #

    # print("val_in ", val_in(0))

# ------------------------- Dimensionless numbers ------------------------------- #

    T = 30
    P = 3000
    u = 100
    T_w = 15
    # print("DN ", DN(T, P, u, T_w))

# ------------------------- Mass Deposition ------------------------------- #

    #   Time and spatial step
    L = 6.45
    Nx = 120.  # Total length & spatial step - x direction 6.45
    R_cyl = 1.27e-2
    Nr = 5.  # Total length & spatial step - r direction
    T = 3.
    Nt = 70000.  # Total time & time step
    dt = T/Nt
    dx = L/Nx
    dr = R_cyl/Nr
    ur = 3

    T = 100
    P = 4000
    T_s = 30
    de = 20
    dm = -10
    # print("m_de", m_de(T, P, T_w, de, dm))


# ------------------------- Heat transferred ------------------------------- #

    tw = 298

    k = np.zeros(30000, dtype=np.float64())
    l = np.linspace(4.2, 100, 30000)
    for i in range(len(l)):
        k[i] = q_h(l[i])

    # k = np.zeros(30000, dtype=np.float64())
    # k[l] = q_he(l[:])
    # for x, i in zip(l, range(len(l))):
    #     k[i] = q_h(x)

    #  in l:
        # k[x] = q_h(x, BW_coe=0.017)

    plt.figure()
    plt.plot(l, k)
    plt.show()
    # print("q_h ", q_h(tw, BW_coe))

# ------------------------- Exponential Smoothing ------------------------------- #
    # p_in = 8000
    # p_0 = 2000
    # n_trans = 70
    # Nx = 200
    # L = 6.45
    # p1 = np.full((Nx+1), p_0, dtype=np.float64)  # Pressure
    # for i in np.arange(0, Nx+1):
    #     p1[i] = exp_smooth(i+n_trans, p_in*2.-p_0, p_0, 0.3, n_trans)

    # x = np.linspace(0, L, Nx+1)
    # plt.plot(x, p1)
    # plt.show()

# ended here plotting

    # print("p1 array", p1)
    # print("P1 smoothing values", p1[i,:])
    #    rho1[i,:]=exp_smooth(i,rho_in,rho_0,0.75,n_trans)
    #    T1[i, :] = T_neck(i)
    # if i<51: T1[i]=T_in
    #    rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]       #NOTE: CHECK TRANSITIONS
    # if i < n_trans+1:
    #     e1[i, :] = 5./2.*p1[i, :]+1./2.*rho1[i, :]*u1[i, :]**2
    #     rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]  # IDEAL GAS LAW

    # # print("p1 matrix after smoothing", p1)
    # else:
    #     e1[i, :] = 5/2*rho1[i, :]/M_n*R*T_in+1/2**rho1[i, :]*u1[i, :]**2


## ----------- plot val_in ------------------- #

    # plt.figure()
    # final_i = 100
    # y = np.zeros((final_i+1), dtype=np.int64)  # heat transfer
    # x = np.linspace(0, 101, 101)
    # print(np.shape(x), np.shape(y))
    # for i in np.arange(0, final_i+1):
    #     y[i] = val_in(i)[0]
    #     print(y[i])
    # plt.scatter(x, y, label="pressure", color='red')
    # plt.title("pressure with i in val_in fcn")
    # plt.xlabel("iteration (i)")
    # plt.ylabel("pressure [Pa]")
    # plt.show()


# EXTRAS
