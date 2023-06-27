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

# u_in_x = np.sqrt(7./5.*R*T_in/M_n)*1.0  # Inlet velocity, m/s (gamma*RT)
u_in_x = 1.

# @numba.jit('f8(f8,f8)')


@jit(nopython=True)
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


@jit(nopython=True)
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
    p1 = np.full((Nx+1, Nr+1), p_0, dtype=(np.float64, np.float64))  # Pressure
    rho1 = np.full((Nx+1, Nr+1), rho_0,
                   dtype=(np.float64, np.float64))  # Density
    ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  # velocity -x
    ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  # velocity -r
    u1 = np.sqrt(np.square(ux1) + np.square(ur1))  # total velocity
    # Internal energy
    e1 = np.full((Nx+1, Nr+1), e_0, dtype=(np.float64, np.float64))
    # CHECK TODO: calculate using equation velocity.
    # TODO: calculate using equation velocity.

    T1 = np.full((Nx+1, Nr+1), T_0,
                 dtype=(np.float64, np.float64))  # Temperature

    rho2 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))
    ux2 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ur2 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    u2 = np.sqrt(np.square(ux2) + np.square(ur2))  # total velocity
    e2 = np.full((Nx+1, Nr+1), e_0, dtype=(np.float64, np.float64))
    T2 = np.full((Nx+1, Nr+1), T_0, dtype=(np.float64, np.float64))
    p2 = np.full((Nx+1, Nr+1), p_0, dtype=(np.float64, np.float64))  # Pressure

    Tw1 = np.full((Nx+1), T_s, dtype=(np.float64))  # Wall temperature
    Tw2 = np.full((Nx+1), T_s, dtype=(np.float64))
    # Temperature of SN2 surface
    Ts1 = np.full((Nx+1), T_0, dtype=(np.float64))
    Ts2 = np.full((Nx+1), T_0, dtype=(np.float64))

    # Average temperature of SN2 layer
    Tc1 = np.full((Nx+1), T_s, dtype=(np.float64))
    Tc2 = np.full((Nx+1), T_s, dtype=(np.float64))
    de0 = np.zeros((Nx+1), dtype=(np.float64))  # Deposition mass, kg/m
    de1 = np.full((Nx+1), 0., dtype=(np.float64))  # Deposition rate
    # de2 = np.full((Nx+1), 0., dtype=(np.float64))  # Deposition rate
    qhe = np.zeros_like(de0, dtype=np.float64)  # heat transfer

    # These matrices are just place holder. These will be overwritten and saved. (remove r=0)
    rho3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    ux3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    ur3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    u3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    e3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    T3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    p3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))

    # Dimensionless number in grid:
    Pe = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                  np.float64))  # Peclet number
    Pe1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                   np.float64))  # Peclet number
    out = [p1, rho1, ux1, ur1, u1, e1, T1, rho2, ux2, ur2, u2, e2, T2, p2, Tw1, Tw2,
           Ts1, Ts2, Tc1, Tc2, de0, de1, qhe, rho3, ux3, ur3, u3, e3, T3, p3, Pe, Pe1]
    return out


def initialize_ghost():
    # This is for the ghost cells for the WENO reconstruction.

    return ro_rec, ux_rec,

    # x-direction
    # if (iflx==1)
    #     qLx,qRx = weno5(nx,ny,nz,q,1)
    # elseif (iflx==2)
    #     qLx,qRx = upwind5(nx,ny,nz,q,1)
    # end
    # FLx = flux(nx,ny,nz,qLx,1)
    # FRx = flux(nx,ny,nz,qRx,1)
    # Fx  = rusanov_3d(q,qLx,FLx,qRx,FRx,nx,ny,nz,1)

    # #y-direction
    # if (iflx==1)
    #     qLy,qRy = weno5(nx,ny,nz,q,2)
    # elseif (iflx==2)
    #     qLy,qRy = upwind5(nx,ny,nz,q,2)
    # end
    # FLy = flux(nx,ny,nz,qLy,2)
    # FRy = flux(nx,ny,nz,qRy,2)
    # Fy  = rusanov_3d(q,qLy,FLy,qRy,FRy,nx,ny,nz,2)

# def rhs_rho(m, n, dr, dx, ur, ux, rho, a, ux_in, rho_in):


def rhs_matrix_initialization():
    # nonconservative form

    rhs_rho = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    rhs_ux = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    rhs_ur = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    rhs_e = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    return rhs_rho, rhs_ux, rhs_ur, rhs_e


def rhs_conservative_matrix_initialization():
    # conservative form

    rhs_rho = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    rhs_max = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    rhs_mar = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    rhs_e = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    return rhs_rho, rhs_max, rhs_mar, rhs_e


def n_matrix():
    # Initialized once when starting main
    n = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for i in np.arange(np.int64(1), np.int64(Nx+1)):
        for j in np.arange(np.int64(1), np.int64(Nr+1)):
            n[i, j] = j
    print("Removing N=0 in matrix")
    n[:, 0] = 1
    n[0, :] = 1
    return n


# def m_matrix():
#     # Initialized once when starting main
#     mm = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
#     for i in np.arange(np.int64(1), np.int64(Nx+1)):
#         for j in np.arange(np.int64(1), np.int64(Nr+1)):
#             mm[i, j] = j
#     return mm


# returns viscosity matrix
def viscous_matrix(T, P):
    visc_matrix = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            visc_matrix[m, n] = mu_n(T[m, n], P[m, n])

# perform NAN value matrix checks:
    print("performing finite check on visc_matrix")

    for x in np.arange(len(visc_matrix)):
        assert np.isfinite(visc_matrix).all()

# negative viscosity check
    if np.any(visc_matrix < 0):
        print("The viscous matrix has at least one negative value")
        exit()

    return visc_matrix


# return S term matrix, input previous de1
def source_mass_depo_matrix(rho_0, T, P, T_s, rho, ur, de):  # -4/D* mdot
    dm = np.zeros((Nx+1), dtype=(np.float64))
    de1 = np.zeros((Nx+1), dtype=(np.float64))
    S = np.zeros((Nx+1), dtype=(np.float64))

# skip m=0, not needed
    for m in np.arange(np.int64(1), np.int64(Nx+1)):
        dm[m] = rho[m, Nr] * Nr*dr*ur[m, Nr]-rho[m, Nr-1]*(Nr-1)*dr*ur[m, Nr-1]
        if rho[m, Nr] > 2.*rho_0:
            # print("temp gas",T1[m,n], "pressure", p1_before_dep, "temp wall: ", Tw1[m],"mass depo", de1[m], "dm", rho1[m, n]*n*dr*ur1[m, n]-rho1[m, n-1]*n*dr*ur1[m, n-1], "n grid point", n)
            # print("inputs m_de calc: [T1, p1, Ts1, de1, rho1, ur1]",
            #       T1[m, n], p1[m, n], Ts1[m], de1[m], rho1[m, n], ur1[m, n])
            # print("stopped here source_mass fcn")
            de1[m] = m_de(T[m, Nr], P[m, Nr], T_s[m],
                          de[m], dm[m])  # used BWD
            # print("m_de / de1 calculated:", de1[m])
            # check_negative(de1[m], n)
        else:
            de1[m] = 0.
    S = -1*4./D * de1
    return S


# returns continuity RHS matrix, including source term S
def rhs_rho(d_dr, m_dx, N, S):
    # calculate source term
    rhs_rho = - 1/N/dr*d_dr - m_dx

    # surface equation
    rhs_rho[:, Nr] = - 1/N[:, Nr]/dr*d_dr[:, Nr] + S[:]
    # rhs_rho[:,Nr] = -1/N/dr*d_dr  + S[:,Nr]

    return rhs_rho


# returns MOMENTUM RHS matrix
def rhs_ma(dp_dx, rho, dt2r_ux, N, ux_dr, dt2x_ux, ux, ux_dx, ur, dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix):

    rhs_ux = -dp_dx/rho + visc_matrix/rho * (
        dt2r_ux + 1/N/dr*ux_dr + dt2x_ux) - ux * ux_dx - ur*ux_dr

    rhs_ur = - dp_dr/rho + visc_matrix/rho * \
        (- ur/(dr**2*N**2) + 1/N/dr*ur_dr +
         dt2r_ur + dt2x_ur) - ux * ur_dx - ur*ur_dr

    # surface equations
    # no momentum equations

    return rhs_ux, rhs_ur

# assures no division by zero


def no_division_zero(array):
    # ensure no division by zero
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if array[m, n] == 0:
                array[m, n] = 0.0001
    return array


# returns ENERGY RHS matrix including source terms
def rhs_energy(grad_r, grad_x, N, S):
    rhs_e = - 1/N/dr*grad_r - grad_x
    rhs_e[:, Nr] = - 1/N[:, Nr]/dr*grad_r[:, Nr] + S[:]

    return rhs_e
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


def energy_difference_dt(e1, e2):
    # sum energy grid at t1
    sum_e1 = np.sum(e1)
# sum energy grid at t2
    sum_e2 = np.sum(e2)

    d_e = sum_e1 - sum_e2

    return d_e


# This iterates RK3 for all equations
def tvdrk3(ux, ur, u, p, rho, T, e, p_in, ux_in, rho_in, T_in, e_in, rho_0, ur_in, de_variable, de_constant, Tw, Ts, Tc):
    q = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  # place holder


# create N matrix:
    N = n_matrix()

    print("Deep copying initial matrices")

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


# substituting for RK3 initial loop
    qq = rho
    pp = p
    uu = u
    tt = T
# First step
# apply BCs
# NOTE: apply outlet Bcs also.
# l = [ux, ur, u, p, rho, T, e, Tw, Ts, Tc]
    for n in np.arange(3):
        if n == 0:
            ux, u, e, T = no_slip(ux, u, p, rho)
            l = inlet_BC(ux, ur, u, p, rho, T, e, p_in,
                         ux_in, rho_in, T_in, e_in, Tw, Ts, Tc)
            # k = outlet_BC(uxn, urn, uun, pn, qn, Tn, en)
            q = l[4]
            ux = l[0]
            ur = l[1]
            e = l[6]

            # print("starting temperature", l[5])
        else:  # n == 1, n==2:
            uxx, uu, ee, tt = no_slip(uxx, uu, pp, qq)
            l = inlet_BC(uxx, urr, uu, pp, qq, tt, ee, p_in,
                         ux_in, rho_in, T_in, e_in, Tw, Ts, Tc)
            # k = outlet_BC(uxn, urn, uun, pn, qn, Tn, en)
            if n == 1:
                print("second RK3 loop temperature", l[5])

            qq = l[4]
            uxx = l[0]
            urr = l[1]
            ee = l[6]

    # Calculating gradients (first and second) ---------------------------------------- #
        print("Calculating gradients for RK3 loop #", n)

        d_dr, m_dx = grad_rho_matrix(ux_in, rho_in, l[1], l[0], l[4])
        dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p_in, l[3], ux_in, l[0])
        dp_dr, ur_dx, ur_dr = grad_ur2_matrix(l[3], l[1], ur_in)
        grad_x, grad_r = grad_e2_matrix(l[1], l[0], ux_in, e_in, l[6])
        dt2x_ux, dt2x_ur = dt2x_matrix(ux_in, ur_in, l[0], l[1])
        dt2r_ux, dt2r_ur = dt2r_matrix(l[0], l[1])
        dt2x_ux, dt2x_ur = dt2x_matrix(ux_in, ur_in, l[0], l[1])
        dt2r_ux, dt2r_ur = dt2r_matrix(l[0], l[1])

    # viscosity calculations
        # print("l: ", l[5], l[3])
        visc_matrix = viscous_matrix(l[5], l[3])
        # if n == 2:
        # print(visc_matrix)
        # print("Checking viscosity matrix")
        assert np.isfinite(visc_matrix).all()

        # de_variable (de1) matrix input.
        # This function takes into account density large enough to have mass deposition case.
        print("Calculating Source term matrix")
        if n == 0:
            de_variable = de_constant
            # print("de_variable: ", de_variable)
        S = source_mass_depo_matrix(
            rho_0, l[5], l[3], l[8], l[4], l[1], de_variable)
        # This de1 is returned again, first calculation is the right one for this iteration. it takes last values.

        # CALCULATING RHS USING LOOP VALUES
        print("Calculating RHS terms matrices")
        r = rhs_rho(d_dr, m_dx, N, S)
        r_ux, r_ur = rhs_ma(dp_dx, l[4], dt2r_ux, N, ux_dr, dt2x_ux,
                            l[0], ux_dx, l[1], dp_dr, dt2r_ur, dt2x_ur, ur_dx, ur_dr, visc_matrix)
        r_e = rhs_energy(grad_r, grad_x, N, S)

# MAIN CALCULATIONS
        if n == 0:

            # first LHS calculations
            qq = q + dt*r
            uxx = ux + dt*r_ux
            urr = ur + dt*r_ur
            ee = e + dt*r_e
# pressure recalculation
            pp = qq * R/M_n * tt
# velocity recalculation
            uu = np.sqrt(uxx**2 + urr**2)
# apply surface conditions
            uxx, uu, ee, tt = no_slip(uxx, uu, pp, qq)

# temperature, pressure, total velocity recalculation
            uu = np.sqrt(uxx**2 + urr**2)
            tt = 2./5.*(ee - 1./2. * qq*uu**2)
            pp = qq * R/M_n * tt

# ensure no division by zero
            qq = no_division_zero(qq)

# NOTE: # mass deposition calculation from first rhs and source terms. Should the mass deposition calculated be the current mdot?
            de_timestep = -1 * S * D/4.
            de_variable = de_timestep

        elif n == 1:
            # Second LHS calculations
            qq = qq + dt*r
            uxx = uxx + dt*r_ux
            urr = urr + dt*r_ur
            ee = ee + dt*r_e

# pressure recalculation
            pp = qq * R/M_n * tt
# velocity recalculation
            uu = np.sqrt(uxx**2 + urr**2)

# de_variable calculation using new looped source term
            de_variable = -1 * S * D/4.

# apply surface conditions
            uxx, uu, ee, tt = no_slip(uxx, uu, pp, qq)

# radial velocity on surface is function of mass deposition
            urr[:, Nr] = de_variable[:]/qq[:, Nr]

# temperature, pressure, total velocity recalculation
            uu = np.sqrt(uxx**2 + urr**2)
            tt = 2./5.*(ee - 1./2. * qq*uu**2)
            pp = qq * R/M_n * tt

# second loop calculation

            qq = 0.75*q + 0.25*qq + 0.25*dt*r
            uxx = 0.75*ux + 0.25*uxx + 0.25*dt*r_ux
            urr = 0.75*ur + 0.25*urr + 0.25*dt*r_ur
            ee = 0.75*e + 0.25*ee + 0.25*dt*r_e

# pressure recalculation
            pp = qq * R/M_n * tt
# velocity recalculation
            uu = np.sqrt(uxx**2 + urr**2)

# apply surface conditions, no slip condition
            uxx, uu, ee, tt = no_slip(uxx, uu, pp, qq)

# de_variable calculation using new looped source term
            de_variable = -1 * S * D/4.

# radial velocity on surface is function of mass deposition
            urr[:, Nr] = de_variable[:]/qq[:, Nr]

# temperature, pressure, total velocity recalculation
            uu = np.sqrt(uxx**2 + urr**2)
            tt = 2./5.*(ee - 1./2. * qq*uu**2)
            pp = qq * R/M_n * tt

# ensure no division by zero
            qq = no_division_zero(qq)

        else:  # n==2
            # Third LHS calculations
            qq = qq + dt*r
            uxx = uxx + dt*r_ux
            urr = urr + dt*r_ur
            ee = ee + dt*r_e

# third (final) loop calculation
            qn = 1/3*q + 2/3*qq + 2/3*dt*r
            uxn = 1/3*ux + 2/3*uxx + 2/3*dt*r_ux
            urn = 1/3*ur + 2/3*urr + 2/3*dt*r_ur
            en = 1/3*e + 2/3*ee + 2/3*dt*r_e


# pressure recalculation
            pn = qn * R/M_n * tn
# velocity recalculation
            un = np.sqrt(uxn**2 + urn**2)

# apply surface conditions, no slip condition
            uxn, un, en, tn = no_slip(uxn, un, pn, qn)

# radial velocity on surface is function of mass deposition
            urn[:, Nr] = de_variable[:]/qn[:, Nr]

# temperature, pressure, total velocity recalculation
            un = np.sqrt(uxx**2 + urr**2)
            tn = 2./5.*(en - 1./2. * qn*un**2)
            pn = qn * R/M_n * tn

# ensure no division by zero
            qn = no_division_zero(qn)


# # No convective heat flux. q2 ?

    print("RK3 looping complete")
    rk_out = [de_timestep, qn, uxn, urn, un, en, tn, pn]
    return rk_out

    # Calculate the SN2 layer thickness
    # del_SN = de0[m]/np.pi/D/rho_sn
    # print("del_SN: ", del_SN)
    # check_negative(del_SN, n)

#                     de1[m] = 0

#

#                     ur2[m, n] = de1[m]/rho2[m, n]
#                     ux2[m, n] = 0.  # no slip boundary condition.
#                     u2[m, n] = np.sqrt(ux1[m, n]**2 + ur1[m, n]**2)


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

 # ENDDDDDDD

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


def plot_imshow(p, ux, T, rho, e):

    fig, axs = plt.subplots(5)
    fig.suptitle('Fields along tube for all R')

    # PRESSURE DISTRIBUTION
    im = axs[0].imshow(p.transpose())
    plt.colorbar(im, ax=axs[0])
    # plt.colorbar(im, ax=ax[0])
    axs[0].set(ylabel='Pressure [Pa]')
    # plt.title("Pressure smoothing")

    # VELOCITY DISTRIBUTION
    # axs[1].imshow()
    im = axs[1].imshow(ux.transpose())
    plt.colorbar(im, ax=axs[1])
    # axs[1].colorbars(location="bottom")
    axs[1].set(ylabel='Ux [m/s]')
    # plt.title("velocity parabolic smoothing")

    # Temperature DISTRIBUTION
    im = axs[2].imshow(T.transpose())
    plt.colorbar(im, ax=axs[2])
    axs[2].set(ylabel='Tg [K]')

    # axs[1].colorbars(location="bottom")
    # axs[2].set(ylabel='temperature [K]')

    im = axs[3].imshow(rho.transpose())
    plt.colorbar(im, ax=axs[3])
    axs[3].set(ylabel='Density [kg/m3]')

    im = axs[4].imshow(e.transpose())
    plt.colorbar(im, ax=axs[4])
    axs[4].set(ylabel='energy [kg/m3]')

    plt.xlabel("L(x)")
    plt.show()
    return


@jit(nopython=True)
def grad_rho2(m, n, ux_in, rho_in, ur, ux, rho):
    # if m == 0:
    #     a = rho_in
    #     m_dx = (rho[m, n]*ux[m, n]-rho_in*ux_in)/dx

    if m == 1:
        m_dx = (rho[m, n]*ux[m, n]-rho_in*ux_in)/dx

    elif m == Nx:
        m_dx = (rho[m, n]*ux[m, n]-rho[m-1, n]*ux[m-1, n])/dx

    # elif (m <= n_trans+2 and m >= n_trans+2):
    #     # NOTE Use four point CD at transition point.
    #     a = rho1[m, n]
    #     m_dx = (rho1[m-2, n] - 8*rho1[m-1, n] + 8 *
    #             rho1[m+1, n] - rho1[m+2, n])/(12*dx)

    else:
        m_dx = (rho[m, n]*ux[m, n]-rho[m-1, n]*ux[m-1, n])/(dx)

    if n == 1:
        # NOTE: SYMMETRY BC
        d_dr = (rho[m, n+2]*(n+2)*dr*ur[m, n+2] -
                rho[m, n] * n*dr*ur[m, n]) / (4*dr)

    elif n == Nr-1:
        d_dr = (rho[m, n]*n*dr*ur[m, n] -
                rho[m, n-1] * (n-1)*dr*ur[m, n-1])/dr

    else:
        d_dr = (rho[m, n+1]*(n+1)*dr*ur[m, n+1] -
                rho[m, n-1] * (n-1)*dr*ur[m, n-1])/(2*dr)

    return d_dr, m_dx


# @jit(nopython=True)
def grad_rho_matrix(ux_in, rho_in, ur, ux, rho):
    # create gradients arrays.
    m_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    d_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for i in np.arange(Nx+1):
        for j in np.arange(Nr+1):
            if i == 1:
                m_dx[i, j] = (rho[i, j]*ux[i, j]-rho_in*ux_in)/dx

            elif i == Nx:
                m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]*ux[i-1, j])/dx

            else:
                m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]*ux[i-1, j])/(dx)

            if j == 1:
                # NOTE: SYMMETRY BC
                d_dr[i, j] = (rho[i, j+2]*(j+2)*dr*ur[i, j+2] -
                              rho[i, j] * j*dr*ur[i, j]) / (4*dr)

            elif j == Nr-1 or j == Nr:
                d_dr[i, j] = (rho[i, j]*j*dr*ur[i, j] -
                              rho[i, j-1] * (j-1)*dr*ur[i, j-1])/dr

            else:
                d_dr[i, j] = (rho[i, j+1]*(i+1)*dr*ur[i, j+1] -
                              rho[i, j-1] * (j-1)*dr*ur[i, j-1])/(2*dr)
    return d_dr, m_dx


# @numba.jit('f8(f8,f8,f8,f8,f8,f8)')

@jit(nopython=True)
def grad_ux2(p_in, p, ux_in, ux, m, n):  # bulk

    if n == 1:
        # NOTE: SYMMETRY CONDITION HERE done
        ux_dr = (ux[m, n+2] - ux[m, n])/(4*dr)

    elif n == Nr-1:
        ux_dr = (ux[m, n] - ux[m, n-1])/dr  # BWD

    else:
        # upwind 1st order  - positive flow - advection
        ux_dr = (ux[m, n] - ux[m, n-1])/(dr)  # CD

    # if m == 0:
    #     # upwind 1st order  - positive flow - advection
    #     dp_dx = (p[m, n] - p_in)/dx
    #     ux_dx = (ux[m, n] - ux_in)/dx
        # 4-point CD
        # dp_dx = (p_in - 8*p_in + 8 *
        #          p1[m+1, n] - p1[m+2, n])/(12*dx)
        # ux_dx = (ux1[m+1, n] - ux_in)/(2*dx)

    # elif (m <= n_trans+2 and m >= n_trans-2):
    #     # NOTE Use four point CD at transition point.
    #     dp_dx = (p1[m-2, n] - 8*p1[m-1, n] + 8 *
    #              p1[m+1, n] - p1[m+2, n])/(12*dx)
    #     ux_dx = (ux1[m-2, n] - 8*ux1[m-1, n] + 8 *
    #              ux1[m+1, n] - ux1[m+2, n])/(12*dx)

    if m == 1:
        dp_dx = (p[m, n] - p_in)/dx  # BWD
        ux_dx = (ux[m, n] - ux_in)/dx  # BWD

    elif m == Nx:
        dp_dx = (p[m, n] - p[m-1, n])/dx  # BWD
        ux_dx = (ux[m, n] - ux[m-1, n])/dx  # BWD

    # elif (m >= 1 and m <= Nx - 2):

    else:
        # upwind 1st order  - positive flow - advection
        dp_dx = (p[m, n] - p[m-1, n])/dx
        ux_dx = (ux[m, n] - ux[m-1, n])/dx
        # dp_dx = (3*p1[m, n] - 4*p1[m-1, n] + p1[m-2, n]) / \
        #     dx  # # upwind, captures shocks
        # ux_dx = (3*ux1[m, n] - 4*ux1[m-1, n] + ux1[m-2, n]) / \
        #     dx  # # upwind, captures shocks

    # else:
    #     dp_dx = (p[m+1, n] - p[m-1, n])/(2*dx)
    #     ux_dx = (ux[m+1, n] - ux[m-1, n])/(2*dx)

    return dp_dx, ux_dx, ux_dr


# @jit(nopython=True)
def grad_ux2_matrix(p_in, p, ux_in, ux):  # bulk
    ux_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dp_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ux_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if n == 1:
                # NOTE: SYMMETRY CONDITION HERE done
                ux_dr[m, n] = (ux[m, n+2] - ux[m, n])/(4*dr)

            elif n == Nr-1 or n == Nr:
                ux_dr[m, n] = (ux[m, n] - ux[m, n-1])/dr  # BWD

            else:
                # upwind 1st order  - positive flow - advection
                ux_dr[m, n] = (ux[m, n] - ux[m, n-1])/(dr)  # CD

            if m == 1:
                dp_dx[m, n] = (p[m, n] - p_in)/dx  # BWD
                ux_dx[m, n] = (ux[m, n] - ux_in)/dx  # BWD

            elif m == Nx:
                dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx  # BWD
                ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx  # BWD

            else:
                # upwind 1st order  - positive flow - advection
                dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx
                ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx

    return dp_dx, ux_dx, ux_dr

# @numba.jit('f8(f8,f8,f8,f8,f8)')


@jit(nopython=True)
def grad_ur2(m, n, p, ur, ur_in):  # first derivatives BULK

    if n == 1:
        # NOTE: Symmetry BC done
        dp_dr = (p[m, n+2] - p[m, n])/(4*dr)
        ur_dr = (ur[m, n+2]-ur[m, n])/(4*dr)  # increased to 2dx

# n == Nr-1

    else:
        dp_dr = (p[m, n] - p[m, n-1])/dr  # BWD
        ur_dr = (ur[m, n] - ur[m, n-1])/dr

    # elif (n != 1 and n != Nr-1):
    #     dp_dr = (p[m, n+1] - p[m, n-1])/(2*dr)  # CD
    #     ur_dr = (ur[m, n+1] - ur[m, n-1])/(2*dr)

    # if m == 0:
    #     ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

    if m == 1:
        ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

    # elif (m <= n_trans+2 and m >= n_trans-2):
    #     ur_dx = (ur1[m-2, n] - 8*ur1[m-1, n] + 8 *
    #              ur1[m+1, n] - ur1[m+2, n])/(12*dx)  # 4 point CD

    elif m == Nx:
        ur_dx = (ur[m, n] - ur[m-1, n])/dx

    elif (m > 1 and m <= Nx - 2):
        # upwind 1st order  - positive flow - advection
        ur_dx = (ur[m, n] - ur[m-1, n])/dx

    else:
        # upwind 1st order  - positive flow - advection
        ur_dx = (ur[m, n] - ur[m-1, n])/(dx)  # CD

    return dp_dr, ur_dx, ur_dr


# @jit(nopython=True)
def grad_ur2_matrix(p, ur, ur_in):  # first derivatives BULK
    dp_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ur_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ur_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):

            if n == 1:
                # NOTE: Symmetry BC done
                dp_dr[m, n] = (p[m, n+2] - p[m, n])/(4*dr)
                ur_dr[m, n] = (ur[m, n+2]-ur[m, n])/(4*dr)  # increased to 2dx

        # n == Nr-1

            else:
                dp_dr[m, n] = (p[m, n] - p[m, n-1])/dr  # BWD
                ur_dr[m, n] = (ur[m, n] - ur[m, n-1])/dr

            # elif (n != 1 and n != Nr-1):
            #     dp_dr = (p[m, n+1] - p[m, n-1])/(2*dr)  # CD
            #     ur_dr = (ur[m, n+1] - ur[m, n-1])/(2*dr)

            # if m == 0:
            #     ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

            if m == 1:
                ur_dx[m, n] = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

            # elif (m <= n_trans+2 and m >= n_trans-2):
            #     ur_dx = (ur1[m-2, n] - 8*ur1[m-1, n] + 8 *
            #              ur1[m+1, n] - ur1[m+2, n])/(12*dx)  # 4 point CD

            elif m == Nx:
                ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/dx

            elif (m > 1 and m <= Nx - 2):
                # upwind 1st order  - positive flow - advection
                ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/dx

            else:
                # upwind 1st order  - positive flow - advection
                ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/(dx)  # CD

            return dp_dr, ur_dx, ur_dr


# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8)')
@jit(nopython=True)
def grad_e2(m, n, ur1, ux1, ux_in, e_in, e1):     # use upwind for Pe > 2

    if n == Nr:
        grad_r = (n*dr*ur1[m, n]*e1[m, n] -
                  (n-1)*dr*ur1[m, n-1]*e1[m, n-1])/dr  # BWD

    # We dont need the surface case, this is the bulk...

    if n == 1:
        # NOTE: Symmetry BC done
        grad_r = ((n+2)*dr*ur1[m, n+2]*e1[m, n+2] - n *
                  dr*ur1[m, n]*e1[m, n])/(4*dr)  # ur=0 @ r=0 #CD

# n == Nr-1:
    else:
        grad_r = ((n)*dr*ur1[m, n]*e1[m, n] - (n-1)
                  * dr*ur1[m, n-1]*e1[m, n-1])/(dr)  # BWD

    # if m == 0:
    #     grad_x = (e1[m+1, n]*ux1[m+1, n]-e_in*ux_in)/(dx)

    if m == Nx:
        # print("e1[m, n]*ux1[m, n]: ", e1[m, n]*ux1[m, n],
        #       "-e1[m-1, n]*ux1[m-1, n]: ", -e1[m-1, n]*ux1[m-1, n])
        grad_x = (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])/dx  # BWD

    # elif (m <= n_trans+2 and m >= n_trans-2):
    #     grad_x = (e1[m-2, n]*ux1[m-2, n] - 8*e1[m-1, n]*ux1[m-1, n] + 8 *
    #               e1[m+1, n]*ux1[m+1, n] - e1[m+2, n]*ux1[m+2, n])/(12*dx)
    elif (m >= 1 and m <= Nx - 2):
        # upwind 1st order  - positive flow - advection
        grad_x = (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])/dx
        # grad_x = 3*(e1[m, n]*ux1[m, n]) - 4*(e1[m-1, n]
        #                                      * ux1[m-1, n]) + (e1[m-2, n]
        #                                                        * ux1[m-2, n]) / dx  # upwind, captures shocks
    else:  # 0 < m < Nx,  1 < n < Nr
        grad_x = (e1[m, n]*ux1[m, n]-e1[m-1, n]
                  * ux1[m-1, n])/dx  # upwind

    return grad_x, grad_r


# @jit(nopython=True)
def grad_e2_matrix(ur1, ux1, ux_in, e_in, e1):     # use upwind for Pe > 2
    grad_r = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    grad_x = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            # We dont need the surface case, this is the bulk...

            if n == 1:
                # NOTE: Symmetry BC done
                grad_r[m, n] = ((n+2)*dr*ur1[m, n+2]*e1[m, n+2] - n *
                                dr*ur1[m, n]*e1[m, n])/(4*dr)  # ur=0 @ r=0 #CD

            # surface case
            if n == Nr:
                grad_r[m, n] = (n*dr*ur1[m, n]*e1[m, n] -
                                (n-1)*dr*ur1[m, n-1]*e1[m, n-1])/dr  # BWD

        # n == Nr-1:
            else:
                grad_r[m, n] = ((n)*dr*ur1[m, n]*e1[m, n] - (n-1)
                                * dr*ur1[m, n-1]*e1[m, n-1])/(dr)  # BWD

            # if m == 0:
            #     grad_x = (e1[m+1, n]*ux1[m+1, n]-e_in*ux_in)/(dx)

            if m == Nx:
                # print("e1[m, n]*ux1[m, n]: ", e1[m, n]*ux1[m, n],
                #       "-e1[m-1, n]*ux1[m-1, n]: ", -e1[m-1, n]*ux1[m-1, n])
                grad_x[m, n] = (e1[m, n]*ux1[m, n]-e1[m-1, n]
                                * ux1[m-1, n])/dx  # BWD

            # elif (m <= n_trans+2 and m >= n_trans-2):
            #     grad_x = (e1[m-2, n]*ux1[m-2, n] - 8*e1[m-1, n]*ux1[m-1, n] + 8 *
            #               e1[m+1, n]*ux1[m+1, n] - e1[m+2, n]*ux1[m+2, n])/(12*dx)
            elif (m >= 1 and m <= Nx - 2):
                # upwind 1st order  - positive flow - advection
                grad_x[m, n] = (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])/dx
                # grad_x = 3*(e1[m, n]*ux1[m, n]) - 4*(e1[m-1, n]
                #                                      * ux1[m-1, n]) + (e1[m-2, n]
                #                                                        * ux1[m-2, n]) / dx  # upwind, captures shocks
            else:  # 0 < m < Nx,  1 < n < Nr
                grad_x[m, n] = (e1[m, n]*ux1[m, n]-e1[m-1, n]
                                * ux1[m-1, n])/dx  # upwind

    return grad_x, grad_r


# @numba.jit('f8(f8,f8,f8,f8)')
@jit(nopython=True)
def dt2nd_radial(ux1, ur1, m, n):
    if n == 1:
        # NOTE: Symmetry Boundary Condition assumed for ur1 radial derivative along x axis..
        # --------------------------- dt2nd radial ux1 ---------------------------------#
        dt2nd_radial_ux1 = (ux1[m, n+2] - ux1[m, n]) / (4*dr**2)

        # --------------------------- dt2nd radial ur1 ---------------------------------#
        dt2nd_radial_ur1 = (ur1[m, n+2] - ur1[m, n]) / (4*dr**2)

        # print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
        # print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

    else:  # (n is between 1 and Nr)

        # --------------------------- dt2nd radial ux1 ---------------------------------#
        dt2nd_radial_ux1 = (ux1[m, n+1] + ux1[m, n-1] -
                            2*ux1[m, n])/(dr**2)  # CD
    # --------------------------- dt2nd radial ur1 ---------------------------------#
        dt2nd_radial_ur1 = (ur1[m, n+1] + ur1[m, n-1] -
                            2*ur1[m, n])/(dr**2)  # CD
        # print("dt2nd_radial_ur1:", dt2nd_radial_ur1)
    return dt2nd_radial_ux1, dt2nd_radial_ur1


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
def dt2r_matrix(ux1, ur1):
    dt2r_ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dt2r_ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):

            if n == 1:
                # NOTE: Symmetry Boundary Condition assumed for ur1 radial derivative along x axis..
                # --------------------------- dt2nd radial ux1 ---------------------------------#
                dt2r_ux1[m, n] = (ux1[m, n+2] - ux1[m, n]) / (4*dr**2)

                # --------------------------- dt2nd radial ur1 ---------------------------------#
                dt2r_ur1[m, n] = (ur1[m, n+2] - ur1[m, n]) / (4*dr**2)

                # print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
                # print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

            elif n == Nr:
                # --------------------------- dt2nd radial ux1 ---------------------------------#
                # NOTE: CHECK
                dt2r_ux1[m, n] = (2*ux1[m, n] - 5*ux1[m, n-1] +
                                  4*ux1[m, n-2] - ux1[m, n-3]) / (dr**2)

                # --------------------------- dt2nd radial ur1 ---------------------------------#
                dt2r_ur1[m, n] = (2*ur1[m, n] - 5*ur1[m, n-1] +
                                  4*ur1[m, n-2] - ur1[m, n-3]) / (dr**2)
            else:  # (n is between 1 and Nr)

                # --------------------------- dt2nd radial ux1 ---------------------------------#
                dt2r_ux1[m, n] = (ux1[m, n+1] + ux1[m, n-1] -
                                  2*ux1[m, n])/(dr**2)  # CD
            # --------------------------- dt2nd radial ur1 ---------------------------------#
                dt2r_ur1[m, n] = (ur1[m, n+1] + ur1[m, n-1] -
                                  2*ur1[m, n])/(dr**2)  # CD
                # print("dt2nd_radial_ur1:", dt2nd_radial_ur1)
    return dt2r_ux1, dt2r_ur1

# @numba.jit('f8(f8,f8,f8,f8,f8,f8)')


@jit(nopython=True)
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


def dt2x_matrix(ux_in, ur_in, ux1, ur1):
    dt2x_ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dt2x_ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if m == 0:
                # --------------------------- dt2nd axial ux1 ---------------------------------#
                dt2x_ux1[m, n] = (ux_in - 2*ux1[m, n] + ux1[m+1, n]) / (dx**2)

            # --------------------------- dt2nd axial ur1 ---------------------------------#
                dt2x_ur1[m, n] = (-ur_in + ur_in - 30 *
                                  ur1[m, n] + 16*ur1[m+1, n] - ur1[m+2, n])/(12*dx**2)

            elif m == Nx:
                # --------------------------- dt2nd axial ux1 ---------------------------------#

                dt2x_ux1[m, n] = (ux1[m-2, n] - 2*ux1[m-1, n] +
                                  ux1[m, n])/(dx**2)  # BWD
                # --------------------------- dt2nd axial ur1 ---------------------------------#
            # Three-point BWD
                dt2x_ur1[m, n] = (
                    ur1[m-2, n] - 2*ur1[m-1, n] + ur1[m, n])/(dx**2)

            else:
                # --------------------------- dt2nd axial ux1 ---------------------------------#
                dt2x_ux1[m, n] = (ux1[m+1, n] + ux1[m-1, n] -
                                  2*ux1[m, n])/(dx**2)  # CD

            # --------------------------- dt2nd axial ur1 ---------------------------------#
                dt2x_ur1[m, n] = (ur1[m+1, n] + ur1[m-1, n] -
                                  2*ur1[m, n])/(dx**2)  # CD
    return dt2x_ux1, dt2x_ur1


@jit(nopython=True)
def f_ps(ts):
    #   Calculate saturated vapor pressure (Pa)
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


# @numba.jit('f8(f8)')
@jit(nopython=True)
def f_ts(ps):
    #   Calculate saturated vapor temperature (K)
    print("Ps for f_ts calc: ", ps)
    ps1 = np.log(ps/100000.0)
    t_sat = 74.87701+6.47033*ps1+0.45695*ps1**2+0.02276*ps1**3+7.72942E-4*ps1**4+1.77899E-5 * \
        ps1**5+2.72918E-7*ps1**6+2.67042E-9*ps1**7+1.50555E-11*ps1**8+3.71554E-14*ps1**9
    return t_sat


# @numba.jit('f8(f8,f8)')
@jit(nopython=True)
def delta_h(tg, ts):
    #   Calculate sublimation heat of nitrogen (J/kg)  ## needed for thermal resistance of SN2 layer when thickness is larger than reset value.
    print("Tg, Ts for delta_h calc: ", tg, ts)
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


# @numba.jit('f8(f8)')
@jit(nopython=True)
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


# @numba.jit('f8(f8)')
@jit(nopython=True)
def v_m(tg):
    #   Calculate arithmetic mean speed of gas molecules (m/s)
    print("Tg for v_m gas: ", tg)
    v_mean = np.sqrt(8.*R*tg/np.pi/M_n)
    # ipdb.set_trace()
    return v_mean


# @numba.jit('f8(f8)')
@jit(nopython=True)
def c_c(ts):
    #   Calculate the heat capacity of copper (J/(kg*K))
    # print("Ts for c_c (specific heat copper) calc: ", ts)
    #  print("ts",ts)
    c_copper = 1.22717-10.74168*np.log10(ts)**1+15.07169*np.log10(
        ts)**2-6.69438*np.log10(ts)**3+1.00666*np.log10(ts)**4-0.00673*np.log10(ts)**5
    c_copper = 10.**c_copper
    return c_copper


# @numba.jit('f8(f8)')
@jit(nopython=True)
def k_cu(T):
    #   Calculate the coefficient of thermal conductivity of copper (RRR=10) (W/(m*K)) (for pde governing copper wall, heat conducted in the x axis.)
    # print("Tw for k_cu copper: ", T)
    k1 = 3.00849+11.34338*T+1.20937*T**2-0.044*T**3+3.81667E-4 * \
        T**4+2.98945E-6*T**5-6.47042E-8*T**6+2.80913E-10*T**7
    k2 = 1217.49161-13.76657*T-0.01295*T**2+0.00188*T**3-1.77578E-5 * \
        T**4+7.58474E-8*T**5-1.58409E-10*T**6+1.31219E-13*T**7
    k3 = k2+(k1-k2)/(1+np.exp((T-70)/1))
    return k3


# @numba.jit('f8(f8,f8)')
@jit(nopython=True)
def D_nn(T_g, P_g):
    #   Calculate self mass diffusivity of nitrogen (m^2/s)
    if T_g > 63:
        D_n_1atm = -0.01675+4.51061e-5*T_g**1.5
    else:
        D_n_1atm = (-0.01675+4.51061e-5*63**1.5)/63**1.5*T_g**1.5
    D_n_p = D_n_1atm*101325/P_g
    D_n_p = D_n_p/1e4
    return D_n_p


# @numba.jit('f8(f8,f8)')
@jit(nopython=True)
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
@jit(nopython=True)
def gamma(a):
    #   Calculate the correction factor of mass flux
    gam1 = np.exp(-np.power(a, 2.))+a*np.sqrt(np.pi)*(1+math.erf(a))
    return gam1


# @numba.jit('f8(f8,f8,f8,f8,f8)')
@jit(nopython=True)
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


@jit(nopython=True)
def bulk_values(T_s):
    T_0 = T_s
    rho_0 = 1e-3  # An arbitrary small initial density in pipe, kg/m3
    p_0 = rho_0/M_n*R*T_0  # Initial pressure, Pa
    e_0 = 5./2.*rho_0/M_n*R*T_0  # Initial internal energy
    ux_0 = 0
    bulk = [T_0, rho_0, p_0, e_0, ux_0]
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


@jit(nopython=True)
def integral_mass_delSN(de):
    de0 = np.zeros((Nx+1), dtype=(np.float64))
    # Integrate deposition mass
    for m in np.arange(Nx+1):
        de0[m] += dt*np.pi*D*de[m]

# Calculate the SN2 layer thickness
    del_SN = de0/np.pi/D/rho_sn
    # print("del_SN: ", del_SN)

    return de0, del_SN  # the de0 is incremented and never restarted

# recalculates Tg to be equal to Ts.
# NOTE: does this affect the velocities? does mde change? and if yes, does it mean ur changes?


def gas_surface_temp_check(T, Ts, ur, e, u, rho):
    for m in np.arange(np.int64(1), np.int64(Nx+1)):
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
    e, p = balance_energy2(rho, T, u)
    return T, e, p, rho, u


@jit(nopython=True)
def Cu_Wall_function(ur, Tx, Twx, Tcx, Tsx, T_in, delSN, de):
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

    for m in np.arange(Nx+1):
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
        Tw2[m] = Twx[m] + dt/(w_coe*c_c(Twx[m]))*(qi[m]-q_h(Twx[m], BW_coe)
                                                  * Do/D) + dt/(rho_cu*c_c(Twx[m]))*k_cu(Twx[m])*dt2nd_w_m[m]
    #     print("Tw2: ", Tw2[m])
    #     check_negative(Tw2[m], n)

# q deposited into frost layer. Nusselt convection neglected
# NOTE: Check this addition operation, is this correct? 2d and 1d rows

    q_dep[:] = de[:]*(1/2*(ur[:, Nr])**2 + delta_h(Tx[:, Nr], Tsx[:]))

    # NOTE: delta_h will change if T =Ts and will be zero.
    # NOTE: Check this logic, very important
    for j in np.arange(Nx+1):
        if Tx[j] < Tsx[j]:
            q_dep[j] = de[j]*(1./2.*ur[j, Nr])**2

# SN2 Center layer Tc equation
    for j in np.arange(Nx+1):
        Tc2[j] = Tcx[j] + dt * (q_dep[j]-qi[j]) / \
            (rho_sn * c_n(Tsx[j])*delSN[j])
    # print("Tc2: ", Tc2[m, n])
    # check_negative(Tc2[m], n)

# Calculate SN2 surface temp
    Ts2 = 2*Tcx - Twx
    # print("Ts2: ", Ts2[m])
    # check_negative(Ts2[m], n)
    for m in np.arange(Nx+1):
        qhe[m] = q_h(Twx[m], BW_coe)

    return Tw2, Ts2, Tc2, qhe, dt2nd_w_m


@jit(nopython=True)
def surface_BC(ux):
    ux[:, Nr] = 0
    return ux


def parabolic_velocity(T, ux, ux_in):
    for i in np.arange(n_trans):
        # diatomic gas gamma = 7/5   WE USED ANY POINT, since this preparation area is constant along R direction.
        # any temperature works, they are equl in the radial direction
        v_max = np.sqrt(7./5.*R*T[i, 4]/M_n)
        for y in np.arange(Nr+1):
            # a = v_max
            a = ux_in
            # a = v_max*(1.0 - ((y*dr)/R_cyl)**2)
            # print("parabolic y", y)
            ux[i, y] = a
            u[i, y] = ux[i, y]

    # print("parabolic ux at center: ", ux1[i, 0])
    out = ux, u
    return out


@jit(nopython=True)
def smoothing_inlet(p, rho, T, ux, ur, ux_in, u, p_in, p_0, rho_in, rho_0, n_trans):
    for i in range(0, Nx+1):
        p[i, :] = exp_smooth(i+n_trans, p_in*2.-p_0, p_0, 0.4, n_trans)
    # print("P1 smoothing values", p1[i,:])
        rho[i, :] = exp_smooth(i + n_trans, rho_in*2 -
                               rho_0, rho_0, 0.4, n_trans)
    #    T1[i, :] = T_neck(i)
        # if i<51: T1[i]=T_in
        T[i, :] = p[i, :]/rho[i, :]/R*M_n
        ux[i, :] = exp_smooth(i + n_trans, ux_in*2, 0, 0.4, n_trans)
        u = np.sqrt(ux**2. + ur**2.)
        # v_max = np.sqrt(7./5.*R*T/M_n)  # diatomic gas gamma = 7/5
    #    u1[i, :] = exp_smooth(i + n_trans, ux_in*2, 0, 0.4, n_trans)

        # if i < n_trans+1:
        #     e1[i, :] = 5./2.*p1[i, :]+1./2.*rho1[i, :]*u1[i, :]**2

    #        rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]  # IDEAL GAS LAW

        # print("p1 matrix after smoothing", p1)
        # else:
        #     e1[i, :] = 5/2*rho1[i, :]/M_n*R*T_in+1/2**rho1[i, :]*u1[i, :]**2
    e, T = balance_energy(p, rho, u)
    # for i in range(0, Nx+1):
    out = p, rho, T, ux, u, e
    return out

# no slip BC at surface of pipe

# remove timestepping folder


def remove_timestepping():
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "timestepping"
        path = os.path.join(location, dir)
        shutil.rmtree(path)


@jit(nopython=True)
def no_slip(ux, u, p, rho):
    ux[:, Nr] = 0
# NOTE: Check radial velocity.... does this change mdot?
    # u[:, Nr] = 0
# NOTE: How do i recalculate u?
    e, T = balance_energy(p, rho, u)
    return ux, u, e, T


# @jit(nopython=True)
def inlet_BC(ux, ur, u, p, rho, T, e, p_inl, ux_inl, rho_inl, T_inl, e_inl, Tw, Ts, Tc):
    ux[0, :] = ux_inl
    u[0, :] = ux_inl
    p[0, :] = p_inl
    rho[0, :] = rho_inl
    T[0, :] = T_inl
    e[0, :] = e_inl
    Tw[0] = T_inl
    Ts[0] = T_inl
    Tc[0] = T_inl
    u = np.sqrt(ux**2. + ur**2.)

    e, T = balance_energy(p, rho, u)

    return [ux, ur, u, p, rho, T, e, Tw, Ts, Tc]


@jit(nopython=True)
def outlet_BC(p, e, rho, ux, ur, u, rho_0):

    for n in np.arange(Nr):
        p[Nx, n] = 2/5*(e[Nx, n]-1/2*rho[Nx, n]
                        * u[Nx, n]**2)  # Pressure

        rho[Nx, n] = max(2*rho[Nx-1, n]-rho[Nx-2, n], rho_0)  # Free outflow
        ux[Nx, n] = max(2*rho[Nx-1, n]*ux[Nx-1, n] -
                        rho[Nx-2, n]*ux[Nx-2, n], 0) / rho[Nx, n]
        u = np.sqrt(ux**2. + ur**2.)
        # e[Nx, n] = 2*e[Nx-1, n]-e[Nx-2, n]
    e, T = balance_energy(p, rho, u)

    bc = [p, rho, ux, u, e]

    # NOTE: check input de to the m_de equation.
    # de1[Nx] = m_de(T2[Nx, n], p1[Nx, n], Ts2[Nx], de1[Nx], 0.)
    # del_SN = de0[Nx]/np.pi/D/rho_sn
    #     if del_SN > 1e-5:
    #         q1 = k_sn*(Tw1[Nx]-Ts1[Nx])/del_SN
    # #           print("line 848", "q1", q1,"Tw1", Tw1[Nx],"Ts1", Ts1[Nx],"ksn", k_sn)
    #         assert not math.isnan(Ts1[Nx])
    #         Ts2[Nx] = Ts1[Nx]+dt/(w_coe*c_c(Tw1[Nx])) * \
    #             (q1-q_h(Tw1[Nx], BW_coe)*Do/D)
    #         Tc2[Nx] = Tc1[Nx]+dt/(de0[m]*c_n(Tc1[Nx])/D/np.pi)*(de1[Nx]
    #                                                             * (1/2*(u1[Nx, n])**2+delta_h(T1[Nx, n], Ts1[Nx])-q1))
    #     else:
    #         q1 = de1[Nx]*(1/2*(u1[Nx, n])**2+delta_h(T1[Nx, n], Ts1[Nx]))
    #         Ts2[Nx] = Ts1[Nx]+dt/(w_coe*c_c(Tw1[Nx])) * \
    #             (q1-q_h(Tw1[Nx], BW_coe)*Do/D)
    #         Tc2[Nx] = Ts2[Nx]
    # Tw2[Nx] = 2*Tc2[Nx]-Ts2[Nx]
    # qhe[Nx] = q_h(Tw1[Nx], BW_coe)*np.pi*Do
    # de0[Nx] += dt*np.pi*D*de1[Nx]
    return bc

# recalculating energy from pressure and velocity


@jit(nopython=True)
def balance_energy(p, rho, u):
    e = 5./2. * p + 1./2 * rho*u**2
    T = p/rho/R*M_n
    return e, T


# recalculating energy from rho, T, and velocity
@jit(nopython=True)
def balance_energy2(rho, T, u):
    e = 5./2. * rho*R*T/M_n + 1./2 * rho*u**2
    p = rho*R*T/M_n
    return e, p


@jit(nopython=True)
def val_in_constant():
    #   Calculate instant flow rate (kg/s)
    p_in = 1000.
    T_in = 298.
    rho_in = p_in / T_in/R*M_n
    ux_in = 30.
    ur_in = 0.
    e_in = 5./2.*rho_in/M_n*R*T_in + 1./2.*rho_in*ux_in**2
    out = np.array([p_in, ux_in, ur_in, rho_in, e_in, T_in])
    return out

# @numba.jit('f8(f8)')


@jit(nopython=True)
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
@jit(nopython=True)
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
    for m in np.arange(np.int64(1), np.int64(Nx+1)):
        for n in np.arange(np.int64(1), np.int64(Nr+1)):
            pe[m, n] = u[m, n]*D_hyd / mu_n(T[m, n], p[m, n])
    return pe

# de1[m] = m_de(T1[m, n], p1[m, n], Tw1[m], de1[m], rho1[m, n]*ur1[m, n]-rho1[m, n-1]*ur1[m, n-1])

# NOTE: I am getting wrong mass deposition values... from 1d it is in the order of e-6


# returns mass deposition rate to put in de1 matrix
# @numba.jit('f8(f8,f8,f8,f8,f8)')
@jit(nopython=True)
def m_de(T, P, T_s, de, dm):
    p_0 = bulk_values(T_s)[2]
    # print("mdot calc: ", "Tg: ", T, " P: ",
    #       P, "Ts: ", T_s, "de: ", de, "dm: ", dm)
    #   Calculate deposition rate (kg/(m^2*s))
    if T == 0:
        T = 0.00001
    rho = P*M_n/R/T
# no division by zero
    if rho == 0:
        rho = 0.00001
    v_m1 = np.sqrt(2*R*T/M_n)  # thermal velocity of molecules
    u_mean1 = de/rho  # mean flow velocity towards the wall.
    beta = u_mean1/v_m1  # this is Beta from Hertz Knudson
    gam1 = gamma(beta)  # deviation from Maxwellian velocity.
    P_s = f_ps(T_s)

    if P > P_s and P > p_0:
        # Correlated Hertz-Knudsen Relation #####
        m_out = np.sqrt(M_n/2/np.pi/R)*Sc_PP * \
            (gam1*P/np.sqrt(T)-P_s/np.sqrt(T_s))
        if T_s > 25:
            print("P>P0, P>Ps")
            # Arbitrary smooth the transition to steady deposition
            # NOTE: Check this smoothing function.
            m_out = m_out*exp_smooth(T_s-25., 1., 0.05,
                                     0.03, (f_ts(P*np.sqrt(T_s/T))-25.)/2.)

        # Speed of sound limit for the condensation flux
        rho_min = p_0*M_n/R/T
        # sqrt(7./5.*R*T/M_n)*rho
        # Used Conti in X-direction, since its absolute flux.
        m_max = D/4./dt*(rho-rho_min)-D/4./dx*dm
        if m_out > m_max:
            m_out = m_max
            # print("mout = mmax")
    else:
        m_out = 0
    rho_min = p_0*M_n/R/T
    m_out = 0  # NO HEAT TRANSFER/ MASS DEPOSITION CASE
    return m_out  # Output: mass deposition flux, no convective heat flux


# @numba.jit('f8(f8,f8)')
@jit(nopython=True)
def q_h(tw, BW_coe):
    # Boiling heat transfer rate of helium (W/(m^2*K))
    # delT = ts-4.2
    delT = tw-4.2

    q_con = 0.375*1000.*delT  # Convection
    q_nu = 58.*1000.*(delT**2.5)  # Nucleate boiling
    q_tr = 7500.  # Transition to film boiling
    print("qcond: ", q_con, "q_nu: ", q_nu, "q_tr: ", q_tr)
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
    print("rate of heat transfer to helium:", q_he)
    print("q_h calc: ", q_he, "Tw: ", tw)
    q_he = 0  # NO HEAT TRANSFER CASE
    return q_he

# Initialization


def save_initial_conditions(rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1, de1, pe, Tc1):
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
    np.savetxt("tw.csv", Tw1, delimiter=",")
    np.savetxt("ts.csv", Ts1, delimiter=",")
    np.savetxt("tc.csv", Tc1, delimiter=",")
    np.savetxt("de.csv", de0, delimiter=",")
    np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    np.savetxt("pe.csv", pe, delimiter=",")


def save_data(tx, dt, rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1, de1, pe):
    increment = (tx+1)*dt

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(increment) + '/'
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
    # np.savetxt("tw.csv", Tw1, delimiter=",")
    # np.savetxt("ts.csv", Ts1, delimiter=",")
    # np.savetxt("de_mass.csv", de0, delimiter=",")
    # np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    np.savetxt("peclet.csv", pe, delimiter=",")


def namestr(obj, namespace):  # Returns variable name for check_negative function
    return [name for name in namespace if namespace[name] is obj]


def check_array(array_in):
    if np.any(array_in < 0):
        array_name = namestr(array_in, globals())[0]
        print(array_name + " has at least one negative value.")
        exit()


def delete_surface_inviscid(rho, ux, ur, u, e, T, p, pe):
    rho3 = np.delete(rho, Nr-1, axis=1)
    ux3 = np.delete(ux, Nr-1, axis=1)
    ur3 = np.delete(ur, Nr-1, axis=1)
    u3 = np.delete(u, Nr-1, axis=1)
    e3 = np.delete(e, Nr-1, axis=1)
    T3 = np.delete(T, Nr-1, axis=1)
    p3 = np.delete(p, Nr-1, axis=1)
    pe3 = np.delete(pe, Nr-1, axis=1)
    return [rho3, ux3, ur3, u3, e3, T3, p3, pe3]


def delete_r0_point(rho2, ux2, ur2, u2, e2, T2, p1, pe):
    rho3 = np.delete(rho2, 0, axis=1)
    ux3 = np.delete(ux2, 0, axis=1)
    ur3 = np.delete(ur2, 0, axis=1)
    u3 = np.delete(u2, 0, axis=1)
    e3 = np.delete(e2, 0, axis=1)
    T3 = np.delete(T2, 0, axis=1)
    p3 = np.delete(p1, 0, axis=1)
    pe3 = np.delete(pe, 0, axis=1)
    return [rho3, ux3, ur3, u3, e3, T3, p3, pe3]


if __name__ == '__main__':
    t_sat = 70
    # p_test = f_ps(t_sat)
    # print("p_test", p_test)

    p_sat = 10000
    # t_test = f_ts(p_sat)
    # print("t_test", t_test)

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

    tg = 298
    ts = 4.2
    # print("delta_h ", delta_h(tg, ts))

# ------------------------- specific heat of solid nitrogen (J/(kg*K)) ------------------------------- #

    # print("c_n ", c_n(ts))
# ------------------------- thermal velocity ------------------------------- #

    # print("vm ", v_m(tg))

# ------------------------- heat capacity of copper (J/(kg*K)) ------------------------------- #

    # print("c_c ", c_c(ts))

# ------------------------- coefficient of thermal conductivity of copper (RRR=10) (W/(m*K)) ------------------------------- #

    T = 4.2
    # print("k_cu", k_cu(T))

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
    BW_coe = 0.02
    print("q_h ", q_h(tw, BW_coe))

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


# def check_negative(var_in, n):  # CHECKS CALCULATIONS FOR NEGATIVE OR NAN VALUES
#     # at surface
#     if n == Nr:

#         if var_in < 0:
#             print("negative Surface", var_in)
#             print("line ", get_linenumber())
#             exit()
#         if math.isnan(var_in):
#             print("NAN Surface ", var_in)
#             print("line ", get_linenumber())
#             assert not math.isnan(var_in)

#     # at BULK
#     else:
#         S1 = "Bulk"

#         if var_in < 0:
#             print("negative Bulk ", var_in)
#             print("line ", get_linenumber())
#             exit()
#         if math.isnan(var_in):
#             print("NAN Bulk ", var_in)
#             print("line ", get_linenumber())
#             assert not math.isnan(var_in)


# def check_negative(var_in, var_name, n):  # CHECKS CALCULATIONS FOR NEGATIVE OR NAN VALUES
#     #    value_name = locals().items()
#     # value_name = varname(var_in)
#     if type(var_in) == list:
#         value_name = namestr(var_name, globals())

#     # elif type(var_in)== np.float64:
#     #     value_name = namestr(var_name, globals())[0]
#     else:
#         value_name = namestr(var_name, globals())[0]

#     print("variable name", value_name)

#     # at surface
#     if n == Nr:
#         S1 = "Surface"

#         if var_in < 0:
#             print("negative " + value_name + " in " + S1, var_in)
#             exit()
#         if math.isnan(var_in):
#             print("NAN " + value_name + " in " + S1)
#             assert not math.isnan(var_in)

#     # at BULK
#     else:
#         S1 = "Bulk"


#         if var_in < 0:
#             print("negative " + value_name + " in " + S1, var_in)
#             exit()
#         if math.isnan(var_in):
#             print("NAN " + value_name + " in " + S1)
#             assert not math.isnan(var_in)
