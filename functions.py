# ----------------- Helper Functions --------------------------------#

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import re
import inspect
from my_constants import *
import os
import shutil
import numba
from numba import jit
import numpy as np
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

# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8)')

# def tvdrk3(nx,ny,nz,dx,dy,dz,q,dt,ivis,iflx,Re)
#     qq = copy(q)
#     qn = copy(q)

#     #First step
#     !(q,nx,ny,nz)
#     r  = rhs(nx,ny,nz,dx,dy,dz,q,ivis,iflx,Re)
#     qq = q + dt*r

#     #Second step
#     expbc!(qq,nx,ny,nz)
#     r  = rhs(nx,ny,nz,dx,dy,dz,qq,ivis,iflx,Re)
#     qq = 0.75*q + 0.25*qq + 0.25*dt*r

#     #Third Step
#     expbc!(qq,nx,ny,nz)
#     r  = rhs(nx,ny,nz,dx,dy,dz,qq,ivis,iflx,Re)
#     qn = 1/3*q + 2/3*qq + 2/3*dt*r

#     return qn
# end



# #Calculate time step
# def calc_dt(cfl,γ,q,nx,ny,nz,dx,dy,dz)
#     a = 0.0
#     a = maximum([a,0.0])
#     for k in 0:nz
#         for j in 0:ny
#             for i in 0:nx
#                 ρ,ρu,ρv,ρw,ρe = q[:,i,j,k]
#                 u,v,w,e       = ρu/ρ, ρv/ρ, ρw/ρ, ρe/ρ
#                 p = ρ*(γ-1)*(e-0.5*(u^2+v^2+w^2))
#                 c = sqrt(γ*p/ρ)
#                 a = maximum([a,abs(u),abs(u+c),abs(u-c)
#                              ,abs(v),abs(v+c),abs(v-c)
#                              ,abs(w),abs(w+c),abs(w-c)])

#             end
#         end
#     end

#     dt = cfl* minimum([dx,dy,dz])/a

#     return dt
# end

def plot_imshow(p, ux, T, rho, e):
        
    fig, axs = plt.subplots(5)
    fig.suptitle('Initial fields along tube for all R')

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
        a = rho[m, n]
        m_dx = (rho[m, n]*ux[m, n]-rho_in*ux_in)/dx

    elif m == Nx:
        a = rho[m, n]
        m_dx = (rho[m, n]*ux[m, n]-rho[m-1, n]*ux[m-1, n])/dx

    # elif (m <= n_trans+2 and m >= n_trans+2):
    #     # NOTE Use four point CD at transition point.
    #     a = rho1[m, n]
    #     m_dx = (rho1[m-2, n] - 8*rho1[m-1, n] + 8 *
    #             rho1[m+1, n] - rho1[m+2, n])/(12*dx)

    else:
        a = rho[m, n]
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

    return a, d_dr, m_dx


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


# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8)')
@jit(nopython=True)
def grad_e2(m, n, ur1, ux1, ux_in, e_in, e1):     # use upwind for Pe > 2

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


# @numba.jit('f8(f8)')
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
    print("Ts for c_n specific heat SN2 calc: ", ts)
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
    print("Ts for c_c (specific heat copper) calc: ", ts)
    #  print("ts",ts)
    c_copper = 1.22717-10.74168*np.log10(ts)**1+15.07169*np.log10(
        ts)**2-6.69438*np.log10(ts)**3+1.00666*np.log10(ts)**4-0.00673*np.log10(ts)**5
    c_copper = 10.**c_copper
    return c_copper


# @numba.jit('f8(f8)')
@jit(nopython=True)
def k_cu(T):
    #   Calculate the coefficient of thermal conductivity of copper (RRR=10) (W/(m*K)) (for pde governing copper wall, heat conducted in the x axis.)
    print("Tw for k_cu copper: ", T)
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
    mu_n_2 = 0
    mu_n_1 = 0
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
def bulk_values():
    T_0 = 298.
    rho_0 = 1e-2  # An arbitrary small initial density in pipe, kg/m3
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

    # for i in range(0, Nx+1):
    out = p, rho, T, ux, u
    return out


@jit(nopython=True)
def inlet_BC(ux, ur, u, p, rho, T, e, p_inl, ux_inl, rho_inl, T_inl, e_inl):
    ux[0, :] = ux_inl
    u[0, :] = ux_inl
    p[0, :] = p_inl
    rho[0, :] = rho_inl
    T[0, :] = T_inl
    e[0, :] = e_inl
    u = np.sqrt(ux**2. + ur**2.)
# val_in(i)
# Tw2[0] = T_inl
# Ts2[0] = T_inl
# Tc2[0] = T_inl
# NOTE: BC INIT
    # Ts1[:] = T1[:, Nr]
    # Ts2[:] = Ts1
    # Tw1[:] = T_s
    # Tw2[:] = T_s
    # Tc1[:] = T_s
    # Tc2[:] = T_s
    return [ux, ur, u, p, rho, T, e]


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
        e = 5./2.*rho/M_n*R*T + 1./2.*rho*u**2

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

# def recalculating_fields(p,rho,T):
    # recalculating fields:


@jit(nopython=True)
def val_in_constant():
    #   Calculate instant flow rate (kg/s)
    p_in = 1000.
    T_in = 298.
    rho_in = p_in / T_in/R*M_n
    ux_in = 1.
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


# @numba.jit('f8(f8,f8,f8,f8,f8)')
@jit(nopython=True)
def m_de(T, P, T_s, de, dm):
    p_0 = bulk_values[2]
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
    # print("Saturation pressure at this Ts", P_s)

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
  # sqrt(7./5.*R*T/M_n)*rho
        # print("m_max_sound:", m_max, "rho", rho, "rho_min", rho_min)
#        m_max = ((rho-rho_min)/dt - 1/(Nr*dr**2)*dm) * \
 #           D/4.         # From continuity equation
#        m_max = 2.564744575054553e-26  #NOTE: added to limit condensation rate...
#        print("m_max_sound:",m_max)
        # print("saturation temp: ", f_ts(P*np.sqrt(T_s/T)))
        # print("mout calculated: ", m_out)
        if m_out > m_max:
            m_out = m_max
            print("mout = mmax")
    else:
        # print("P<P0")
        m_out = 0
#    m_out = 0
    rho_min = p_0*M_n/R/T
#    m_max = ((rho-rho_min)/dt - 1/(Nr*dr**2)*dm) * \
#       D/4.         # From continuity equation
    # print("m_max_sound:", m_max, "rho", rho, "rho_min", rho_min)

    print("mout final: ", m_out)
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


# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)')
def save_initial_conditions(rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1, de1, pe):
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
    # np.savetxt("de.csv", de0, delimiter=",")
    # np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    np.savetxt("pe.csv", pe, delimiter=",")


# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)')
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


# @numba.jit('f8(f8,f8)')
def namestr(obj, namespace):  # Returns variable name for check_negative function
    return [name for name in namespace if namespace[name] is obj]


# @numba.jit('f8(f8)')
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


# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8)')
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
