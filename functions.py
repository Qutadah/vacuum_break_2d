# ----------------- Helper Functions --------------------------------#

import re
import inspect
from my_constants import *
import os
import shutil

import numpy as np
from numba import jit
# from scipy.ndimage.filters import laplace
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

u_in_x = np.sqrt(7./5.*R*T_in/M_n)*1.0  # Inlet velocity, m/s (gamma*RT)


def dt2nd_radial(ux1,ur1, dr,m,n):
    if n == 1:
    # NOTE: Symmetry Boundary Condition assumed for ur1 radial derivative along x axis..
    # --------------------------- dt2nd radial ux1 ---------------------------------#
        grad_ux1 = (ux1[m, n+2] - ux1[m, n])/(4*dr)
        dt2nd_radial_ux1 = (ux1[m, n+2] - ux1[m, n]) / (4*dr**2)

        # --------------------------- dt2nd radial ur1 ---------------------------------#
        grad_ur1 = (ur1[m, n+2] - ur1[m, n])/(4*dr)
        dt2nd_radial_ur1 = (ur1[m, n+2] - ur1[m, n]) / (4*dr**2)

        print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
        print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

    else:  # (n is between 1 and Nr)

# --------------------------- dt2nd radial ux1 ---------------------------------#
        dt2nd_radial_ux1 = (ux1[m, n+1] + ux1[m, n-1] - 2*ux1[m, n])/dr**2  # CD
    # --------------------------- dt2nd radial ur1 ---------------------------------#
        dt2nd_radial_ur1 = (ur1[m, n+1] + ur1[m, n-1] - 2*ur1[m, n])/(dr**2)  # CD
        print("dt2nd_radial_ur1:", dt2nd_radial_ur1)    
    return dt2nd_radial_ux1, dt2nd_radial_ur1

def dt2nd_axial(ux_in, ur_in, ux1, ur1, m, n, dx):    
    if m == 0:
    # --------------------------- dt2nd axial ux1 ---------------------------------#
        dt2nd_axial_ux1 = (ux_in - 2*ux1[m, n] + ux1[m+1, n]) / (dx**2)
        # dt2nd_axial_ux1 = (ux1[m+2,n] -2*ux1[m+1,n] + ux1[m,n])/(dx**2) #FWD

    # --------------------------- dt2nd axial ur1 ---------------------------------#
                        #                        dt2nd_axial_ur1 = (ur1[m+2,n] -2*ur1[m+1,n] + ur1[m,n])/(dx**2) #FWD
                        # FWD
        dt2nd_axial_ur1 = (-ur_in + ur_in - 30 * ur1[m, n] + 16*ur1[m+1, n] - ur1[m+2, n])/(12*dx**2)
        print("dt2nd_axial_ur1:", dt2nd_axial_ur1)
 #                        dt2nd_axial_ur1 = (2*ur1[m,n] - 5*ur1[m+1,n] + 4*ur1[m+2,n] -ur1[m+3,n])/(dx**3)  # FWD

    elif m == Nx:
    # --------------------------- dt2nd axial ux1 ---------------------------------#

        dt2nd_axial_ux1 = (ux1[m-2, n] - 2*ux1[m-1, n] + ux1[m, n])/(dx**2)  # BWD
    # dt2nd_axial_ux1 = (2*ux1[m,n] - 5*ux1[m-1,n] + 4*ux1[m-2,n] -ux1[m-3,n])/(dx**3) # BWD
                     # --------------------------- dt2nd axial ur1 ---------------------------------#
    # Three-point BWD
        dt2nd_axial_ur1 = (ur1[m-2, n] - 2*ur1[m-1, n] + ur1[m, n])/(dx**2)
        print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

    else:
    # --------------------------- dt2nd axial ux1 ---------------------------------#
        dt2nd_axial_ux1 = (ux1[m+1, n] + ux1[m-1, n] - 2*ux1[m, n])/(dx**2)  # CD

    # --------------------------- dt2nd axial ur1 ---------------------------------#
        dt2nd_axial_ur1 = (ur1[m+1, n] + ur1[m-1, n] - 2*ur1[m, n])/(dx**2)  # CD
        print("dt2nd_axial_ur1:", dt2nd_axial_ur1)
    
    return dt2nd_axial_ux1, dt2nd_axial_ur1


def gradients_ux2(p_in, p1, ux_in, ux1, dx, dr, m, n):
    if m == 0 and n != 1:
        # 4-point CD
        dp_dx = (p_in - 8*p_in + 8 *
                 p1[m+1, n] - p1[m+2, n])/(12*dx)
        ux_dx = (ux1[m, n] - ux_in)/dx
        ux_dr = (ux1[m, n+1] - ux1[m, n])/dr

    elif m == 0 and n == 1:
        # 4-point CD
        dp_dx = (p_in - 8*p_in + 8 *
                 p1[m+1, n] - p1[m+2, n])/(12*dx)
        ux_dx = (ux1[m, n] - ux_in)/dx

        # NOTE: SYMMETRY CONDITION HERE done
        ux_dr = (ux1[m, n+2] - ux1[m, n])/(4*dr)

    elif m == Nx and n != 1:
        dp_dx = (p1[m, n] - p1[m-1, n])/dx
        ux_dx = (ux1[m, n] - ux1[m-1, n])/dx
        ux_dr = (ux1[m, n+1] - ux1[m, n])/dr

    elif m == Nx and n == 1:
        dp_dx = (p1[m, n] - p1[m-1, n])/dx
        ux_dx = (ux1[m, n] - ux1[m-1, n])/dx

        # NOTE: SYMMETRY CONDITION HERE done
        ux_dr = (ux1[m, n+2] - ux1[m, n])/(4*dr)

    else:
        dp_dx = (p1[m+1, n] - p1[m, n])/dx
        ux_dx = (ux1[m+1, n] - ux1[m, n])/dx
        ux_dr = (ux1[m, n+1] - ux1[m, n])/dr

    return dp_dx, ux_dx, ux_dr


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


def f_ts(ps):
    #   Calculate saturated vapor temperature (K)
    print("Ps for f_ts calc: ", ps)
    ps1 = np.log(ps/100000.0)
    t_sat = 74.87701+6.47033*ps1+0.45695*ps1**2+0.02276*ps1**3+7.72942E-4*ps1**4+1.77899E-5 * \
        ps1**5+2.72918E-7*ps1**6+2.67042E-9*ps1**7+1.50555E-11*ps1**8+3.71554E-14*ps1**9
    return t_sat


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


def v_m(tg):
    #   Calculate arithmetic mean speed of gas molecules (m/s)
    print("Tg for v_m gas: ", tg)
    v_mean = np.sqrt(8.*R*tg/np.pi/M_n)
    # ipdb.set_trace()
    return v_mean


def c_c(ts):
    #   Calculate the heat capacity of copper (J/(kg*K))
    print("Ts for c_c (specific heat copper) calc: ", ts)
    #  print("ts",ts)
    c_copper = 1.22717-10.74168*np.log10(ts)**1+15.07169*np.log10(
        ts)**2-6.69438*np.log10(ts)**3+1.00666*np.log10(ts)**4-0.00673*np.log10(ts)**5
    c_copper = 10.**c_copper
    return c_copper


def k_cu(T):
    #   Calculate the coefficient of thermal conductivity of copper (RRR=10) (W/(m*K)) (for pde governing copper wall, heat conducted in the x axis.)
    print("Tw for k_cu copper: ", T)
    k1 = 3.00849+11.34338*T+1.20937*T**2-0.044*T**3+3.81667E-4 * \
        T**4+2.98945E-6*T**5-6.47042E-8*T**6+2.80913E-10*T**7
    k2 = 1217.49161-13.76657*T-0.01295*T**2+0.00188*T**3-1.77578E-5 * \
        T**4+7.58474E-8*T**5-1.58409E-10*T**6+1.31219E-13*T**7
    k3 = k2+(k1-k2)/(1+np.exp((T-70)/1))
    return k3


def D_nn(T_g, P_g):
    #   Calculate self mass diffusivity of nitrogen (m^2/s)
    if T_g > 63:
        D_n_1atm = -0.01675+4.51061e-5*T_g**1.5
    else:
        D_n_1atm = (-0.01675+4.51061e-5*63**1.5)/63**1.5*T_g**1.5
    D_n_p = D_n_1atm*101325/P_g
    D_n_p = D_n_p/1e4
    return D_n_p


def mu_n(T, P):
    #   Calculate viscosity of nitrogen (Pa*s)
    print("viscosity temp and pressure", T, P)
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
    print("viscosity from function:", (mu_n_1+mu_n_2)/1e6)

    return (mu_n_1+mu_n_2)/1e6


def gamma(a):
    #   Calculate the correction factor of mass flux
    gam1 = np.exp(-np.power(a, 2.))+a*np.sqrt(np.pi)*(1+math.erf(a))
    return gam1


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


def val_in(n):
    #   Calculate instant flow rate (kg/s)
    # Fitting results
    #    A1 = 0.00277; C = 49995.15263  # 50 kPa
    A1 = 0.00261
    C = 100902.5175  # 100 kPa
   # A1 = 0.00277; C = 10000.15263  # 50 kPa
    P_in_fit = np.power(A1*n*dt+np.power(C, -1./7.), -7.)

    # P_in_fit = 1./2.*P_in_fit
    # print("pressure is halved P_in_fit", P_in_fit)

    dP_in_fit = -7.*A1*np.power(A1*n*dt+np.power(C, -1./7.), -8.)

    q_in = -(np.power(C, 2./7.)*0.230735318/1.4/297./T_in) * \
        (np.power(P_in_fit, -2./7.)*dP_in_fit)
    ma_in_x = q_in/A
    rho_in = ma_in_x/u_in_x
    p_in = rho_in/M_n*R*T_in
    ux_in = ma_in_x/rho_in
#    ux_in = 10
    ur_in = 0.
    e_in = 5./2.*rho_in/M_n*R*T_in + 1./2.*rho_in*ux_in**2
    # print("u_in_x", u_in_x)
    out = np.array([q_in, ux_in, ur_in, rho_in, p_in, e_in])
    print(
        "val_in from function [q_in, ux_in, ur_in, rho_in, p_in, e_in]: ", out)
    return out


def DN(T, P, u, T_w):
    #   Calculate dimensionless numbers
    rho = P*M_n/R/T
    # rho_w=f_ps(T_w)*M_n/R/T #_w
    mu = mu_n(T, P)
    # print("mu", mu)
    Re = rho*(u)*D/mu  # Reynolds number
    D_n = D_nn(T, P)
    Sc = mu/rho/D_n  # Schmidt number
    Kn = 2*mu/P/np.sqrt(8*M_n/np.pi/R/T)/D
    mu_w = mu_n(T_w, f_ps(T_w))
    Sh = 0.027*Re**0.8*Sc**(1/3)*(mu/mu_w)**0.14  # Sherwood number
    Nu = 0.027*Re**0.8*Pr_n**(1/3)*(mu/mu_w)**0.14  # Nusselt number
    Cou = u*dt/dx  # Courant Number
    DN_all = np.array([Re, Sc, Kn, Sh, Nu, Cou])
    print(DN_all)
    # print("Courant Number is: ", Cou)
    return DN_all

# de1[m] = m_de(T1[m, n], p1[m, n], Tw1[m], de1[m], rho1[m, n]*ur1[m, n]-rho1[m, n-1]*ur1[m, n-1])

# NOTE: I am getting wrong mass deposition values... from 1d it is in the order of e-6


def m_de(T, P, T_s, de, dm):
    print("mdot calc: ", "Tg: ", T, " P: ",
          P, "Ts: ", T_s, "de: ", de, "dm: ", dm)
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
    print("Saturation pressure at this Ts", P_s)

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
        print("m_max_sound:", m_max, "rho", rho, "rho_min", rho_min)
#        m_max = ((rho-rho_min)/dt - 1/(Nr*dr**2)*dm) * \
 #           D/4.         # From continuity equation
#        m_max = 2.564744575054553e-26  #NOTE: added to limit condensation rate...
#        print("m_max_sound:",m_max)
        print("saturation temp: ", f_ts(P*np.sqrt(T_s/T)))
        print("mout calculated: ", m_out)
        if m_out > m_max:
            m_out = m_max
            print("mout = mmax")
    else:
        print("P<P0")
        m_out = 0
#    m_out = 0
    rho_min = p_0*M_n/R/T
#    m_max = ((rho-rho_min)/dt - 1/(Nr*dr**2)*dm) * \
#       D/4.         # From continuity equation
    # print("m_max_sound:", m_max, "rho", rho, "rho_min", rho_min)

    print("mout final: ", m_out)
    m_out = 0  # NO HEAT TRANSFER/ MASS DEPOSITION CASE
    return m_out  # Output: mass deposition flux, no convective heat flux


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


def save_initial_conditions(rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1, de1):
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
    np.savetxt("de.csv", de0, delimiter=",")
    np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")


def save_data(tx, dt, rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1, de1):
    increment = (tx+1)*dt
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(increment) + '/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "timestepping"
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
    np.savetxt("tw.csv", Tw1, delimiter=",")
    np.savetxt("ts.csv", Ts1, delimiter=",")
    np.savetxt("de_mass.csv", de0, delimiter=",")
    np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")


def namestr(obj, namespace):  # Returns variable name for check_negative function
    return [name for name in namespace if namespace[name] is obj]


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


def check_array(array_in):
    if np.any(array_in < 0):
        array_name = namestr(array_in, globals())[0]
        print(array_name + " has at least one negative value.")
        exit()


def delete_r0_point(rho2, ux2, ur2, u2, e2, T2, p1):
    rho3 = np.delete(rho2, 0, axis=1)
    ux3 = np.delete(ux2, 0, axis=1)
    ur3 = np.delete(ur2, 0, axis=1)
    u3 = np.delete(u2, 0, axis=1)
    e3 = np.delete(e2, 0, axis=1)
    T3 = np.delete(T2, 0, axis=1)
    p3 = np.delete(p1, 0, axis=1)
    return [rho3, ux3, ur3, u3, e3, T3, p3]


if __name__ == '__main__':
    t_sat = 70
    p_test = f_ps(t_sat)
    print("p_test", p_test)

    p_sat = 10000
    t_test = f_ts(p_sat)
    print("t_test", t_test)

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
    print("delta_h ", delta_h(tg, ts))

# ------------------------- specific heat of solid nitrogen (J/(kg*K)) ------------------------------- #

    print("c_n ", c_n(ts))
# ------------------------- thermal velocity ------------------------------- #

    print("vm ", v_m(tg))

# ------------------------- heat capacity of copper (J/(kg*K)) ------------------------------- #

    print("c_c ", c_c(ts))

# ------------------------- coefficient of thermal conductivity of copper (RRR=10) (W/(m*K)) ------------------------------- #

    T = 4.2
    print("k_cu", k_cu(T))

# ------------------------- self mass diffusivity of nitrogen (m^2/s) ------------------------------- #

    T_g = 298
    P_g = 1000
    print("D_nn", D_nn(T_g, P_g))

# ------------------------- Viscosity ------------------------------- #

    T = 273.15
    P = 9806649
    print("mu_n ", mu_n(T, P))

# ------------------------- Error function ------------------------------- #

#    a = umean/vm1
    a = 0.5
    print("gamma ", gamma(a))


# ------------------------- Inlet values ------------------------------- #

    print("val_in ", val_in(5))

# ------------------------- Dimensionless numbers ------------------------------- #

    T = 30
    P = 3000
    u = 100
    T_w = 15
    print("DN ", DN(T, P, u, T_w))

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
    print("m_de", m_de(T, P, T_w, de, dm))


# ------------------------- Heat transferred ------------------------------- #

    tw = 298
    BW_coe = 0.02
    print("q_h ", q_h(tw, BW_coe))

# ------------------------- Exponential Smoothing ------------------------------- #
    p_in = 8000
    p_0 = 2000
    n_trans = 70
    Nx = 200
    L = 6.45
    p1 = np.full((Nx+1), p_0, dtype=np.float64)  # Pressure
    for i in np.arange(0, Nx+1):
        p1[i] = exp_smooth(i+n_trans, p_in*2.-p_0, p_0, 0.3, n_trans)
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
    x = np.linspace(0, L, Nx+1)
    plt.plot(x, p1)
    plt.show()
