from inspect import currentframe
from functions import *
from my_constants import *


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


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


#### -----------------------------------------   Calculate initial values ----------------------------------------- #
# Internal energy - defined in constants file

# rho_0 = 1e-2  # An arbitrary small initial density in pipe, kg/m3
# p_0 = rho_0/M_n*R*T_0  # Initial pressure, Pa
# e_0 = 5./2.*rho_0/M_n*R*T_0  # Initial internal energy

# Kinetic energy

u_in_x = np.sqrt(7./5.*R*T_in/M_n)*1.0  # Inlet velocity, m/s (gamma*RT)
u_in_r = 0

# Stability factors
F = 1.*dt/dx**2.  # Stability indictor   ### Q:
artv = 0.06  # Control parameter for the artificial viscosity


# ----------------- Array initialization ----------------------------
# rho12 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
p1 = np.full((Nx+1, Nr+1), p_0, dtype=(np.float64, np.float64))  # Pressure
rho1 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  # velocity -x
ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  # velocity -r
u1 = np.sqrt(np.square(ux1) + np.square(ur1))  # total velocity
# Internal energy
e1 = np.full((Nx+1, Nr+1), e_0, dtype=(np.float64, np.float64))
# CHECK TODO: calculate using equation velocity.
# TODO: calculate using equation velocity.

T1 = np.full((Nx+1, Nr+1), T_0, dtype=(np.float64, np.float64))  # Temperature

rho2 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))
ux2 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
ur2 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
u2 = np.sqrt(np.square(ux2) + np.square(ur2))  # total velocity
e2 = np.full((Nx+1, Nr+1), e_0, dtype=(np.float64, np.float64))
T2 = np.full((Nx+1, Nr+1), T_0, dtype=(np.float64, np.float64))
p2 = np.full((Nx+1, Nr+1), p_0, dtype=(np.float64, np.float64))  # Pressure


Tw1 = np.full((Nx+1), T_s, dtype=(np.float64))  # Wall temperature
Tw2 = np.full((Nx+1), T_s, dtype=(np.float64))
Ts1 = np.full((Nx+1), T_s, dtype=(np.float64))  # Temperature of SN2 surface
Ts2 = np.full((Nx+1), T_s, dtype=(np.float64))

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

# Initialization

# ps = np.zeros(Nx+1, np.float64)
# mdot = np.zeros(Nx+1, np.float64)
# for x in np.arange(np.int64(0), np.int64(Nx+1)):
#     dm = rho1[x, Nr]*Nr*dr*ur1[x, Nr]-rho1[x, Nr-1]*(Nr-1)*dr*ur1[x, Nr-1]
#     ps[x] = f_ps(Ts1[x])
#     mdot[x] = m_de(T1[x,Nr],p1[x],Tw1[x], de1[x], dm)
# np.savetxt("ps.csv", ps, delimiter=",")
# np.savetxt("m_de.csv", mdot, delimiter=",")

# np.savetxt("ps.csv", ps, delimiter=",")
# np.savetxt("mdot.csv", mdot, delimiter=",")


# ---------------------  Smoothing inlet --------------------------------

# Set Initial Conditions:

q_in, ux_in, ur_in, rho_in, p_in, e_in = val_in(0)  # define inlet values
print("Initial conditions simulation start", val_in(0))

# ux_in = 10

### ------------------------------------- PREPPING AREA - smoothing ------------------------------------------------- ########

for i in range(0, Nx+1):
    p1[i, :] = exp_smooth(i+n_trans, p_in*2.-p_0, p_0, 0.4, n_trans)
   # print("P1 smoothing values", p1[i,:])
    rho1[i, :] = exp_smooth(i + n_trans, rho_in*2, rho_0, 0.4, n_trans)
#    T1[i, :] = T_neck(i)
    # if i<51: T1[i]=T_in
    T1[i, :] = p1[i, :]/rho1[i, :]/R*M_n
    # v_max = np.sqrt(7./5.*R*T/M_n)  # diatomic gas gamma = 7/5
#    u1[i, :] = exp_smooth(i + n_trans, ux_in*2, 0, 0.4, n_trans)

    # if i < n_trans+1:
    #     e1[i, :] = 5./2.*p1[i, :]+1./2.*rho1[i, :]*u1[i, :]**2

#        rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]  # IDEAL GAS LAW

    # print("p1 matrix after smoothing", p1)
    # else:
    #     e1[i, :] = 5/2*rho1[i, :]/M_n*R*T_in+1/2**rho1[i, :]*u1[i, :]**2


# for i in range(0, Nx+1):


####### ---------------------------- PARABOLIC VELOCITY PROFILE - inlet prepping area-------------------------------------------------------- ######

for i in np.arange(60):
    # diatomic gas gamma = 7/5   WE USED ANY POINT, since this preparation area is constant along R direction.
    v_max = np.sqrt(7./5.*R*T1[i, 4]/M_n)
    for y in np.arange(Nr+1):
        a = v_max*(1.0 - ((y*dr)/R_cyl)**2)
        ux1[i, y] = a
        u1[i, y] = ux1[i, y]


### ---------------------------------------------------------- NO SLIP BOUNDARY CONDITION ----------------------------------------------------------###
ux1[:, Nr] = 0
u1[:, Nr] = 0

e1[:, Nr] = 5./2. * p1[:, Nr]
# recalculate energies

## ------------------------------------------------------------- SAVING INITIAL CONDITIONS ---------------------------------------------------------------- #####

save_initial_conditions(rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1, de1)

##### ----------------------------------------- PLOTTING INITIAL CONDITIONS ---------------------------------------------------------------------------####

fig, axs = plt.subplots(3)
fig.suptitle('Initial Conditions along tube for all R')

# PRESSURE DISTRIBUTION
im = axs[0].imshow(p1.transpose())
plt.colorbar(im, ax=axs[0])
# plt.colorbar(im, ax=ax[0])
axs[0].set(ylabel='Pressure [Pa]')
plt.title("Pressure smoothing")


# VELOCITY DISTRIBUTION
# axs[1].imshow()
im = axs[1].imshow(ux1.transpose())
plt.colorbar(im, ax=axs[1])
# axs[1].colorbars(location="bottom")
axs[1].set(ylabel='Ux [m/s]')
plt.title("velocity parabolic smoothing")

# Temperature DISTRIBUTION
im = axs[2].imshow(T1.transpose())
plt.colorbar(im, ax=axs[2])
# axs[1].colorbars(location="bottom")
axs[2].set(ylabel='temperature [K]')


plt.xlabel("L(x)")
plt.show()


## ------------------------------------------------ BC INLET starting matrices  ------------------------------------------------- #

# NOTE: BC INIT
Ts1[:] = T1[:, Nr]
Ts2[:] = Ts1
Tw1[:] = Ts1
Tw2[:] = Ts1
Tc1[:] = Ts1
Tc2[:] = Ts1

#        ux_in = 50
# e1[0,:] = e_in                #set energy BC
# rho1[0, :] = rho_in
# ux1[0, :] = ux_in
# ur1[0, :] = ur_in
# u1[0, :] = ux_in
#        print("u1 matrix before starting timestep",u1)
# p1[0,:] = p_in
# print(out)
#        e1[0, :] = 5/2*p_in + 1/2
# T1[0, :] = 298.

# NOTE: This means at m=0 no mass deposition and no helium...We dont want the surface to freeze.
# Tw1[0] = 298.
# Tw2[0] = 298.
# Ts1[0] = 298.
# Ts2[0] = 298.
# print("Tw init:", Tw1)


def main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2, ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3):

    for i in np.arange(np.int64(0), np.int64(4)):  # NOTE: Nt+1, starts with first number

        # REA += 1
        # print("REA", REA)
        q_in, ux_in, ur_in, rho_in, p_in, e_in = val_in(
            i)  # define inlet values
        print("pressure value inlet", p_in)

#        rho1[1, :] = rho_in
 #       ux1[1, :] = ux_in

  #      u1[1, :] = ux_in
   #     e1[1, :] = 5/2*p_in

        # for x in np.arange(Nx):
        #     for y in np.arange(Nr):

        #         if x == 0:
        #             rho12[x, y] = (rho1[x, y] + rho_in)/2.
        #         elif x == Nx:
        #             rho12[x, y] = (rho1[x, y] + rho1[x-1, y])/2.
        #         else:
        #             rho12[x, y] = (rho1[x, y] + rho1[x+1, y])/2.

        # starts from np start [0,Nx]
        for m in np.arange(np.int64(0), np.int64(Nx+1)):
            for n in np.arange(np.int64(1), np.int64(Nr+1)):
                print("[i,m,n]:", [i, m, n])
                # Internal energy (multiplied by rho) NOTE: check later

                ############## Case 1: At boundaries (with mass deposition).##########################################################
                # print("looping" ,"m",m, "n", n)
                if n == Nr:  # (at cylinder wall) #check Nr or Nr+1
                    print("THIS IS A SURFACE, MASS DEPOSITION:")

                    # intenal energy current timestep
                    eps = 5./2.*p1[m, n]
                    e1[m, n] = eps + 1./2. * rho1[m, n] * ur1[m, n]**2
                    print("rho1 ", rho1[m, n])
                    check_negative(rho1[m, n], n)

                    # print("p1p", p1p, "e1[m,n]", e1[m, n], "rho1[m,n]", rho1[m, n], "u1[m,n]", u1[m, n])
                    # print("printed",T1[m, n], p1p, Tw1[m], de1[m], rho1[m, n]*ur1[m, n]-rho1[m, n-1]*ur1[m, n-1])

    # Only consider mass deposition at a large enough density, otherwise the program will arise negative density
                    if rho1[m, n] > 2.*rho_0:
                        # print("printed",T1[m, n], p1p, Tw1[m], de1[m], rho1[m, n]* ur1[m, n]-rho1[m, n-1]*ur1[m, n-1])
                        # mass deposition rate (m_out)
                        # print("temp gas",T1[m,n], "pressure", p1_before_dep, "temp wall: ", Tw1[m],"mass depo", de1[m], "dm", rho1[m, n]*n*dr*ur1[m, n]-rho1[m, n-1]*n*dr*ur1[m, n-1], "n grid point", n)
                        print("inputs m_de calc: [T1, p1, Ts1, de1, rho1, ur1]",
                              T1[m, n], p1[m, n], Ts1[m], de1[m], rho1[m, n], ur1[m, n])
                        de1[m] = m_de(T1[m, n], p1[m, n], Ts1[m], de1[m], rho1[m, n]
                                      * n*dr*ur1[m, n]-rho1[m, n-1]*(n-1)*dr*ur1[m, n-1])  # used BWD
                        print("m_de / de1 calculated:", de1[m])
                        check_negative(de1[m], n)

                    # print("mass calculated de1[m]", de1[m], "m", m, "n", n)
                    # No convective heat flux. q2
                    else:
                        de1[m] = 0
                        print("NO MASS DEPOSITION: ")

                    # Integrate deposition mass
                    de0[m] += dt*np.pi*D*de1[m]
                    print("deposition mass surface", de0[m])
                    check_negative(de0[m], n)

                # Calculate the SN2 layer thickness
                    del_SN = de0[m]/np.pi/D/rho_sn
                    print("del_SN: ", del_SN)
                    check_negative(del_SN, n)

                    # density calculation
                    rho2[m, n] = rho1[m, n] - dt/(n*dr*dr)*(rho1[m, n]*(n)*dr*ur1[m, n] - rho1[m, n-1]*(
                        n-1)*dr*ur1[m, n-1]) - 4*dt/D * de1[m]
                    # rho2[m, n] = rho1[m, n] - dt/(n*dr*dr)*(rho1[m, n]*(n)*dr*ur1[m, n] - rho1[m, n-1]*(
                    #     n-1)*dr*ur1[m, n-1]) - dt/dx*(dm) - 4*dt/D * de1[m]   ## New assumption
                    print("rho2 surface", rho2[m, n])
                    check_negative(rho2[m, n], n)
                    #     print("dr term:", -dt/(n*dr*dr)*(rho1[m, n]*(n)*dr*ur1[m, n] - rho1[m, n-1]*(
                    #         n-1)*dr*ur1[m, n-1]), "rho1[m,n]:", rho1[m, n], "rho1[m,n-1]:", rho1[m, n-1])
                    #     print("rho1", rho1[m, n])

                    # ensure no division by zero
                    if rho2[m, n] == 0:
                        rho2[m, n] = 0.0001
                        print("Density went to zero")

# NOTE: Check with Yolanda, should i use rho1 or rho2 ??? Also de1 or de2?

                    # velocity calculation #  I think we need some momentum R equation... this is not correct.
                    ur2[m, n] = de1[m]/rho2[m, n]
                    u2[m, n] = ur2[m, n]  # no slip boundary condition.
                    ux1[m, n] = 0.
                    ux2[m, n] = 0.
                    print("ur2 surface", ur2[m, n], "u2 surface", u2[m, n])
                    check_negative(ur2[m, n], n)
                    check_negative(u2[m, n], n)

                    # energy calculation
                    # radial kinetic enery on surface.
                    delta_e = n*dr*ur1[m, n]*e1[m, n] - \
                        (n-1)*dr*ur1[m, n-1]*e1[m, n-1]  # BWD
                    e2[m, n] = e1[m, n]-dt / \
                        (n*dr*dr)*(delta_e) - dt*4/D*de1[m]*(e1[m, n])
                    print("e1 surface", e1[m, n], "e2 surface", e2[m, n])
                    check_negative(e1[m, n], n)
                    check_negative(e2[m, n], n)

                # Calculate the gas temperature and ensure it's higher than the SN2 surface temperature

                    T2[m, n] = 2./5.*(e2[m, n]-1./2.*rho2[m, n] *
                                      ur2[m, n]**2.)*M_n/rho2[m, n]/R
                    print("T2 surface", T2[m, n])
                    check_negative(T2[m, n], n)

                    if T2[m, n] < Ts1[m]:
                        e2[m, n] = 5./2.*rho2[m, n]*R*Ts1[m] / \
                            M_n
                        print("THIS IS T2 < Ts")
                        print("e2 surface", e2[m, n])
                        check_negative(e2[m, n], n)

                        T2[m, n] = 2./5.*(e2[m, n] - 1./2.*rho2[m, n]
                                          * ur2[m, n]**2.)*M_n/rho2[m, n]/R
                        print(
                            "T2 surface recalculated to make it equal to wall temperature (BC)", T2[m, n])
                        check_negative(T2[m, n], n)

                    # NOTE: Check if the dt2nd is correct.
#                    second_derivative = laplace(Tw1)/dx
                    if m == 0:
                        dt2nd = (T_in - 2 * Tw1[m] +
                                 Tw1[m+1])/(dx**2)  # 3-point CD
#                        dt2nd = Tw1[m+1]-Tw1[m]-Tw1[m-1]+T_in
                    elif m == Nx:
                        print("m=Nx", m)
                        dt2nd = (-Tw1[m-3] + 4*Tw1[m-2] -
                                 5*Tw1[m-1] + 2*Tw1[m]) / (dx**2)  # Four point BWD
                    else:
                        dt2nd = Tw1[m-1]-2*Tw1[m]+Tw1[m+1]/(dx**2)

                    if math.isnan(dt2nd):
                        print("NAN dt2nd surface")
                        assert not math.isnan(dt2nd)

                # Radial heat transfer within Copper section

# Only consider the thermal resistance through SN2 layer when thickness is larger than a small preset value (taking average value)

# NOTE: CHECK THIS LOGIC TREE

                        # q deposited into frost layer. Nusselt convection neglected
                    q_dep = de1[m]*(1/2*(ur1[m, n])**2 +
                                    delta_h(T1[m, n], Ts1[m]))

                    if del_SN > 1e-5:
                        print(
                            "This is del_SN > 1e-5 condition, conduction across SN2 layer considered")

                        # heatflux into copper wall from frost layer
                        qi = k_sn*(Ts1[m]-Tw1[m])/del_SN
                        print("qi: ", qi)
                        check_negative(qi, n)

                       # pipe wall equation
                        Tw2[m] = Tw1[m] + dt/(w_coe*c_c(Tw1[m]))*(
                            qi-q_h(Tw1[m], BW_coe)*Do/D)+dt/(rho_cu*c_c(Tw1[m]))*k_cu(Tw1[m])*dt2nd
                        print("Tw2: ", Tw2[m])
                        check_negative(Tw2[m], n)

                        # SN2 Center layer Tc equation
                        Tc2[m] = Tc1[m] + dt * \
                            (q_dep-qi) / (rho_sn * c_n(Ts1[m, n]*del_SN))
                        print("Tc2: ", Tc2[m, n])
                        check_negative(Tc2[m], n)

                    else:
                        # heatflux into copper wall from frost layer
                        qi = 0
                        print("qi: ", qi)
                        check_negative(qi, n)

                       # pipe wall equation
                        Tw2[m] = Tw1[m] + dt/(w_coe*c_c(Tw1[m]))*(
                            qi-q_h(Tw1[m], BW_coe)*Do/D)+dt/(rho_cu*c_c(Tw1[m]))*k_cu(Tw1[m])*dt2nd
                        print("Tw2: ", Tw2[m])
                        check_negative(Tw2[m], n)

                        # SN2 Center layer Tc equation
                        # NOTE: Is this te wall temperature?
                        Tc2[m] = Tw2[m]
                        print("Tc2: ", Tc2[m])

                    # Calculate SN2 surface temp
                    Ts2[m] = 2*Tc2[m] - Tw2[m]
                    print("Ts2: ", Ts2[m])
                    check_negative(Ts2[m], n)

                    # Heat transfer rate helium
                    qhe[m] = q_h(Tw1[m], BW_coe)*np.pi*Do

                    # print("line 759", "Ts1", Ts1[m], "Ts2", Ts2[m], "Tc2", Tc2[m], "c_c(Ts1[m])", c_c(Ts1[m]), "qh", q_h(Ts1[m], BW_coe), "k_cu(Ts1[m])", k_cu(Ts1[m]), "dt2nd", dt2nd)
                    print("qhe: ", qhe[m])
                    check_negative(qhe[m], n)

                    # Update pressure
                    p2[m, n] = 2./5.*(e2[m, n] - 1./2.*rho2[m, n]*ur2[m, n]**2)
                    check_negative(p2[m, n], n)
#                    p2[m, n] = rho2[m, n] * R * T2[m, n]/M_n


################################################################### Case 2: no mass deposition (within flow field,away from wall in radial direction) ########

                else:
                    print("THIS IS THE BULK:")
                    print("rho1 bulk: ",
                          rho1[m, n], "T1 bulk:", T1[m, n])
                    eps = 5./2.*p1[m, n]
                    print("eps bulk:", eps)
                    if eps < 0:
                        print("negative eps Bulk ", eps)
                        exit()
                    if math.isnan(eps):
                        print("NAN EPS Bulk ", eps)
                        assert not math.isnan(eps)

#                    print("u1 for p1 calculation:", u1[m, n])

                    # Calculate mass\momentum\energy at time n+1 with no mass deposition - no heat transfer

# -------------------------------------------- Different Boundary Solutions ------------------------------------------------------#

                    # Find density at next timestep.
                    # Define second derivative ux axial direction. (consider m as reference)

                    # NOTE:A add dt2nd limiter...
                    # if dt2nd_axial_ux1 > 10000:
                    #     dt2nd_axial_ux1 = 1000

                    if m == 0:
                        rho2[m, n] = rho_in - dt/(n*dr*dr)*(rho1[m, n+1]*(n+1)*dr*ur1[m, n+1] - rho1[m, n]
                                                            * n*dr*ur1[m, n]) - dt/dx*(rho1[m, n]*ux1[m, n]-rho_in*ux_in)
                        #                        dt2nd_axial_ux1 = (2*ux1[m,n] - 5*ux1[m+1,n] + 4*ux1[m+2,n] -ux1[m+3,n])/(dx**3) #FWD

                        # --------------------------- dt2nd axial ux1 ---------------------------------#
                        dt2nd_axial_ux1 = (
                            ux_in - 2*ux1[m, n] + ux1[m+1, n]) / (dx**2)
                        # dt2nd_axial_ux1 = (ux1[m+2,n] -2*ux1[m+1,n] + ux1[m,n])/(dx**2) #FWD

                    # --------------------------- dt2nd axial ur1 ---------------------------------#
                        #                        dt2nd_axial_ur1 = (ur1[m+2,n] -2*ur1[m+1,n] + ur1[m,n])/(dx**2) #FWD
                        # FWD
                        dt2nd_axial_ur1 = (-ur_in + ur_in - 30 *
                                           ur1[m, n] + 16*ur1[m+1, n] - ur1[m+2, n])/(12*dx**2)
                        print("dt2nd_axial_ur1:", dt2nd_axial_ur1)
 #                        dt2nd_axial_ur1 = (2*ur1[m,n] - 5*ur1[m+1,n] + 4*ur1[m+2,n] -ur1[m+3,n])/(dx**3)  # FWD

                    elif m == Nx:
                        rho2[m, n] = rho1[m, n] - dt/(n*dr*dr)*(rho1[m, n+1]*(n+1)*dr*ur1[m, n+1] - rho1[m, n]
                                                                * n*dr*ur1[m, n]) - dt/dx*(rho1[m, n]*ux1[m, n]-rho1[m-1, n]*ux1[m-1, n])
                        # --------------------------- dt2nd axial ux1 ---------------------------------#

                        dt2nd_axial_ux1 = (
                            ux1[m-2, n] - 2*ux1[m-1, n] + ux1[m, n])/(dx**2)  # BWD
#                        dt2nd_axial_ux1 = (2*ux1[m,n] - 5*ux1[m-1,n] + 4*ux1[m-2,n] -ux1[m-3,n])/(dx**3) # BWD
                     # --------------------------- dt2nd axial ur1 ---------------------------------#
                        # Three-point BWD
                        dt2nd_axial_ur1 = (
                            ur1[m-2, n] - 2*ur1[m-1, n] + ur1[m, n])/(dx**2)
                        print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

                    else:
                        rho2[m, n] = rho1[m, n] - dt/(n*dr*dr)*(rho1[m, n+1]*(n+1)*dr*ur1[m, n+1] - rho1[m, n]
                                                                * n*dr*ur1[m, n]) - dt/dx*(rho1[m+1, n]*ux1[m+1, n]-rho1[m, n]*ux1[m, n])
                        print("rho1 bulk", rho1[m, n],
                              "rho2 bulk:", rho2[m, n])
                    # print("density inside the bulk:", rho2[m, n])

                        # --------------------------- dt2nd axial ux1 ---------------------------------#
                        dt2nd_axial_ux1 = (
                            ux1[m+1, n] + ux1[m-1, n] - 2*ux1[m, n])/(dx**2)  # CD
                        # --------------------------- dt2nd axial ur1 ---------------------------------#
                        dt2nd_axial_ur1 = (
                            ur1[m+1, n] + ur1[m-1, n] - 2*ur1[m, n])/(dx**2)  # CD
                        print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

                    print("rho2 bulk", rho2[m, n])
                    check_negative(rho2[m, n], n)

                    # Define second derivatives in radial direction (consider n as reference)

                    if n == 1:
                        # --------------------------- dt2nd radial ux1 ---------------------------------#
                        dt2nd_radial_ux1 = (
                            ux1[m, n+2] - 2*ux1[m, n+1] + ux1[m, n])/(dr**2)  # FWD
 #                       dt2nd_radial_ux1 = (2*ux1[m,n] - 5*ux1[m,n+1] + 4*ux1[m,n+2] -ux1[m,n+3])/(dr**3) # FWD

                    # --------------------------- dt2nd radial ur1 ---------------------------------#

                    # NOTE: Symmetry Boundary Condition assumed for ur1 radial derivative along x axis..

                        grad_ur1_n1 = (ur1[m, n+2] - ur1[m, n])/(4*dr)
                        dt2nd_radial_ur1_n1 = (
                            ur1[m, n+2] - ur1[m, n]) / (4*dr**2)
                        print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1_n1)
                        dt2nd_radial_ur1 = 0.

                    else:  # (n is between 1 and Nr)

                        # --------------------------- dt2nd radial ux1 ---------------------------------#
                        dt2nd_radial_ux1 = (
                            ux1[m, n+1] + ux1[m, n-1] - 2*ux1[m, n])/dr**2  # CD

                    # --------------------------- dt2nd radial ur1 ---------------------------------#
                        dt2nd_radial_ur1 = (
                            ur1[m, n+1] + ur1[m, n-1] - 2*ur1[m, n])/(dr**2)  # CD
                        print("dt2nd_radial_ur1:", dt2nd_radial_ur1)

                    if m == 0:
                        # 4-point CD
                        dp = (p_in - 8*p_in + 8 *
                              p1[m+1, n] - p1[m+2, n])/(12*dx)
                    #     print("first term:", ux1[m, n], "pressure term:", -dt*dp/rho1[m, n], "viscosity:", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ux1 + (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) + dt2nd_axial_ux1), "dt2nd_axial_ux", dt2nd_axial_ux1, "dt2nd_radial_ux",
                    #           dt2nd_radial_ux1, "ux1 term:", dt*ux1[m, n] * (ux1[m, n] - ux_in)/dx, "ur1 term:", dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr, "mu_n", mu_n(T1[m, n], p1[m, n]), "extra", (ux1[m, n+1]-ux1[m, n]), "extra2", dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr)
                    # # print("rho2[m,n] inside bulk=", rho2[m, n])
                        ux2[m, n] = ux1[m, n] - dt*dp/rho1[m, n] + \
                            mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                            (dt2nd_radial_ux1 + (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) +
                             dt2nd_axial_ux1) -\
                            dt*ux1[m, n] * (ux1[m, n] - ux_in)/dx -\
                            dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr

                    elif m == Nx:
                        # print("first term:", ux1[m, n], "pressure term:", -dt*(p1[m, n] - p1[m-1, n])/(rho1[m, n]*dx), "viscosity:", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ux1 + (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) + dt2nd_axial_ux1), "dt2nd_axial_ux", dt2nd_axial_ux1,
                        #       "dt2nd_radial_ux", dt2nd_radial_ux1, "ux1 term:", dt*ux1[m, n] * (ux1[m, n] - ux1[m-1, n])/dx, "ur1 term:", dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr, "mu_n", mu_n(T1[m, n], p1[m, n]), "extra", (ux1[m, n+1]-ux1[m, n]), "extra2", dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr)
                        # print("rho2[m,n] inside bulk=", rho2[m, n])
                        ux2[m, n] = ux1[m, n] - dt*(p1[m, n] - p1[m-1, n])/(rho1[m, n]*dx) + \
                            mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                            (dt2nd_radial_ux1 + (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) +
                             dt2nd_axial_ux1) -\
                            dt*ux1[m, n] * (ux1[m, n] - ux1[m-1, n])/dx -\
                            dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr

                    else:
                        print("first term:", ux1[m, n], "pressure term:", -dt*(p1[m+1, n] - p1[m, n])/(rho1[m, n]*dx), "viscosity:", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ux1 + (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) + dt2nd_axial_ux1), "dt2nd_axial_ux", dt2nd_axial_ux1, "dt2nd_radial_ux", dt2nd_radial_ux1, "ux1 term:", dt*ux1[m, n] * (
                            ux1[m+1, n] - ux1[m, n])/dx, "ur1 term:", dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr, "mu_n", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ux1 + (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) + dt2nd_axial_ux1), "extra", (ux1[m, n+1]-ux1[m, n]), "extra2", dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr)
                        print("rho2 bulk=", rho2[m, n])
                        ux2[m, n] = ux1[m, n] - dt*(p1[m+1, n] - p1[m, n])/(rho1[m, n]*dx) + \
                            mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                            (dt2nd_radial_ux1 + (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) +
                             dt2nd_axial_ux1) -\
                            dt*ux1[m, n] * (ux1[m+1, n] - ux1[m, n])/dx -\
                            dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr

                    # if ux2[m, n] < 1:
                    #     ux2[m, n] = 0

                    print("ux1 bulk", ux1[m, n], "ux2 bulk:", ux2[m, n])
                    # print("ux2 bulk=", ux2[m, n])
                   # print("matrix ux", ux2)
                    check_negative(ux2[m, n], n)

                    # if ux2[m, n] > 370:
                    #     print("high ux2 bulk", ux2[m, n])
                    #     exit()

                    # if ux2[m, n] > 350:
                    #     print("ux2 bulk larger than ux_in", ux2[m, n])
                    #     print("m,n",m,n,"dtux1 term:", -dt*ux1[m, n] * (ux1[m, n] - ux1[m-1, n])/dx, "dtur1 term:", dt*ur1[m, n]*(ux1[m, n+1] - ux1[m, n])/dr)
                    #     print("ux1[m,n]",ux1[m,n], ux1[m-1,n])
                    #     print("ur1[m,n]", ux1[m, n+1], ux1[m, n])
                    #     print("pressure term: ",- dt*(p1[m+1, n] - p1[m, n])/(rho1[m, n]*dx))
                    #     print("viscous term: ",mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                    #     ((ux1[m, n+1] + ux1[m, n-1] - 2*ux1[m, n])/(dr**2) +
                    #      (1/(n*dr)) * ((ux1[m, n+1]-ux1[m, n])/dr) +
                    #      (ux1[m+1, n] + ux1[m-1, n] - 2*ux1[m, n])/(dx**2)))
                    #     print("ux1[m+1,n]: ",ux1[m+1, n],"ux1[m-1,n]: ",ux1[m-1, n], "ux1[m,n]: ",ux1[m, n] )
                    #     print("second derivative", (ux1[m+1, n] + ux1[m-1, n] - 2*ux1[m, n])/(dx**2))
                    #     exit()

                    print("T1 bulk: ", T1[m, n])

                    if (m != 0 and m != Nx and n == 1):
                        #                        rho1[m,n] = (rho1[m,n] + rho1[m,n])/2.
                        grad_p1_n1 = (p1[m, n+2] - p1[m, n])/(4*dr)

                        ur2[m, n] = ur1[m, n] - dt*(grad_p1_n1)/(rho1[m, n]) +\
                            mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                            (dt2nd_radial_ur1_n1 +
                             (1/(n*dr))*grad_ur1_n1 + dt2nd_axial_ur1 -
                             - ur1[m, n]/(dr**2*n**2)) + \
                            dt*ux1[m, n] * (ur1[m+1, n] - ur1[m, n])/dx - \
                            dt*ur1[m, n]*grad_ur1_n1

                        # print("ur1", ur1[m, n],, "p_n+1, p_n", [p1[m, n+2], p1[m, n]],
                        #       "press term", dt *
                        #       (p1[m, n+1] - p1[m, n])/(rho1[m, n]*dr),
                        #       "viscous", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ur1_n1 + (
                        #           1/(n*dr))*(ur1[m, n+1]-ur1[m, n])/dr + dt2nd_axial_ur1 - ur1[m, n]/(dr**2*n**2)),
                        #       "ux1 term", dt*ux1[m, n] *
                        #       (ur1[m, n] - ur_in)/dx,
                        #       "ur1 term", dt *
                        #       ur1[m, n]*(ur1[m, n+1] - ur1[m, n])/dr
                        #       )

                    elif (m == 0 and n == 1):
                        #                        rho1[m,n] = (rho1[m,n] + rho1[m,n])/2.

                        grad_p1_n1 = (p1[m, n+2] - p1[m, n])/(4*dr)
                        ur2[m, n] = ur1[m, n] - dt*(grad_p1_n1)/(rho1[m, n]) +\
                            mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                            (dt2nd_radial_ur1_n1 +
                             (1/(n*dr))*grad_ur1_n1 + dt2nd_axial_ur1 -
                             - ur1[m, n]/(dr**2*n**2)) + \
                            dt*ux1[m, n] * (ur1[m, n] - ur_in)/dx - \
                            dt*ur1[m, n]*grad_ur1_n1

                        print("ur1", ur1[m, n], "p_n+1, p_n", [p1[m, n+1], p1[m, n]],
                              "press term", dt *
                              (p1[m, n+1] - p1[m, n])/(rho1[m, n]*dr),
                              "viscous", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ur1_n1 + (
                                  1/(n*dr))*grad_ur1_n1 + dt2nd_axial_ur1 - ur1[m, n]/(dr**2*n**2)),
                              "ux1 term", dt*ux1[m, n] *
                              (ur1[m, n] - ur_in)/dx,
                              "ur1 term", dt *
                              ur1[m, n]*(ur1[m, n+1] - ur1[m, n])/dr
                              )
                    elif (m == 0 and n != 1):
                        #                        rho1[m,n] = (rho1[m,n] + rho1[m,n])/2.
                        ur2[m, n] = ur1[m, n] - dt*(p1[m, n+1] - p1[m, n])/(rho1[m, n]*dr) +\
                            mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                            (dt2nd_radial_ur1 +
                             (1/(n*dr))*(ur1[m, n+1]-ur1[m, n])/dr + dt2nd_axial_ur1 -
                             - ur1[m, n]/(dr**2*n**2)) + \
                            dt*ux1[m, n] * (ur1[m, n] - ur_in)/dx - \
                            dt*ur1[m, n]*(ur1[m, n+1] - ur1[m, n])/dr

                        # print("ur1", ur1[m, n],, "p_n+1, p_n", [p1[m, n+1], p1[m, n]],
                        #       "press term", dt *
                        #       (p1[m, n+1] - p1[m, n])/(rho1[m, n]*dr),
                        #       "viscous", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ur1 + (
                        #           1/(n*dr))*(ur1[m, n+1]-ur1[m, n])/dr + dt2nd_axial_ur1 - ur1[m, n]/(dr**2*n**2)),
                        #       "ux1 term", dt*ux1[m, n] *
                        #       (ur1[m, n] - ur_in)/dx,
                        #       "ur1 term", dt *
                        #       ur1[m, n]*(ur1[m, n+1] - ur1[m, n])/dr
                        #       )

                    else:  # case1: (m== Nx and n==1): case2" m ==Nx, n!=1

                        ur2[m, n] = ur1[m, n] - dt*(p1[m, n+1] - p1[m, n])/(rho1[m, n]*dr) +\
                            mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * \
                            (dt2nd_radial_ur1 +
                             (1/(n*dr))*(ur1[m, n+1]-ur1[m, n])/dr + dt2nd_axial_ur1 -
                             - ur1[m, n]/(dr**2*n**2)) + \
                            dt*ux1[m, n] * (ur1[m, n] - ur1[m-1, n])/dx - \
                            dt*ur1[m, n]*(ur1[m, n+1] - ur1[m, n])/dr

                        # print("ur1", ur1[m, n], , "p_n+1, p_n", [p1[m, n+1], p1[m, n]], "press term", dt*(p1[m, n+1] - p1[m, n])/(rho1[m, n]*dr), "viscous", mu_n(T1[m, n], p1[m, n]) * dt/rho1[m, n] * (dt2nd_radial_ur1 + (1/(n*dr))*(
                        #     ur1[m, n+1]-ur1[m, n])/dr + dt2nd_axial_ur1 - ur1[m, n]/(dr**2*n**2)), "ux1 term", dt*ux1[m, n] * (ur1[m, n] - ur1[m-1, n])/dx, "ur1 term", dt*ur1[m, n]*(ur1[m, n+1] - ur1[m, n])/dr)

                    # if ur2[m, n] < 1:
                    #     ur2[m, n] = 0
                    print("ur1 bulk: ", ur1[m, n], "ur2 bulk: ", ur2[m, n])
                    print("pn+1, pn bulk: ", p1[m, n+1], p1[m, n])
                    # print("ur2 inside bulk=", ur2[m, n])
                 #   print("matrix ur", ur2)
#                    print("Value of ur2", ur2[m, n])

                    check_negative(ur2[m, n], n)

                    if rho1[m, n] == 0:
                        rho1[m, n] = 0.0001
                    u2[m, n] = np.sqrt(ux2[m, n]**2. + ur2[m, n]**2.)
                  #  print("matrix", u2)
                    print("u2 bulk: ", u2[m, n])
                    check_negative(e2[m, n], n)

            # print("u2[m,n] inside bulk=", u2[m, n])

                    e1[m, n] = eps + \
                        1./2.*rho1[m, n] * \
                        u1[m, n]**2.  # Initial internal energy
                    eps_in = 5./2.*p_in
#                    eps_in = 5./2.*rho_in/M_n*R * T_in

                    # We dont need the surface case, this is the bulk...

                    e_in_x = eps_in + 1./2.*rho_in*ux_in**2.
                    if (m == 0 and n != 1):
                        e2[m, n] = e1[m, n]-dt/(n*dr*dr)*(n*dr*ur1[m, n]*e1[m, n] - (
                            n-1)*dr*ur1[m, n-1]*e1[m, n-1]) - dt/dx*(e1[m, n]*ux1[m, n]-e_in_x*ux_in)

                    elif (m == 0 and n == 1):  # NOTE: FIX DIFFERENCING # ur =0 at  n =0
                        e2[m, n] = e1[m, n] - dt/dx * \
                            (e1[m, n]*ux1[m, n]-e_in_x*ux_in) - \
                            dt/(n*dr*dr)*(n*dr*ur1[m, n]*e1[m, n])

                    elif (m == Nx and n != 1):
                        e2[m, n] = e1[m, n]-dt/(n*dr*dr)*(n*dr*ur1[m, n]*e1[m, n] - (
                            n-1)*dr*ur1[m, n-1]*e1[m, n-1]) - dt/dx*(e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])

                    elif (m == Nx and n == 1):  # NOTE: FIX DIFFERENCING
                        e2[m, n] = e1[m, n] - dt/dx * \
                            (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n]
                             ) - dt/(n*dr*dr)*(n*dr*ur1[m, n]*e1[m, n])

                    elif (m != Nx and m != 0 and n != 1 and n != Nr):
                        e2[m, n] = e1[m, n]-dt/(n*dr*dr)*(n*dr*ur1[m, n]*e1[m, n] - (
                            n-1)*dr*ur1[m, n-1]*e1[m, n-1]) - dt/dx*(e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])

                    elif (m != Nx and m != 0 and n == 1):
                        e2[m, n] = e1[m, n] - dt/dx * \
                            (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n]
                             ) - dt/(n*dr*dr)*(n*dr*ur1[m, n]*e1[m, n])

                    # print("internal energy", e2[m, n])
                    # print("Kinetic energy", 1/2*rho2[m, n] *
                    # u2[m, n]**2)
                    print("e1 bulk: ", e1[m, n], "e2 bulk: ", e2[m, n])
                    check_negative(e1[m, n], n)
                    check_negative(e2[m, n], n)

                    # NOTE: Check temperature calculation..
                    print("temp calculation: [e2, rho2, u2]",
                          e2[m, n], rho2[m, n], u2[m, n])
                    T2[m, n] = 2/5*(e2[m, n]-1/2*rho2[m, n] *
                                    u2[m, n]**2)*M_n/rho2[m, n]/R
                    print("T2 bulk:", T2[m, n])
                    check_negative(T2[m, n], n)

                    # # calculate e2 again. #NOTE: New added.
                    # eps2 = 5./2.*rho2[m, n]/M_n*R * \
                    #     T2[m, n]
                    p2[m, n] = 2./5.*(e2[m, n] - 1./2.*rho2[m, n]*u2[m, n]**2)


############################################## Updating timesteps finished ############################################################


############################################## Boundary Conditions ############################################################


# ------------------------------------ Outlet boundary conditions ------------------------------------------- #
        print("This is the ", Nx)
        p1[Nx, n] = 2/5*(e1[Nx, n]-1/2*rho1[Nx, n]
                         * u1[Nx, n]**2)  # Pressure
        # NOTE: check input de to the m_de equation.
        de1[Nx] = m_de(T2[Nx, n], p1[Nx, n], Ts2[Nx], de1[Nx], 0.)
        del_SN = de0[Nx]/np.pi/D/rho_sn
        if del_SN > 1e-5:
            q1 = k_sn*(Tw1[Nx]-Ts1[Nx])/del_SN
    #           print("line 848", "q1", q1,"Tw1", Tw1[Nx],"Ts1", Ts1[Nx],"ksn", k_sn)
            assert not math.isnan(Ts1[Nx])
            Ts2[Nx] = Ts1[Nx]+dt/(w_coe*c_c(Tw1[Nx])) * \
                (q1-q_h(Tw1[Nx], BW_coe)*Do/D)
            Tc2[Nx] = Tc1[Nx]+dt/(de0[m]*c_n(Tc1[Nx])/D/np.pi)*(de1[Nx]
                                                                * (1/2*(u1[Nx, n])**2+delta_h(T1[Nx, n], Ts1[Nx])-q1))
        else:
            q1 = de1[Nx]*(1/2*(u1[Nx, n])**2+delta_h(T1[Nx, n], Ts1[Nx]))
            Ts2[Nx] = Ts1[Nx]+dt/(w_coe*c_c(Tw1[Nx])) * \
                (q1-q_h(Tw1[Nx], BW_coe)*Do/D)
            Tc2[Nx] = Ts2[Nx]
        Tw2[Nx] = 2*Tc2[Nx]-Ts2[Nx]
        qhe[Nx] = q_h(Tw1[Nx], BW_coe)*np.pi*Do
        de0[Nx] += dt*np.pi*D*de1[Nx]
        rho2[Nx, n] = max(2*rho2[Nx-1, n]-rho2[Nx-2, n], rho_0)  # Free outflow
        u2[Nx, n] = max(2*rho2[Nx-1, n]*u2[Nx-1, n] -
                        rho2[Nx-2, n]*u2[Nx-2, n], 0) / rho2[Nx, n]
        e2[Nx, n] = 2*e2[Nx-1, n]-e2[Nx-2, n]

#                Set inlet boundary conditions

        # NOTE: COMMENTED ALL BOUNDARY CONDITIONS
        # val_in(i)
        # Tw2[0] = T_in
        # Ts2[0] = T_in
        # Tc2[0] = T_in
        # rho2[0, :] = rho_in
        # p2[0, :] = p_in
        # ux2[0, :] = ux_in
        # ur2[0, :] = ur_in
        # T2[0, :] = T_in
        # e2[0, :] = e_in
        # print("pressure val_in fitting BC", p_in) # from fitting function

        # ------------------------ Temperature Boundary condition ------------------------------------- #
        # Calculate the gas temperature and ensure it's higher than the SN2 surface temperature
        T2 = 2/5*(e2-1/2*rho2 *
                  u2**2)*M_n/rho2/R

# NOTE: check this and compare to 1d case
        for b in np.arange(np.int64(0), np.int64(Nx-1)):
            if (T2[b, Nr] < Ts2[b]):  # or(rho1[Nx]<rho_sw*k_c(T2[Nx],p1p)):
                e2[b, Nr] = 5/2*rho2[b, Nr]*R*Ts2[b] / \
                    M_n+1./2.*rho2[b, Nr]*ur2[b, Nr]**2

        # Set energy Boundary condition at the inlet.
        # rho2[0, :] = rho_in
        # e2[0, :] = 5/2*p_in+1/2*rho2[0, :] * \
        #     np.square(u2[0, :]) + 5./2.*rho1[m, n]/M_n*R*T1[0, :]

        # -------------------------------- CHECK ARRAYS FOR NEGATIVE VALUES ------------------------------------- #
        arrays = [ux2, ur2, T2, e2, p2, rho2, Tw2, Ts2, Tc2]
        for s in np.arange(len(arrays)):
            check_array(arrays[s])

        # --------------------------------- Sending back the results of current time step ------------------------- #

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

        # -------------------------------------- DELETING R=0 Point/Column  ---------------------------------------------------
        # The 3 index indicates matrices with no r=0, deleted column..
        rho3, ux3, ur3, u3, e3, T3, p3 = delete_r0_point(
            rho2, ux2, ur2, u2, e2, T2, p2)

        print("shape rho3", np.shape(rho3))

        save_data(i, dt, rho3, ux3, ur3, u3, e3, T3, Tw2, Ts2, de0, p3, de1)

## -------------------------------------------- Plotting values after BCs-------------------------------------------- ##

        # fig, axs = plt.subplots(2, 2)
#        print("Radius", R_cyl)
        r1 = np.linspace(0, R_cyl, Nr+1)  # r = 0 plotted
        r = np.delete(r1, 0, axis=0)  # r = 0 point removed from array
#        print("array", r)
        X = np.linspace(0, L, Nx+1)

#        print("linspace", R)
       # print("shape r", np.shape(r))
        # RADIAL DIRECTION
        # a = rho1[0,:]
        b = u3[0, :]
        c = T3[0, :]
        # d = Ts1[:]
        # e = Tw1[0,:]
        f = p3[0, :]

        # AXIAL DIRECTION
        # a = rho3[:,Nr]
        # b = u3[:, Nr]
        # c = T1[:, Nr]
        # d = Ts1[:]
        # e = Tw1[:]
        # f = p3[:, Nr]
        # g= de1[:]
        # h= de0[:]

#       #        print("shape y", np.shape(y))
        # plt.scatter(r,a)
        # plt.scatter(r,b)
#        print("shape T3", np.shape(T3), "shape r", np.shape(r))
        fig, axs = plt.subplots(4)
        fig.tight_layout()
        fig.suptitle('Properties along radial axis @ m=0')
        axs[0].scatter(r, b, label="Velocity", color='red')
        axs[0].set(ylabel='U [m/s]')
        # plt.ylabel("Velocity [m/s]")
        axs[1].scatter(r, c, label="Temperature", color='blue')
        axs[1].set(ylabel='Temperature [K]')
        # plt.ylabel("Temperature [K]")
        axs[2].scatter(r, f, label="Pressure", color='green')
        axs[2].set(ylabel='Pressure [Pa]')
        # plt.ylabel("Pressure [Pa]")
        axs[3].scatter(r, b, label="Ur", color='yellow')
        axs[3].set(ylabel='Ur [m/s]')
        plt.xlabel("radius (m)")

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
        plt.show()


# define global tx to save in worksheets.

#        tx = t

    return


if __name__ == "__main__":
    main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2,
             ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3)
