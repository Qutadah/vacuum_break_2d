import math
import scipy
from scipy import constants
import numpy as np


D = 2.54e-2
M_n = 0.028
R = constants.R
k_b = constants.k
N_A = constants.N_A
d_n = 364e-12
Pr_n = 0.72

#   Time and spatial step
L = 6.45  # With prepping region.
# L = 6.45
Nx = 450  # Total length & spatial step - x direction 6.45
R_cyl = 1.27e-2
Nr = 16  # Total length & spatial step - r direction
T_sim = 3.
Nt = 1000000.  # Total time & time step
dt = T_sim/Nt
dx = L/Nx
dr = R_cyl/Nr
print("dx", dx, "dr", dr, "dt", dt, "Cou_r",
      20*dt/dr, "Cou_x_Max", 800*dt/dx)
gamma_n2 = 1.4
#   Tuning parameters
# Coefficient for the film boiling of He I (we need to adjust optimal value)
Sc_PP = 0.95  # Condensation\boiling coeffcient

#   Parameters
sample = 100  # Sample coeffcient
n_trans = 50  # Smoothing control for initial condition
T_s = 4.2  # Temperature boundaries

#   Constants
M_n = 0.028
R = constants.R
k_b = constants.k
N_A = constants.N_A
Pr_n = 0.72
d_n = 364e-12  # Molecule diameter of GN2
D = 2.54e-2
Do = 2.79e-2
A = np.pi/4.*D**2.  # Inner diameter/Outer diameter/Inner cross-section area of copper pipe
rho_cu = 8960.  # Copper density
rho_sn = 1020.
k_sn = 0.9  # Density and thermal conductivity of SN2
w_coe = rho_cu*(Do**2.-D**2.)/4./D
W_p = np.pi * D
D_hyd = 4 * A / W_p

# w_coe_s = rho_ss*(Do_s**2.-D_s**2.)/4./D  # Interim parameters

# Q How do I use the bath diameter
# D_b = 12.*2.54e-2  # Bath diameter, m

# L_h = 0.7208/0.05  # Copper pipe length per bath depth, m/m
dH_He = 12000  # 20720.59@1atm  # Latent heat of LHe, J/kg
dH_He_V = dH_He*125  # Latent heat of LHe, J/m^3


# Stability factors
F = 1.*dt/dx**2.  # Stability indictor   ### Q:
artv = 0.06  # Control parameter for the artificial viscosity
