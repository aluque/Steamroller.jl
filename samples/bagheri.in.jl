#=
This sample input code reproduces the setup described in the article
"Comparison of six simulation codes for positive streamers in air," B Bagheri et al 2018
Plasma Sources Sci. Technol. 27 095002.

All dimensional quantities must be expressed in SI units.
=#
import Steamroller as sr

#
# AUXILIARY VARIABLES
#

# Numerical type. Consider Float32 as experimental.
T = Float64

## INITIAL CONDITIONS
A  = 5e18        # Peak of the gaussian ion density
w   = 4e-4       # Width of the gaussian
z0  = 0.01       # Location of the gaussian peak
nbg = 1e13       # Background density


#
# PROPER INPUT PARAMETERS
#
# Spatial dimensions. D=3 and D=3 are supported
D = 2

# Initial conditions. Specified as a list of pairs (species number) => initial density. See
# initcond.jl for more options to specify initial conditions.
initial_conditions = [1 => sr.Background(nbg),                  # electrons
                      2 => sr.Background(nbg) + sr.Gaussian(;A, z0, w)]

trans = sr.BagheriTransportModel()
chem = sr.NetIonization(trans)

# Number of blocks at root level
rootsize = (1, 1)

# Size of each root-level block
L = 1.25e-2

# Whether to set free boundary conditions as described by Malagón-Romero and Luque Comp. Phys. Comm.
# 225(2018)114–121
freebnd = false
flux_scheme = :koren

# Background field
eb = [0.0, -18.75e3 / L]

#init_populate_level = 4

# empty_photoionization sets up a simulation without photo-ionization (first test case in Bagheri
# et al. 2018).  Use bourdon2 / bourdon3 for the 2- or 3-term fitting by Bourdon A, Bonaventura Z
# and Celestin S 2010 Plasma Sources Sci. Technol. 19 034012.
phmodel = sr.empty_photoionization(T)
# phmodel = sr.bourdon2(T, D)

# Density-based refinement criterium threshold
refine_density_value = 1e16

# The density based criterium will be applied up to this time.
refine_density_upto = 4e-9

# Do not refine below this level
refine_min_h = 5e-7

# Do not derefine if the resulting spacing is larger than this
derefine_max_h = 20e-6

# The parameters for the Teunissen refinement criterium
refine_teunissen_c0 = 1.0
refine_teunissen_c1 = 1.3

# Refinement flags persist for this time.
refine_persistence = 2e-10

poisson_fmg = false
poisson_iter = 2
poisson_level_iter = (2, 2, 2)

# Output times
output = 0:1e-9:16e-9
