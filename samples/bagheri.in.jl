import Steamroller as sr

#
# AUXILIARY VARIABLES
#
T = Float64

## INITIAL CONDITIONS
A  = 5e18       # Peak of the gaussian ion density
w   = 4e-4       # Width of the gaussian
z0  = 0.01       # Location of the gaussian peak
nbg = 1e13       # Background density


#
# PROPER INPUT PARAMETERS
#
D = 2
initial_conditions = [1 => sr.Background(nbg),                  # electrons
                      2 => sr.Background(nbg) + sr.Gaussian(;A, z0, w)]

trans = sr.BagheriTransportModel()
chem = sr.NetIonization(trans)

rootsize = (1, 1)
L = 1.25e-2
freebnd = false
flux_scheme = :koren

# Background field
eb = [0.0, -18.75e3 / L]

#init_populate_level = 4

phmodel = sr.empty_photoionization(T)
#phmodel = sr.bourdon2(T, D)

refine_density_value = 1e16
refine_density_upto = 4e-9

# Do not refine below this level
refine_min_h = 5e-7

# Do not derefine if the resulting spacing is larger than this
derefine_max_h = 20e-6

# The parameters for the Teunissen refinement criterium
refine_teunissen_c0 = 1.0
refine_teunissen_c1 = 1.3

refine_persistence = 2e-10

poisson_fmg = false
poisson_iter = 2
poisson_level_iter = (2, 2, 2)

output = 0:1e-9:16e-9
