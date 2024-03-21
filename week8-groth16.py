import numpy as np
import random
import galois
import functools 
from py_ecc.bn128 import FQ12, G1,G2, pairing, multiply, add, curve_order, eq, Z1, Z2, final_exponentiate, neg

# For test purpose
# curve_order = 79

# Define the matrices
L = np.array([[0,0,3,0,0,0],
               [0,0,0,0,1,0],
               [0,0,1,0,0,0]])

R = np.array([[0,0,1,0,0,0],
               [0,0,0,1,0,0],
               [0,0,0,5,0,0]])

O = np.array([[0,0,0,0,1,0],
               [0,0,0,0,0,1],
               [-3,1,1,2,0,-1]])

print('Initialize GF_curve...')
GF = galois.GF(curve_order)
print('...done!')

nbPublicInputs = 2
nbPrivateInputs = 4

#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
# Functions
#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
def interpolate_column(col):
    xs = GF(list(range(1, len(col)+1)))
    return galois.lagrange_poly(xs, col)

def inner_product(ec_points, coeffs, Z):
    return functools.reduce(add, (multiply(point, int(coeff)) for point, coeff in zip(ec_points, coeffs)), Z)

def matrixToGallois(matrix, order): 
    result = []
    for i, a in enumerate(matrix):
        result.append([])
        for j, v in enumerate(a):
            result[i].append( GF(v) if v >= 0 else GF(order+v) )
    return GF(np.array(result))

def generate_powers_of_tau(tau, degree, G):
    return [multiply(G, int(tau ** i)) for i in range(degree)]

def inner_product_polynomials_with_witness(polys, witness):
    mul_ = lambda x, y: x * y
    sum_ = lambda x, y: x + y
    return functools.reduce(sum_, map(mul_, polys, witness))

def inner_product_points_with_witness(points, witness):
    mul_ = lambda x, y: multiply(x, int(y))
    sum_ = lambda x, y: add(x, y)
    return functools.reduce(sum_, map(mul_, points, witness))

#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
# TRUSTED SETUP
#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
print('## Trusted setup ##')

L_galois = matrixToGallois(L, curve_order)
R_galois = matrixToGallois(R, curve_order)
O_galois = matrixToGallois(O, curve_order)

# axis 0 is the columns. apply_along_axis is the same as doing a for loop over the columns and collecting the results in an array
U_polys = np.apply_along_axis(interpolate_column, 0, L_galois)
V_polys = np.apply_along_axis(interpolate_column, 0, R_galois)
W_polys = np.apply_along_axis(interpolate_column, 0, O_galois)

# Variable to be deleted ### ### ### ### ### ###
tau = GF(random.randint(1,curve_order))
alpha = GF(random.randint(1,curve_order))
beta = GF(random.randint(1,curve_order))
delta = GF(random.randint(1,curve_order))
gamma = GF(random.randint(1,curve_order))
C_polys_public  = int(beta) * U_polys[:nbPublicInputs] + int(alpha) * V_polys[:nbPublicInputs] + W_polys[:nbPublicInputs]
C_polys_private = int(beta) * U_polys[nbPublicInputs:] + int(alpha) * V_polys[nbPublicInputs:] + W_polys[nbPublicInputs:]
# ### ### ### ### ### ### ### ### ### ### ### ###

alpha1 = multiply(G1, int(alpha))
beta2 = multiply(G2, int(beta))
delta2 = multiply(G2, int(delta))
gamma2 = multiply(G2, int(gamma))
delta1 = multiply(G1, int(delta))
beta1 = multiply(G1, int(beta))

t = 1
for i in range(len(R)):
    t = t * galois.Poly([1, curve_order-(i+1)], field = GF)

# power_of_taus
powers_of_tau1 = generate_powers_of_tau(tau, len(R), G1)
powers_of_tau2 = generate_powers_of_tau(tau, len(R), G2)
powers_of_T_tau1 = [multiply(G1, int(GF((tau ** i) * t(tau) / delta))) for i in range(len(R))]
powers_of_tau1_c_public = [multiply(G1, int(GF(p(tau) / gamma)) ) for p in C_polys_public]
powers_of_tau1_c_private = [multiply(G1, int(GF(p(tau) / delta)) ) for p in C_polys_private]

# Destroy
tau = 0
alpha = 0
beta = 0
delta = 0
gamma = 0
C_polys_public = 0
C_polys_private = 0


#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
# Prover steps
#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
print('## Prover steps ##')

# pick values for x and y
x = GF(3)
y = GF(10)

# Compute witness
out = GF(3) * x * x * y + GF(5) * x * y - x - GF(2)  *y + GF(3)
v1 = GF(3)*x*x
v2 = v1 * y
w = GF(np.array([1, out, x, y, v1, v2]))

# pick r & s and destroy it after that
r = GF(random.randint(1,curve_order))
s = GF(random.randint(1,curve_order))

# compute A, B and C
a = inner_product_polynomials_with_witness(U_polys, w)
b = inner_product_polynomials_with_witness(V_polys, w)
c = inner_product_polynomials_with_witness(W_polys, w)

cShiftedprime1_private = inner_product_points_with_witness(powers_of_tau1_c_private, w[nbPublicInputs:])

h = (a * b - c) // t
assert a * b - c == h * t, "Remainder should be 0"


a1 = inner_product(powers_of_tau1, a.coeffs[::-1], Z1)
b2 = inner_product(powers_of_tau2, b.coeffs[::-1], Z2)
b1 = inner_product(powers_of_tau1, b.coeffs[::-1], Z1)

# H(t)T(h)
HT1 = inner_product(powers_of_T_tau1, h.coeffs[::-1], Z1)

## introducing alpha & beta
a1Shifted = add(alpha1, add(a1, multiply(delta1, int(r))))
b2Shifted = add(beta2, add(b2, multiply(delta2, int(s))))
b1Shifted = add(beta1, add(b1, multiply(delta1, int(s))))

# c1Shifted = add(cShiftedprime1, HT1)
c1Shifted_private = add(
    cShiftedprime1_private, 
    add(HT1, 
        add(multiply(a1Shifted, int(s)),
            add(multiply(b1Shifted, int(r)),
                multiply(neg(delta1), int(GF(r*s)))
            )
        )
    )
)

# destroy r & s
r = 0
s = 0

witnessPublic = w[:nbPublicInputs]

# prover returns:
# a1Shifted, b2Shifted, c1Shifted_private, witnessPublic

#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
# Verifier steps
#### ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ####
print('## Verifier steps ##')

# from prover:
# a1Shifted, b2Shifted, c1Shifted_private, witnessPublic

# from trusted setup:
# alpha1, beta2, gamma2, delta2, powers_of_tau1_c_public

# prover computation
cShiftedprime1_public = inner_product_points_with_witness(powers_of_tau1_c_public, witnessPublic)

assert final_exponentiate(
        pairing(b2Shifted, neg(a1Shifted)) 
        * pairing(beta2, alpha1) 
        * pairing(gamma2, cShiftedprime1_public) 
        * pairing(delta2, c1Shifted_private)
    ) == FQ12.one(), "** Verification failed! **"
    

print("Verification passed!")
