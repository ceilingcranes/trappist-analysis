import numpy as np
import random
import math

# Root Mean squares using A^T*A*x = A^T*b method
def rootms(x, y):
    if len(x) != len(y):
        print("Error: Input array sizes must match.")
        return -1
    a1 = [1 for i in range(0, len(x))]

    # Slightly backwards - easier to find A^T and then transpose that ([A^T]^T = A)
    aT = np.matrix([y, a1])
    a = np.transpose(aT)
    b = np.transpose(np.matrix([x]))

    x = np.linalg.inv(aT*a)*aT*b

    return x


# Project a vector x into n-1 dimensional space
def project(x):
    cond = 0
    w = np.zeros(len(x))
    for r in x:
        cond += math.pow(r, 2)
    w[0] = math.sqrt(cond)
    w = np.transpose(np.matrix(w))

    v = w - x
    P = (v * np.transpose(v)) / (np.transpose(v) * v)
    return np.identity(len(P)) - 2 * P

# given a 2xn matrix A and a 1xn matrix b, will solve for a root mean
# squares solution using QR factorization. Returns the x matrix.
def householder(A,b):

    # First column = x
    x=A[:,0:1]

    H1 = project(x)
    H1A = H1*A

    # Value for H2 - second column excluding the first value to avoid messing it up
    x = H1A[1:,1:]
    H2_int = project(x)

    # Put in the correct form - first row and column as ID matrix, again to avoid messing up first value.
    H2 = np.matrix(np.zeros(len(np.array(x))))
    H2 = np.concatenate((H2, H2_int))
    id = np.zeros(len(np.array(x))+1)
    id[0] = 1
    H2 = np.concatenate((np.transpose(np.matrix(id)), H2), axis=1)

    R = H2*H1*A

    Q = H1*H2
    R = R[:2, :2]
    d = (np.transpose(Q)*b)
    d = d[:2]

    # Solve Rx = d where R is upper nxn of R, and d is upper n of Q^T*b
    x = np.linalg.inv(R)*d
    return x

def main():

    epochb = [0,2,6,8,10,12,15,26,28]
    bjyb = [7322.5161,7325.5391,7331.5803,7334.60490,7337.6249,7340.6474,7345.18011,7361.79960,7364.82137]
    a1 = [1 for i in range(0, len(bjyb))]

    # Slightly backwards - easier to find A^T and then transpose that ([A^T]^T = A)
    aT = np.matrix([epochb, a1])
    a = np.transpose(aT)
    b = np.transpose(np.matrix(bjyb))
    estb = householder(a, b)
    print("Trappist 1 b householder estimation: ", estb)

    # compare to other method
    rms = rootms(epochb, bjyb)
    print("Trappist 1b rms: ", rms)

    epochc = [0, 21, 33, 35, 42]
    bjyc = [7282.8058, 7333.6633, 7362.72623, 7367.5699, 7384.5230]
    a1 = [1 for i in range(0, len(bjyc))]
    aT = np.matrix([epochc, a1])
    a = np.transpose(aT)
    b = np.transpose(np.matrix(bjyc))
    estc = householder(a, b)
    print("Trappist 1 c householder estimation: ", estc)

    # Other calculations involving the found period

    # Period in seconds
    period_b = float(estb[0])*24*60*60
    period_c = float(estc[0])*24*60*60

    # The mass of the trappist-1 sun (kg)
    smass = 1.6*10**29

    # Gravitational constant - m^3/kg/s^2
    G = 6.67408*10**-11

    # p^2 = GM/(4pi^2) * a^3
    a_b = math.pow((period_b**2)*G*smass/4/(math.pow(math.pi,2)),1/3)
    # AU/m
    mPerAU = (1.5*10**11)
    a_b = a_b / mPerAU

    a_c = math.pow((period_c**2)*G*smass/4/(math.pow(math.pi,2)),1/3)
    a_c = a_c / mPerAU

    print("semi-major axis of b: ", a_b)
    print("semi-major axis of c: ", a_c)


main()