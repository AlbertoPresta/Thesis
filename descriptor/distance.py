import numpy as np
import matplotlib.pyplot as plt



# Function to calculate Chi-distace
def chi2_distance(A, B):
    chi = 0

    for i in range(78):
        if ((A[i] + B[i])**2==0):

            continue
        else:

            chi = chi + ((A[i] - B[i])**2)/((A[i] + B[i])**2)
    return 0.5*chi

def Eucledian_distance(A,B):
    return np.linalg.norm(A - B)
