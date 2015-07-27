# coding: utf-8

# 2/3 scripts to run

# Import packages. #############################################################
################################################################################

from __future__ import division
import numpy as np
from scipy.optimize import curve_fit
import math

n_morph = 6

def load_data():

    # Load the parameters from Function_fitting.py
    ############################################################################

    data=np.load("npy/fixed_bin_size_params_2.npy")

    return data

def f(x,A0,AM,AR,Az):

    # Function is linear combination of (magnitude, size, redshift) + an offset
    ############################################################################

    return A0 + AM*x[0] + AR*x[1] + Az*x[2]

def fit_mrz(data):

    # Fit a linear function of M, R and z to the bins. #########################
    ############################################################################

    params=np.zeros((n_morph,9))

    kmin=np.zeros((n_morph,2))
    kmax=np.zeros((n_morph,2))
    cmin=np.zeros((n_morph,2))
    cmax=np.zeros((n_morph,2))

    # Loop over GZ morphologies
    for a in range(0,n_morph):

        data_arm=data[data['answer'] == a]

        M=data_arm['M_r']
        R=data_arm['R_50']
        redshift=data_arm['redshift']

        x=np.array([M,R,redshift])

        k=data_arm['k']
        c=data_arm['c']

        cmax[a,:]=data_arm[['k', 'c']][np.argmax(c)]
        cmin[a,:]=data_arm[['k', 'c']][np.argmin(c)]
        kmax[a,:]=data_arm[['k', 'c']][np.argmax(k)]
        kmin[a,:]=data_arm[['k', 'c']][np.argmin(k)]

        kp,kc=curve_fit(f,x,k,maxfev=1000) # Fit k and c to the parameters.
        cp,cc=curve_fit(f,x,c,maxfev=1000)

        params[a,0]=a
        params[a,1:5]=kp
        params[a,5:]=cp

    return params,cmin,cmax,kmin,kmax

def save_params(params,cmin,cmax,kmin,kmax):

    # Save results to numpy arrays
    ############################################################################

    np.save("npy/kc_fit_params.npy",params)
    np.save("npy/cmin.npy",cmin)
    np.save("npy/cmax.npy",cmax)
    np.save("npy/kmin.npy",kmin)
    np.save("npy/kmax.npy",kmax)

    return None

################################################################################
################################################################################

if __name__ == "__main__":

    data = load_data()
    params,cmin,cmax,kmin,kmax = fit_mrz(data)
    save_params(params,cmin,cmax,kmin,kmax)
