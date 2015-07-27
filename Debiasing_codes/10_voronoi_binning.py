#!/usr/bin/env python

"""
Copyright (C) 2003-2014, Michele Cappellari
E-mail: cappellari_at_astro.ox.ac.uk

    V1.0.0: Michele Cappellari, Vicenza, 13 February 2003
    V1.0.1: Use astro library routines to read and write files.
        MC, Leiden, 24 July 2003
    V2.0.0: Translated from IDL into Python. MC, London, 19 March 2014
    V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014

"""

from __future__ import print_function

import numpy as np
from time import clock

from voronoi_2d_binning import voronoi_2d_binning
from astropy.io import fits

# This part added by me to load our data without the need to create files earlier. #
############################################################################
import params

source_dir = params.source_dir
full_sample = params.full_sample
N_cut = params.N_cut
p_cut = params.p_cut
select = np.load(source_dir + "full_cut.npy")
############################################################################
############################################################################
full_data = fits.getdata(source_dir + full_sample,1)

data_table = np.array([full_data.field(c) for c in ["PETROR50_R_KPC","PETROMAG_MR"]])
data_table = np.concatenate([data_table, np.ones((2,len(data_table.T)))])

data_table = ((data_table.T)[select]).T
############################################################################

#-----------------------------------------------------------------------------

def voronoi_binning_example(data_table):
    """
    Usage example for the procedure VORONOI_2D_BINNING.

    It is assumed below that the file voronoi_2d_binning_example.txt
    resides in the current directory. Here columns 1-4 of the text file
    contain respectively the x, y coordinates of each SAURON lens
    and the corresponding Signal and Noise.

    """

    x, y, signal, noise = data_table
    targetSN = 25

    n = 500
    val, xedge, yedge = np.histogram2d(x, y, n)
    print(val)
    x = 0.5*(xedge[:-1] + xedge[1:])
    y = 0.5*(yedge[:-1] + yedge[1:])

#######################

    np.savetxt('d20_bin_pos.out.txt', np.array([[np.min(x),np.max(x),
               0.5*(xedge[1]-xedge[0])],[np.min(y),np.max(y),
               0.5*(yedge[1]-yedge[0])]]))

    x=(x-np.min(x))/(np.max(x)-np.min(x))
    y=(y-np.min(y))/(np.max(y)-np.min(y))

#######################

    x = x.repeat(n).reshape(n, n).ravel()
    y = y.repeat(n).reshape(n, n).T.ravel()
    signal = val.ravel()
    noise = np.sqrt(signal)
    ok = signal > 0
    signal = signal[ok]
    noise = noise[ok]
    x = x[ok]
    y = y[ok]

    for i in range(len(signal)):
        print(x[i], y[i], signal[i], noise[i])
        
    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, targetSN, plot=1, quiet=0, wvt=True)

    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
    np.savetxt(source_dir + "bin_edges.out.txt", np.column_stack([x, y, binNum]),
               fmt=b'%10.6f %10.6f %8i')

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    t = clock()
    voronoi_binning_example(data_table)
    print('Elapsed time: %.2f seconds' % (clock() - t))
