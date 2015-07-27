# Debiasing
#
Methods for a new debiasing of the Galaxy Zoo 2 data. Originally by @RossHart. 

i. The first file to run is the file named i_fitting.py- this fits a logistic curve to the cumulative histograms for each bin.
ii. The next file that needs to be run is the file named ii_linear_fits.py. This linearly fits the parameters of the logistic function with respect to M,R and z.
iii. Finally, the file called iii_debias.py uses the linear fits to fit each galaxy in a given fits file to a low redshift equivalent logistic curve. Data is output as a np table.
