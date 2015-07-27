# Debiasing

Methods for a new debiasing of the Galaxy Zoo 2 data.
Originally by @RossHart.
Refactored by @willetk.
Contributions by @bamford.

## Order of operations

1. `select.py` Select the sample
2. `voronoi_binning.py` Create Voronoi bins in terms of absolute magnitude and size
3. `voronoi_assignment.py` Assign galaxies to the Voronoi bins
4. `redshift_binning` Adaptively bin galaxies in redshift to maintain roughly constant number of well
    classified galaxies (?) per bin
5. `fitting.py` Fits a logistic curve to the cumulative histograms for each bin
6. `linear_fits.py` Fits the parameters of the logistic function with
    a linear function of M, R and z
7. `debias.py` Uses the linear fits to fit each galaxy in a given fits file to a low redshift equivalent logistic curve
