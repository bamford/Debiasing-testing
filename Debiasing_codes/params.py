source_dir = "../../fits/" # Directory containing the fits files.

full_sample = "Full_sample_spec_w_urls.fits" # Full sample of galaxies (to be voronoi binned). This should contain all of the galaxies in the sample.
vl_sample = "Volume_limited_sample_Baldry_w_bd.fits" # Volume-limited galaxy sample to be debiased.

N_cut = 10 # Only include galaxies with >= N_cut spiral votes.
p_cut = 0.5 # Only include galaxies with > p_cut spiral vote fraction.