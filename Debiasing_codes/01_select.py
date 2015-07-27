import numpy as np
from astropy.io import fits
###############################
import params

source_dir = params.source_dir
full_sample = params.full_sample
vl_sample = params.vl_sample
N_cut = params.N_cut
p_cut = params.p_cut
###############################

def select(data,N,p,name):

    p_spiral=(data.field("t01_smooth_or_features_a02_features_or_disk_debiased")*
              data.field("t02_edgeon_a05_no_debiased")*
              data.field("t04_spiral_a08_spiral_debiased"))

    N_spiral=(data.field("t04_spiral_a08_spiral_count"))

    select = (p_spiral > p_cut) & (N_spiral >= N_cut) # Only select gals. that make the N and p cuts.
    
    # Save this table for future use:
    np.save(source_dir + name + ".npy",select)
    
    return None
  
###############################

full_data = fits.getdata(source_dir + full_sample,1)
vl_data = fits.getdata(source_dir + vl_sample,1)

select(data=full_data,N=N_cut,p=p_cut,name="full_cut")
select(data=full_data,N=N_cut,p=p_cut,name="vl_cut")
###############################
