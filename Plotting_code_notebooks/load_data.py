import numpy as np
from astropy.io import fits
import scipy.stats.distributions as dist
import params

data_file = params.filename
debiased_table = params.deb_vals
gz_dir = params.gz_dir

##############################################################################
##############################################################################
def get_column(c,data,col_name):
  
    if len(c) == 2:
        column = data.field(c[0]) - data.field(c[1])
    else:
        column = data.field(c[0])
        if len(c) > 2:
            print(col_name + "has >2 fields. Only the first field was used.")
	    
    return column
    

def load(cx,cy,p_th,N_th,norm,p_values):
    '''
    Loads the data------------------------------------------------------------
    --------------------------------------------------------------------------
    Arguments:

    cx,cy: are the columns to load that will be plotted in x and y.
    
    p_th,N_th: are the min. threshold for spiral vote fraction + number.
    
    norm: f norm is True, debiased values are normalised to =1.
    
    p_values: can set as "w" (Willett 2013),"r" (raw), or "d" (debiased).
    --------------------------------------------------------------------------
    Returns:
    
    tb_reduced: spiral galaxy table.
    tb: table with all galaxies in the volume-limited sample.
    
    * Both arrays have the following columns:
    [p_1,p_2,p_3,p_4,p_5+,p_ct,y-values,x-values]
    --------------------------------------------------------------------------
    '''

    gal_data = fits.getdata(gz_dir 
        + data_file,1) # FITS containing the 
    # volume-limited sample is loaded.
    cols = ["t11_arms_number_a31_1_","t11_arms_number_a32_2_"
        ,"t11_arms_number_a33_3_","t11_arms_number_a34_4_"
        ,"t11_arms_number_a36_more_than_4_","t11_arms_number_a37_cant_tell_"]  
    # Columns containing the vote data from the FITS file.
    
    if p_values == "w": # This section of code loads the vote fractions.
        cols = [s + "debiased" for s in cols]
        f_v = np.array([gal_data.field(c) for c in cols]).T     
    elif p_values == "r":
        cols = [s + "weighted_fraction" for s in cols]
        f_v = np.array([gal_data.field(c) for c in cols]).T 
    else:  
        f_v = np.load(gz_dir + debiased_table).T
        if norm is True:
            debiased = (debiased.T/np.sum(debiased,axis=1)).T
            
    x_column = get_column(c=cx,data=gal_data,col_name="x") 
    y_column = get_column(c=cy,data=gal_data,col_name="y") # Get the data from
    # the FITS file.
    
    tb = np.concatenate([f_v,np.array([y_column,x_column]).T],axis=1)
        
    p_spiral = (
        gal_data.field("t01_smooth_or_features_a02_features_or_disk_debiased")
        *gal_data.field("t02_edgeon_a05_no_debiased")
        *gal_data.field("t04_spiral_a08_spiral_debiased"))
    N_spiral = (gal_data.field("t04_spiral_a08_spiral_count")) # Load values to
    # allow data cuts to be made.
    
    tb_reduced = tb[(p_spiral > p_th) & (N_spiral > N_th) # Threshold cut.
        & (np.isfinite(tb[:,-1])) & (np.isfinite(tb[:,-2]))
        & (tb[:,-1] > -999) & (tb[:,-2] > -999)] # Check values sre finite.

    return tb_reduced,tb

  
def assign(table,Nb,th,equal_samples,redistribute,rd_th,ct_th,print_sizes):
    '''
    Bins the data by a specific column. --------------------------------------
    --------------------------------------------------------------------------
    Arguments:
    
    table: input table to bin.
    
    Nb: number of bins to divide the data in to.

    th: threshold for a galaxy to count as a specific arm number.
    
    equal_samples: if True, bin into equally sized samples. If False, bin 
    into equally spaced bins.
    
    redistribute: if True, can't tell galaxies are put in to other categories.
    
    rd_th: min value of P_n/P_ct to redistribute the galaxies (eg 0.5).
    
    ct_th: max value of P_ct to redsitribute (eg. 0.5).
    
    print_sizes: if True, sample sizes will be printed at the end.
    --------------------------------------------------------------------------
    Returns:
    
    bins: array with the following columns: [bin assignment,arm no]
    table: corresponding table (now sorted).
    --------------------------------------------------------------------------
    '''
    
    table = table[np.argsort(table[:,-1])] # Sort the table by binning column.
    fracs = np.zeros((Nb,3)) # Pre-allocate array with Nb bins.
    
    if equal_samples is True:
        bin_sp = np.linspace(0,1,Nb+1)
        bin_sp[-1] = 2
        bin_v = np.linspace(0,1,len(table))
        bins = np.digitize(bin_v,bins=bin_sp)     
    else:
        bin_sp = np.linspace(np.min(table[:,-1]),np.max(table[:,-1]),Nb+1)
        bin_sp[-1] = bin_sp[-1]+1
        bins = np.digitize(table[:,-1],bins=bin_sp)

    arm_assignments = np.ones((1,len(table)))*(-999) # Assigned arm numbers 
    # initially is an array of -999s. -999 means 'no assignment'.
    for m in range(6):
        a = (np.argmax(table[:,:6],axis=1) == m) & (table[:,m] >= th)
        arm_assignments[:,a] = m
        
    if redistribute is True: # Redistribute according to thresholds.
        for m in range(5):
            arm_assignments[(np.argmax(table[:,:5],axis=1) == m) 
                & (arm_assignments == 5) & (table[:,m]/table[:,5] > rd_th) 
                & (table[:,5] <= ct_th)] = m

    if print_sizes is True:
        print("total sample: " + str(len(bins)))
        print("total 'assigned' sample: " 
	    + str(np.sum(arm_assignments[0] != -999)))
        for m in range(6):
            print("m = " + str(m+1) + ": " 
                + str(np.sum(arm_assignments[0] == m)))
      
    return (np.array([bins,arm_assignments[0]])).T,table

  
def get_fracs(table,bins,m,c,full_data):
    '''
    Get fractions according to the binned data assignments.-------------------
    --------------------------------------------------------------------------
    Arguments:
    
    table: input table that has been binned.
    
    bins: array returned from the 'assign' function.

    m: arm number considered here.
    
    c: error value eg. 0.683 for 1 sigma.
    
    full_data: can be "all spirals" for all of the spiral galaxies or 
    "assigned spirals" for only spiral galaxies with an assigned arm number.
    --------------------------------------------------------------------------
    Returns:
    
    fracs: table with the following columns:
    [Mean bin value, N_gal, N_tot, lower fraction limit, upper fraction limit]
    --------------------------------------------------------------------------
    '''
    
    fracs = np.zeros((np.max(bins[:,0]),5)) # Pre-assign the fractions array.
    
    for n in range(int(np.max(bins[:,0]))):
      
        s_tr = table[bins[:,0] == n+1]
        fracs[n,0] = np.mean(s_tr[:,-1]) # Mean of the 'binned by' parameter 
        # for each bin.
        
        if full_data == "all spirals":
            bin_n = bins[bins[:,0] == n+1]
	    
        else:
            bin_n = bins[(bins[:,0] == n+1) & (bins[:,1] != -999) 
                & (bins[:,1] != 5)]
            if full_data != "assigned spirals":
                print("Invalid full_data string; using 'assigned spirals'")
        
        fracs[n,2] = len(bin_n) # Total bin size.
        
        bin_a = bin_n[bin_n[:,1] == m]
        fracs[n,1] = len(bin_a) # Number of gals assigned with m arms for 
        # each bin.
        
    for r in range(len(fracs)): # Now get errors according to the Cameron
    # et al. paper:
        n = fracs[r,2] # Number of gals w. m arms.
        k = fracs[r,1] # Number of gals in the bin in total.

        fracs[r,3] = dist.beta.ppf((1-c)/2.,k+1,n-k+1)
        fracs[r,4] = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)
        
    return fracs
  
  ############################################################################
  ############################################################################
