################################################################################
# Import packages ##############################################################

import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits

#################################################################################
import params

source_dir = params.source_dir
full_sample = params.full_sample
N_cut = params.N_cut
p_cut = params.p_cut

select = np.load(source_dir + "full_cut.npy")
#################################################################################

# Can set code to plot bins (voronoi cells) and plot galaxies (where each galaxy
# is colour coded to its voronoi bin) to check that the binning is reasonable: 

plot_bins=False
plot_gals=False

# Load the data ################################################################
grid_data = np.loadtxt(source_dir + "bin_edges.out.txt") # Output from voronoi binning.
full_data = fits.getdata(source_dir + full_sample,1) # FITS file with all of the data.

gal_tb=np.array([full_data.field(c) for c in ["PETROR50_R_KPC","PETROMAG_MR"]])
gal_tb =  ((gal_tb.T)[select]).T

# Plot bins ####################################################################

if plot_bins is True:
    plt.figure(1)

    for N in range(0,int(np.max(grid_data[:,2]))+1):
    
        d=grid_data[grid_data[:,2] == N]
    
        plt.plot(d[:,0],d[:,1],".")
        plt.ylabel(r"$R_{50} (kpc)$")
        plt.xlabel(r"$M_r$")

# Alter the bins (in the voronoi code, all bins are between 0 and 1) ###########

h,w=np.shape(gal_tb)
gal_tb_mod=np.zeros((h,w))

for c in [0,1]:
    
    x=gal_tb[c]
    
    gal_tb_mod[c]=(x-np.min(x))/(np.max(x)-np.min(x))

# Get a grid of voronoi cells and galaxies. ####################################

n=100 # Data was divided in to an n x n grid in the voronoi program. 

x=np.digitize(gal_tb_mod[0],bins=np.linspace(0,1,n+1))
y=np.digitize(gal_tb_mod[1],bins=np.linspace(0,1,n+1))

x_grid=np.digitize(grid_data[:,0],bins=np.linspace(0,1,n+1))
y_grid=np.digitize(grid_data[:,1],bins=np.linspace(0,1,n+1))

grid_xy=np.array([x_grid,y_grid,grid_data[:,2]]).T

gal_xy=np.array([x,y]).T

# Assign each galaxy a voronoi bin #############################################

gal_xy_i=np.zeros((1,len(gal_xy))) 

for r in range(0,len(gal_xy)):

    u=grid_xy[:,2][(grid_xy[:,0] == gal_xy[r,0]) & 
                   (grid_xy[:,1] == gal_xy[r,1])]
    
    if len(u) >0:
    
        u=u[0]
    
        gal_xy_i[0,r]=u
        
gal_xy_i=gal_xy_i[0] # This parameter is a list of voronoi bins for each galaxy.

# It will therefore be saved as an NP table:
#np.save(source_dir + "voronoi_list.npy",gal_xy_i)

# Can now plot each galaxy colour coded by it's voronoi bin ####################

if plot_gals is True:
    
    plt.figure(2)

    for N in range(int(np.min(gal_xy_i)),int(np.max(gal_xy_i))+1):
    
        d=gal_tb.T[gal_xy_i == N]
    
        plt.plot(d[:,0],d[:,1],".")
        
        plt.xlabel(r"$R_{50} (kpc)$")
        plt.ylabel(r"$M_r$")
    
plt.show()

################################################################################
# Cell for assigning a z bin to each of the galaxies with a voronoi bin.

cols=["t11_arms_number_a31_1_weighted_fraction","t11_arms_number_a32_2_weighted_fraction",
      "t11_arms_number_a33_3_weighted_fraction","t11_arms_number_a34_4_weighted_fraction",
      "t11_arms_number_a36_more_than_4_weighted_fraction","t11_arms_number_a37_cant_tell_weighted_fraction",
      "t04_spiral_a08_spiral_count","REDSHIFT_1"]

###################

v_min=int(np.min(gal_xy_i))
v_max=int(np.max(gal_xy_i))

min_gals=50 # Minimum number of galaxies in each bin.

###################

arms=np.array([full_data.field(c) for c in cols])
arms[6]=1/arms[6]
arms =  ((arms.T)[select]).T

vor_z=np.zeros((6,len(arms[0])))

full_tb=np.concatenate([np.array([gal_xy_i]),vor_z,arms])

print(np.array([gal_xy_i]).shape,vor_z.shape,arms.shape)


for v in range(v_min,v_max+1):
    
    v_sel=full_tb[0] == v
    
    v_f=(full_tb.T[v_sel]).T
    
    for a in range(0,6):
        
        vr_sel=(full_tb[0] == v) & (full_tb[a+7] >= arms[6])
    
        vr_f=(full_tb.T[vr_sel]).T
        
        N=len(vr_f.T)/min_gals # Select bins uch that >=50 galaxies with vf>1 are in each bin.
        
        # Can set the min/max bin numbers here. # 
        
        if N < 5:
            
            N=5
            
        #########################################
    
        order_z=np.argsort(vr_f[-1])
        
        vr_f_sorted=(vr_f.T[order_z]).T
    
        bin_edges=np.linspace(np.min(order_z),np.max(order_z),N+1)
        
        bin_edges=bin_edges.astype(int)
        
        z_edges=vr_f_sorted[-1][bin_edges]
        
        z_edges[0]=0
        z_edges[-1]=1
        
        nos=np.digitize(v_f[-1],z_edges)
        
        #print(nos)
        
        ((full_tb[a+1]).T[v_sel])=nos


# In[8]:

np.save(source_dir + "assignments.npy",full_tb)


# In[9]:

for c in range(15):

    print(c,":",np.min(full_tb[c]), "->" ,np.max(full_tb[c]))



