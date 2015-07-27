
# coding: utf-8

# In[1]:

from scipy.optimize import curve_fit
from prefig import Prefig

Prefig()

#  Import the required files (the complete data set FITS and voronoi bin data).

os.chdir("/home/ppxrh/Zoo_catalogues/Week_9/FITS")
gal_data=fits.getdata("d20.fits",1)

os.chdir("/home/ppxrh/Zoo_catalogues/voronoi")
pos=np.loadtxt("d20_bin_pos.out.txt")


# In[2]:

vor_bins=(np.loadtxt("d20.out.txt")).T

MR_data=np.array([gal_data.field("PETROMAG_MR"),gal_data.field("R50_KPC")])

n=200

for c in [0,1]:

    vor_bins[c]=(pos[c,1]-pos[c,0])*vor_bins[c]+pos[c,0]
        
#xwidth=(np.max(vor_bins[0])-np.min(vor_bins[0]))/(2*(n))
#ywidth=(np.max(vor_bins[1])-np.min(vor_bins[1]))/(2*(n))

vor_bins=np.array([vor_bins[0]-pos[0,2],vor_bins[0]+pos[0,2],vor_bins[1]-pos[1,2],vor_bins[1]+pos[1,2],vor_bins[2]+1])


# In[3]:

gal_vb=np.zeros((1,len(gal_data)))

for r in range(0,len(gal_data)):
    
    M=MR_data[0,r]
    R=MR_data[1,r]
    
    sel=(M > vor_bins[0]) & (M <= vor_bins[1]) & (R > vor_bins[2]) & (R <= vor_bins[3])
    
    if np.sum(sel) != 0:
    
        gal_vb[0,r]=vor_bins[4][sel][0]
        
MR_data_2=np.concatenate([MR_data,gal_vb])

MR_data_2[2][MR_data_2[2] == 0]=np.max(MR_data_2[2])

#MR_data_2=(MR_data_2.T[gal_data.field("PETROMAG_R") <= 17]).T


# In[4]:

plot=1

if plot ==1:

    for v in range(int(np.min(MR_data_2[2])),int(np.max(MR_data_2[2]))+1):
            
            plt.plot(MR_data_2[0][MR_data_2[2] == v],MR_data_2[1][MR_data_2[2] == v],".")
            
    plt.ylabel("$R_{50}$ (kpc)")
    plt.xlabel("$M_r$") 
    
    plt.xlim(-24,-16)
    plt.ylim(0,20)
    
    plt.show()


# In[5]:

cols=["t11_arms_number_a31_1_weighted_fraction","t11_arms_number_a32_2_weighted_fraction",
      "t11_arms_number_a33_3_weighted_fraction","t11_arms_number_a34_4_weighted_fraction",
      "t11_arms_number_a36_more_than_4_weighted_fraction","t11_arms_number_a37_cant_tell_weighted_fraction",
      "t04_spiral_a08_spiral_count","REDSHIFT_1"]


# In[6]:

# Cell for assigning a z bin to each of the galaxies with a voronoi bin.

###################

v_min=int(np.min(MR_data_2[2]))
v_max=int(np.max(MR_data_2[2]))

min_gals=50

###################

arms=np.array([gal_data.field(c) for c in cols])

arms[6]=1/arms[6]

vor_z=np.zeros((6,len(arms[0])))

full_tb=np.concatenate([[MR_data_2[2]],vor_z,arms])


# In[7]:

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

np.save("vor_arm_z.npy",full_tb)


# In[9]:

for c in range(15):

    print(c,":",np.min(full_tb[c]), "->" ,np.max(full_tb[c]))


# In[9]:



