import matplotlib.pyplot as plt
import urllib
from PIL import Image
from astropy.io import fits
import numpy as np
import random
import os

#--------------------------------------------------------------------------------------

def load_data(columns,p_th,N_th,gz_dir):

    gal_data = fits.getdata(gz_dir + "Volume_limited_sample_Baldry_w_bd.fits",1)
    
    cols_data = ["t11_arms_number_a31_1_","t11_arms_number_a32_2_",
                 "t11_arms_number_a33_3_","t11_arms_number_a34_4_",
                 "t11_arms_number_a36_more_than_4_","t11_arms_number_a37_cant_tell_"]
    
    urls = gal_data.field("jpeg_url")
    
    cols_data = [s + "weighted_fraction" for s in cols_data]
    
    raw = np.array([gal_data.field(c) for c in cols_data]).T 
    debiased = np.load(gz_dir + "debiased.npy").T
    
    c_array = np.zeros((len(raw),len(columns)))
    check_finite = np.ones(len(c_array))
    check_finite = check_finite.astype(int)
    
    for c in range(len(columns)):
        
        cx = columns[c]

        if len(cx) == 2:
            x_column = gal_data.field(cx[0]) - gal_data.field(cx[1]) # Can have 2 columns if you want colours etc. 
        else:
            x_column = gal_data.field(cx[0])

        c_array[:,c] = x_column
        check_finite = check_finite*(np.isfinite(x_column))*(x_column > -999)
        
    p_spiral = (gal_data.field("t01_smooth_or_features_a02_features_or_disk_debiased")*
                gal_data.field("t02_edgeon_a05_no_debiased")*
                gal_data.field("t04_spiral_a08_spiral_debiased"))
    N_spiral = (gal_data.field("t04_spiral_a08_spiral_count"))
    
    i = np.array([np.arange(0,len(p_spiral))]).T # index for jpeg url column
    
    full_tb = np.concatenate([raw,debiased,c_array,i],axis=1)
    
    tb = full_tb[(p_spiral > p_th) & (N_spiral >= N_th) & (check_finite == 1)] # Can (hopefully) 
    # remove any entries without data. 
    
    return full_tb,tb,urls # Has the following data:[p_1 -> p_ct (raw); p_1 -> p_ct (debiased); column_1,column_2,...; url index]
  
#--------------------------------------------------------------------------------------

def cut_data(table,column,values):
    return table[(table[:,column+12] > values[0]) & (table[:,column+12] <= values[1])]

#------------------------------------------------------------


def cut_probability(table,values):
    return table[(np.max(table[:,6:12],axis=1) > values[0]) & (np.max(table[:,6:12],axis=1) <= values[1])]

#------------------------------------------------------------

def assign(table,bin_column,Nb,bin_range,set_manual):
    
    if set_manual is True:
        bin_edges = np.linspace(bin_range[0],bin_range[1],Nb+1)
                                
    else:
        bin_edges = np.linspace(np.min(table[:,bin_column+12]),np.max(table[:,bin_column+12]),Nb+1)
    
    bins = np.digitize(table[:,bin_column+12],bin_edges)
    
    m=np.argmax(table[:,6:12],axis=1)
    B = np.array([bins,m]).T
    
    return B,bin_edges

#------------------------------------------------------------

def display_image(url_name,crop_in):
    
    # Get the url name:
    urllib.request.urlretrieve(url_name,"image.jpg")
    
    # Open -> crop -> display -> remove the image.
    im=Image.open("image.jpg")
    l=424 # Image size
    im=im.crop((crop_in,crop_in,l-crop_in,l-crop_in))
    plt.imshow(im)
    os.remove("image.jpg") 
    
    plt.xticks([])
    plt.yticks([])
    
    return None
  
#------------------------------------------------------------

def plot_images(table,bins,urls,relative_size):
    
    H=30*relative_size
    W=21*relative_size
    
    n = 0
    
    plt.figure(figsize=(W,H))
    
    for m in range(6):
        
        for z in range(5):
            
            n = n+1
            
            t_mz = table[(bins[:,0] == z+1) & (bins[:,1] == m)]
            
            if len(t_mz) >= 1:
                
                plt.subplot(6,5,n)

                i = random.choice(range(0,len(t_mz)))
                
                p_deb = t_mz[i,m+6]
                p_raw = t_mz[i,m]
                
                url = urls[t_mz[i,-1]]
                
                plt.text(x=112, y=15, s="$p_{debiased}="+"{0:.2f}".format(p_deb)+
                         ",p_{raw}="+"{0:.2f}".format(p_raw) + "$", fontsize=12,
                         ha='center', va='center',size=20,color="white")
                
                display_image(url_name=url,crop_in=100)
                
    plt.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.02,hspace=0.01,wspace=0.01)
            
    return None
  
#-----------------------------------------------------------

def add_labels(z_edges):
    
    y_label_positions = [1,6,11,16,21,26]
    x_label_positions = range(26,31)
    
    y_labels = ["$m=1$","$m=2$","$m=3$","$m=4$","$m=5$","$m=??$"]
    x_labels = ["${0:.3f} < z".format(z_edges[n]) + 
                "\leq {0:.3f}$".format(z_edges[n+1]) for n in range(len(z_edges) - 1)]
    
    for xn in range(len(x_label_positions)):
        
        plt.subplot(6,5,x_label_positions[xn])
        plt.xlabel(x_labels[xn],fontsize=20,family="serif")
        
    for yn in range(len(y_label_positions)):
        
        plt.subplot(6,5,y_label_positions[yn])
        plt.ylabel(y_labels[yn],fontsize=20,family="serif")

    return None

#----------------------------------

def make_axes(relative_size):

    H=30*relative_size
    W=21*relative_size
    plt.figure(figsize=(W,H))
    
    return None
