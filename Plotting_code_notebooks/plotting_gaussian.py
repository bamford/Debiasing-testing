# Put all of the 'shared' plotting functions in here.

from  matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.ndimage
from astropy.io import fits
import numpy as np
import scipy.stats.distributions as dist
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM,z_at_value
import astropy.units as u
import math
from scipy.stats import kde

c = 0.683

cosmo=FlatLambdaCDM(H0=70,Om0=0.3)

gz_dir = "../../fits/"

def get_mass_z_lim(mass_lims,mr_limit): # Has the values [lower mass limit,upper mass limit]

    data = fits.getdata(gz_dir + "Volume_limited_sample_Baldry_w_bd.fits",1)

    colour_mass = np.array([data.field("PETROMAG_MU")-data.field("PETROMAG_MR"),data.field("LOGMSTAR_BALDRY06")]).T
    colour_mass = colour_mass[(colour_mass[:,1] >= mass_lims[0]) & (colour_mass[:,1] < mass_lims[1])]
    u_r_percentile = colour_mass[int(0.001*len(colour_mass))-1,0]
    
    if u_r_percentile > 0.79/0.38:
        ML_limit = -0.16 + 0.18*u_r_percentile
        
    else:
        ML_limit = -0.95 + 0.56*u_r_percentile

    L_limit = (10**mass_lims[0])/(10**ML_limit)
    Mag_limit = -2.5*math.log10(L_limit) + 4.75
    
    D = 10**(((mr_limit-Mag_limit-0.1)/5) + 1)
    z = z_at_value(cosmo.luminosity_distance, (D/10**6) * u.Mpc, zmax=1.5)
    
    return z # => This is the redshift limit to which we have a stellar mass limited sample.

#-------------------------------------------------------------------------------

def load_data(cx,cy,p_th,N_th,norm,p_values,mass_limit,mass_range):
    
    # Loads the data ######################################################
    #######################################################################
    # cx,cy: are the columns to load that will be plotted in x and y.
    # p_th,N_th: are the min. threshold for spiral vote fraction + number.
    # norm: f norm is True, debiased values are normalised to =1.
    # p_values: can set as "w" (Willett 2013),"r" (raw), or "d" (debiased).
    # mass limit: use this to create a stellar mass limited sample, with
    # mass_range [low,high]
    #######################################################################

    #gal_data=fits.getdata("../../Week_9/FITS/Volume_limited_sample_Baldry_w_bd.fits",1)
    gal_data = fits.getdata(gz_dir + "Volume_limited_sample_Baldry_w_bd.fits",1)
    
    cols_data = ["t11_arms_number_a31_1_","t11_arms_number_a32_2_",
                 "t11_arms_number_a33_3_","t11_arms_number_a34_4_",
                 "t11_arms_number_a36_more_than_4_","t11_arms_number_a37_cant_tell_"]
    
    if p_values == "w":
        cols_data = [s + "debiased" for s in cols_data]
        debiased = np.array([gal_data.field(c) for c in cols_data]).T  
    elif p_values == "r":
        cols_data = [s + "weighted_fraction" for s in cols_data]
        debiased = np.array([gal_data.field(c) for c in cols_data]).T 
    else:   
        #debiased=np.load("../../Week_9/FITS/debiased_Volume_limited_sample_Baldry.npy").T
        #debiased = np.load(gz_dir + "debiased_Volume_limited_sample_Baldry.npy").T
        debiased = np.load(gz_dir + "debiased.npy").T
        
        if norm is True:
            debiased = (debiased.T/np.sum(debiased,axis=1)).T

    if len(cx) == 2:
        x_column = gal_data.field(cx[0]) - gal_data.field(cx[1]) # Can have 2 columns if you want colours etc. 
    else:
        x_column = gal_data.field(cx[0])
        
    if len(cy) == 2:
        y_column = gal_data.field(cy[0]) - gal_data.field(cy[1]) # Can have 2 columns if you want colours etc. 
    else:
        y_column = gal_data.field(cy[0])

    tb = np.concatenate([debiased,np.array([y_column,x_column]).T],axis=1)
    
    if mass_limit is True:
        z_lim = get_mass_z_lim(mass_lims=[mass_range[0],mass_range[1]],mr_limit=17)
        print("z ->" + str(z_lim))
    else:
        z_lim = 10
        

    p_spiral = (gal_data.field("t01_smooth_or_features_a02_features_or_disk_debiased")*
                gal_data.field("t02_edgeon_a05_no_debiased")*
                gal_data.field("t04_spiral_a08_spiral_debiased"))
    N_spiral = (gal_data.field("t04_spiral_a08_spiral_count"))
    
    tb_reduced = tb[(p_spiral > p_th) & (N_spiral > N_th) & (np.isfinite(tb[:,-1])) 
        & (np.isfinite(tb[:,-2])) & (tb[:,-1] > -999) & (tb[:,-2] > -999) & 
        (gal_data.field("REDSHIFT_1") <= z_lim) & (gal_data.field("LOGMSTAR_BALDRY06") >= mass_range[0])
        & (gal_data.field("LOGMSTAR_BALDRY06") < mass_range[1])] # Can (hopefully) 
    
    tb = tb[(gal_data.field("REDSHIFT_1") <= z_lim) & (gal_data.field("LOGMSTAR_BALDRY06") >= mass_range[0])
        & (gal_data.field("LOGMSTAR_BALDRY06") < mass_range[1])]
    # remove any entries without data. 

    return tb_reduced,tb # Columns: [p_1,p_2,p_3,p_4,p_5+,p_ct,y-values,x-values]
  
  ###################################################################################################
  ###################################################################################################
  
def assign(table,Nb,th,bin_type,redistribute,rd_th,ct_th):
    
    # Bins the data by a specific column. #################################
    #######################################################################
    # table: input table to bin.
    # Nb: number of bins to divide the data in to.
    # It is not the table column, but the column specified in 'load 
    # data' (ie. 0 rather than 6 for example.)
    # a: arm number
    # th: threshold for a galaxy to count as a specific arm number.
    # bin_type: can set as "equal samples" to bin into equally sized bins.
    # redistribute: if True, can't tell galaxies are put in to other categories.
    # rd_th: min value of P_n/P_ct to redistribute the galaxies (eg 0.5).
    # ct_th: max value of P_ct to redsitribute (eg. 0.5).
    #######################################################################
    
    table = table[np.argsort(table[:,-1])]
    fracs = np.zeros((Nb,3))
    
    if bin_type == "equal samples":
    
        bin_sp = np.linspace(0,1,Nb+1)
        bin_sp[-1] = 2
        bin_v = np.linspace(0,1,len(table))
        bins = np.digitize(bin_v,bins=bin_sp)
            
    else:
        bin_sp = np.linspace(np.min(table[:,-1]),np.max(table[:,-1]),Nb+1)
        bin_sp[-1] = bin_sp[-1]+1
        bins = np.digitize(table[:,-1],bins=bin_sp)

    arm_assignment = np.ones((1,len(table)))*(-999)
    
    for a in range(6):
    
        a_a = (np.argmax(table[:,:6],axis=1) == a) & (table[:,a] >= th)
        arm_assignment[:,a_a] = a
        
    if redistribute is True:
        for a in range(5):
            arm_assignment[(np.argmax(table[:,:5],axis=1) == a) & (arm_assignment == 5) & 
                           (table[:,a]/table[:,5] > rd_th) & (table[:,5] <= ct_th)] = a

    print("total sample:" + str(len(bins)))
    print("m = 1:" + str(np.sum(arm_assignment[0] == 0)))
    print("m = 2:" + str(np.sum(arm_assignment[0] == 1)))
    print("m = 3:" + str(np.sum(arm_assignment[0] == 2)))
    print("m = 4:" + str(np.sum(arm_assignment[0] == 3)))
    print("m = 5+:" + str(np.sum(arm_assignment[0] == 4)))
    print("m = ct:" + str(np.sum(arm_assignment[0] == 5)))
            
    return (np.array([bins,arm_assignment[0]])).T,table # Columns: [bin assignment,arm no]
  
  ###################################################################################################
  ###################################################################################################
  
def get_fracs(table,bins,a,Nb):
    
    fracs = np.zeros((Nb,3))
    
    for n in range(Nb):
        
        s_tr = table[bins[:,0] == n+1]
        fracs[n,0] = np.mean(s_tr[:,-1])
        
        bin_n = bins[bins[:,0] == n+1]
        fracs[n,2] = len(bin_n)
        
        bin_a = bin_n[bin_n[:,1] == a]
        fracs[n,1] = len(bin_a)
        
    fracs = get_errors(fracs)
        
    return fracs # Returned columns: 
#[Mean of value binned by (eg. redshift), N_gal, N_tot, lower fraction limit, upper fraction limit]

def get_errors(fracs):
    
    # Gets the errors according to the Cameron et al. paper.
    ########################################################
    
    p = np.zeros((len(fracs),2))
    
    for r in range(len(fracs)):
        
        n = fracs[r,2]
        k = fracs[r,1]

        p[r,0] = dist.beta.ppf((1-c)/2.,k+1,n-k+1)
        p[r,1] = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)
        
    fracs_ret = np.concatenate([fracs,p],axis=1)
    
    return fracs_ret # Adds lower and upper bounds to each of the fractions.
  
###################################################################################################
###################################################################################################
  
# Codes for making the 3x2 plots:
  
def make_2x3_plot(xlims,ylims,xticks,yticks,x_label,y_label,title_position,f_size,fig_size):

    f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, sharex=True, sharey=True,figsize=fig_size)
    ps=[ax1,ax2,ax3,ax4,ax5,ax6]
    
    for pn in range(6):
        ax=ps[pn]
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
    ax_edit_2x3(f=f,ps=ps,x_label=x_label,y_label=y_label,title_position=title_position,f_size=f_size)

    return f,ps
  
def make_2x3_legend(styles,labels,ax,f_size):
    
    for n in range(len(styles)):
        ax.plot([1000,1000],styles[n],label=labels[n])

    ax.legend(frameon=False,fontsize=f_size)
    
    return None
  
def ax_edit_2x3(f,ps,x_label,y_label,title_position,f_size):
    
    T=["1","2","3","4","5+","??"]
        
    for a in range(6):
        
        ax=ps[a]
        ax.text(title_position[0],title_position[1],r"$m={}$".format(T[a]),
                family="serif",horizontalalignment='left',verticalalignment='top',transform = ax.transAxes,
                size=f_size)

    extent=[0.07,0.98,0.08,0.98] 
    f.subplots_adjust(hspace=0,wspace=0,left=extent[0],right=extent[1],bottom=extent[2],top=extent[3])        
            
    f.text(extent[1]-((extent[1]-extent[0])/2), 0.02, x_label, ha='center', va='center',size=f_size)
    f.text(0.02, extent[3]-((extent[3]-extent[2])/2), y_label, ha='center', va='center', rotation='vertical',size=f_size)
    

######################################################################################################
######################################################################################################

def histogram(cx,Nb,bin_extent,ps,full_hist,mass_limit,mass_range):
  
    table,full_table = load_data(cx=cx,cy="REDSHIFT_1",p_th=0.5,N_th=10,norm=False,p_values="d",mass_limit=mass_limit,mass_range=mass_range)
    bins,table = assign(table=table,Nb=20,th=0.5,bin_type="equal samples",redistribute=False,rd_th=0,ct_th=0)
    
    C=["purple","red","magenta","green","blue","orange"]
    
    bin_values=np.linspace(bin_extent[0],bin_extent[1],Nb+1)
    
    for a in range(6):
        
        ax=ps[a]
        
        t_select=table[bins[:,1] == a]
        
        if full_hist is True:
            ax.hist(table[:,-1],bins=bin_values,normed=True,histtype="step",linewidth=2,
                    color="0",alpha=1)
        
        ax.hist(t_select[:,-1],bins=bin_values,normed=True,histtype="step",linewidth=2,color=C[a])

    return None
  
######################################################################################################
######################################################################################################
  
def con(D,xlims,ylims,N):
    
    xedges=np.linspace(xlims[0],xlims[1],N+1)
    yedges=np.linspace(ylims[0],ylims[1],N+1)

    extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

    H,xi,yi=np.histogram2d(D[:,-2],D[:,-1], bins=(yedges, xedges),normed=True)
    H=scipy.ndimage.filters.gaussian_filter(input=H,sigma=2,order=0)
    
    H = H/np.max(H)
    
    return H,extent,xedges,yedges
  
def kde(data):
  
    D = data[:,-2:]
    
    [x,y] = D
    k = kde.gaussian_kde(D)
    
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    return xi,yi,zi

def line_f(x,a,b):
    return a*x+b

def contour(cx,cy,grid,f,ps,con_full,plot_line,xlims,ylims,f_size,levels,mass_limit,mass_range):
  
    table,full_table = load_data(cx=cx,cy=cy,p_th=0.5,N_th=10,norm=False,p_values="d",mass_limit=mass_limit,mass_range=mass_range)
    bins,table = assign(table=table,Nb=20,th=0.5,bin_type="equal samples",redistribute=False,rd_th=0,ct_th=0)
    
    C=["purple","red","magenta","green","blue","orange"]
    
    x_ex=np.array(xlims)
    
    if con_full == 1:
        H,extent,xedges,yedges=con(full_table,N=100,xlims=xlims,ylims=ylims)
        po,pc=curve_fit(line_f,full_table[:,-1],full_table[:,-2])

        for a in range(6):
            ax=ps[a]
            rng=np.array([xlims,ylims])
            #ax.contourf(H,extent=extent,alpha=0.5,cmap="Greys",levels=np.linspace(0,1000,100))
            ax.hist2d(full_table[:,-1], full_table[:,-2], bins=125,cmap="Greys",range=rng,norm=LogNorm())
            
            #H,extent,xedges,yedges=con(full_table,N=100,xlims=xlims,ylims=ylims)
            po,pc=curve_fit(line_f,full_table[:,-1],full_table[:,-2])

        for a in range(6):
            ax=ps[a]
            rng=np.array([xlims,ylims])

            #X, Y = np.meshgrid(xedges, yedges)
            #ax.pcolormesh(X, Y, H,cmap="Greys")
            #ax.set_aspect('auto')
                
            if plot_line == 1:
                ax.plot(x_ex,line_f(x_ex,po[0],po[1]),linewidth=2,color="black")
    
    for a in range(6):
            
        ax=ps[a]
        t_select=table[bins[:,1] == a]

        #H,extent,xedges,yedges=con(t_select,N=grid,xlims=xlims,ylims=ylims)
        
        H,xi,yi = kde(data=table)
        
        c = ax.contour(H,extent=extent,colors=C[a],linewidths=2,norm=True,levels=levels)
        #ax.clabel(c, inline=1, fontsize=10)
        
        if plot_line == 1:
            po,pc=curve_fit(line_f,t_select[:,-1],t_select[:,-2])
            ax.plot(x_ex,line_f(x_ex,po[0],po[1]),linewidth=2,color=C[a])
            
    # Add a colourbar        
    extent_c=[0.9, 0.08, 0.02, 0.9]
    
    f.subplots_adjust(right=extent_c[0]-0.01)
    plt.hist2d(full_table[:,-1], full_table[:,-2], bins=125,cmap="Greys",range=rng,norm=LogNorm())
    cbar_ax = f.add_axes(extent_c)  
    plt.colorbar(cax=cbar_ax) 
    
    f.text(0.98,(extent_c[1]+extent_c[3])-(extent_c[3]/2), "$N_{gal}$", 
           ha='center', va='center', rotation='vertical',size=f_size)

    for a in range(6):
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    
    return None
  
############################################################################################################
############################################################################################################
  
def plot_fractions(style,rd_th,ct_th,mass_limit,mass_range):

    C=["purple","red","magenta","green","blue","orange"]
    L=["1","2","3","4","5+","CT"]

    t,t_full=load_data(cx=["REDSHIFT_1"],cy=["PETROMAG_MR","PETROMAG_MZ"],p_th=0.5,N_th=10,norm=False,p_values="d",mass_limit=mass_limit,mass_range=mass_range)
    b,t=assign(table=t,Nb=20,th=0,bin_type="equal samples",redistribute=False,rd_th=rd_th,ct_th=ct_th)

    for a in range(6):
        d2=np.zeros((np.max(b[:,0]),2))
    
        for B in range(int(np.max(b[:,0]))): 
        
            d_s=t[b[:,0] == B]
        
            d2[B,:]=[np.mean(d_s[:,a]),np.mean(d_s[:,-1])]
            #d2[B,:]=[np.mean(d_s[:,a]/(1-d_s[:,5])),np.mean(d_s[:,-1])]
 
        plt.xlabel("Redshift")
        plt.ylabel(r"$<f _v>$")
        plt.plot(d2[:,1],d2[:,0],linewidth=2,color=C[a],linestyle=style,label=L[a])
        
    return None
  
def plot_data(cx,ps,Nb,bin_type,style,errors,data_type,mass_limit,mass_range):
  
    table,full_table = load_data(cx=cx,cy="REDSHIFT_1",p_th=0.5,N_th=10,norm=False,p_values=data_type,mass_limit=mass_limit,mass_range=mass_range)
    bins,table = assign(table=table,Nb=20,th=0.5,bin_type="equal samples",redistribute=False,rd_th=0,ct_th=0)
    
    C=["purple","red","magenta","green","blue","orange"]

    for a in range(6):
    
        fracs=get_fracs(table=table,bins=bins,a=a,Nb=Nb)
        ax=ps[a]
        ax.plot(fracs[:,0],fracs[:,1]/fracs[:,2],color=C[a],linestyle=style,linewidth=2)
        
        if errors == True:
            ax.fill_between(fracs[:,0],fracs[:,3],fracs[:,4],color=C[a],alpha=0.3)
    
    return None
  
def plot_individual(cx,Nb,bin_type,style,errors,data_type,mass_limit,mass_range,ax):
  
    table,full_table = load_data(cx=cx,cy="REDSHIFT_1",p_th=0.5,N_th=10,norm=False,p_values=data_type,mass_limit=mass_limit,mass_range=mass_range)
    bins,table = assign(table=table,Nb=20,th=0.5,bin_type=bin_type,redistribute=False,rd_th=0,ct_th=0)
    
    C=["purple","red","magenta","green","blue","orange"]

    for a in range(6):
    
        fracs=get_fracs(table=table,bins=bins,a=a,Nb=Nb)
        ax.plot(fracs[:,0],fracs[:,1]/fracs[:,2],color=C[a],linestyle=style,linewidth=2)
        
        if errors == True:
            ax.fill_between(fracs[:,0],fracs[:,3],fracs[:,4],color=C[a],alpha=0.3)
    
    return None
  
############################################################################################################
############################################################################################################

def make_6x1_plot(xlims,xticks,x_label,y_label,title_position,f_size,fig_size,N_yticks):

    f, ((ax1,ax2,ax3,ax4,ax5,ax6)) = plt.subplots(6,1, sharex=True, sharey=False,figsize=fig_size)    
    ps=[ax1,ax2,ax3,ax4,ax5,ax6]
    
    for pn in range(6):
        ax=ps[pn]
        ax.set_xlim(xlims)
        ax.set_xticks(xticks)
        #ax.set_yticks(MaxNLocator(nbins = 5,prune="upper"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins = N_yticks-1,prune="upper"))
        
    ax_edit_6x1(f=f,ps=ps,x_label=x_label,y_label=y_label,title_position=title_position,f_size=f_size)

    return f,ps
  
def make_6x1_legend(styles,labels,ax,f_size):
    
    for n in range(len(styles)):
        ax.plot([1000,1000],styles[n],label=labels[n])

    ax.legend(frameon=False,fontsize=f_size)
    
    return None
  
def ax_edit_6x1(f,ps,x_label,y_label,title_position,f_size):
    
    T=["1","2","3","4","5+","??"]
        
    for a in range(6):
        
        ax=ps[a]
        ax.text(title_position[0],title_position[1],r"$m={}$".format(T[a]),
                family="serif",horizontalalignment='left',verticalalignment='top',transform = ax.transAxes,
                size=f_size)

    extent=[0.13,0.98,0.08,0.98] 
    f.subplots_adjust(hspace=0,wspace=0,left=extent[0],right=extent[1],bottom=extent[2],top=extent[3])        
            
    f.text(extent[1]-((extent[1]-extent[0])/2), 0.02, x_label, ha='center', va='center',size=f_size)
    f.text(0.03, extent[3]-((extent[3]-extent[2])/2), y_label, ha='center', va='center', rotation='vertical',size=f_size)
    
##################################################################################################
##################################################################################################
    
def make_1x2_plot(xlims,ylims,xticks,yticks,xlabel,ylabel,f_size,fig_size):
    
    f,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=fig_size)
    ps = [ax1,ax2]
    
    for pn in range(2):
        
        ax=ps[pn]
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
    ax_edit_1x2(f=f,ps=ps,x_label=xlabel,y_label=ylabel,f_size=f_size)
        
    return f,ps

def ax_edit_1x2(f,ps,x_label,y_label,f_size):
        
    extent=[0.07,0.98,0.08,0.98] 
    f.subplots_adjust(hspace=0,wspace=0,left=extent[0],right=extent[1],bottom=extent[2],top=extent[3])        
            
    f.text(extent[1]-((extent[1]-extent[0])/2), 0.02, x_label, ha='center', va='center',size=f_size)
    f.text(0.02, extent[3]-((extent[3]-extent[2])/2), y_label, ha='center', va='center', rotation='vertical',size=f_size)
    
def contour_1x2(cx,cy,grid,f,ps,xlims,ylims,f_size,a_vals,mass_limit,mass_range):
  
    table,full_table = load_data(cx=cx,cy=cy,p_th=0.5,N_th=10,norm=False,p_values="d",mass_limit=mass_limit,mass_range=mass_range)
    bins,table = assign(table=table,Nb=20,th=0.5,bin_type="equal samples",redistribute=False,rd_th=0,ct_th=0)
    
    C=["purple","red","magenta","green","blue","orange"]
    
    for pn in range(2):

        ax = ps[pn]

        for a in a_vals:
  
            rng=np.array([xlims,ylims])

            t_select=table[bins[:,1] == a]

            H,extent,xedges,yedges=con(t_select,N=grid,xlims=xlims,ylims=ylims)
            ax.contour(H,extent=extent,colors=C[a],linewidths=2)
    
    return None
    
    
