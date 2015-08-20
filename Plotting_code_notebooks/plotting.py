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
import load_data
c = 0.683

titles = ["1","2","3","4","5+","??"]
colours = ["purple","red","magenta","green","blue","orange"]
cmaps=["YlOrBr","Reds","BuPu","Greens","Blues"]
colours=["orange","red","purple","green","blue"]


##############################################################################
##############################################################################

def make_grid(title,title_position,x_limits,y_limits,x_label,y_label,x_ticks
        ,y_ticks,figsize):
    '''
    Makes the grid plot-------------------------------------------------------
    --------------------------------------------------------------------------
    Arguments:

    title: if True, a titles are displayed in each subplot.
    
    title_position: should be a 3 item list of the form: 
    [x position, y position, font size]
    
    x_limits,ylimits: 2 item lists with [lower bound, upper bound].
    
    x,label,ylabel: strings with the appropriate labels.
    
    x_ticks,y_ticks: should be 3 item lists of the form:
    [lower limit, upper limit, spacing]
    
    figsize: size of the returned plot (list of [width,height]).
    --------------------------------------------------------------------------
    Returns:
    
    f: the figure.
    axarr: the subplots (ravelled to a 5x1 array, rather than  2x3).
    --------------------------------------------------------------------------
    '''

    # Make the axes:
    f,axarr=plt.subplots(2,3,sharex=True,sharey=True
        ,figsize=(figsize[0],figsize[1]))
    axarr = np.ravel(axarr)
    f.subplots_adjust(hspace=0,wspace=0)
    f.delaxes(axarr[-1])
    
    # Set the labels, limits, etc.
    axarr[3].set_xlabel(x_label)
    axarr[4].set_xlabel(x_label)
    axarr[0].set_ylabel(y_label)
    axarr[3].set_ylabel(y_label)

    axarr[0].set_xlim(x_limits)
    axarr[0].set_ylim(y_limits)

    axarr[0].set_xticks(np.arange(x_ticks[0],x_ticks[1]+10**(-5),x_ticks[2]))
    axarr[0].set_yticks(np.arange(y_ticks[0],y_ticks[1]+10**(-5),y_ticks[2]))
    
    # Put titles on the axes if titles is true.
    if title == True:
        for m in range(5):
            axarr[0].text(title_position[0],title_position[1]
	        ,r"$m={}$".format(titles[m]),family="serif"
	        ,horizontalalignment='left',verticalalignment='top'
                ,transform = axarr[m].transAxes,size=title_position[2])

    return f,axarr
  
  
def make_stack(title,title_position,x_limits,x_label,y_label,x_ticks,figsize):
    '''
    Makes the stacked plot----------------------------------------------------
    --------------------------------------------------------------------------
    Arguments:

    title: if True, a titles are displayed in each subplot.
    
    title_position: should be a 3 item list of the form: 
    [x position, y position, font size]
    
    x_limits: 2 item list with [lower bound, upper bound].
    
    x,label,ylabel: strings with the appropriate labels.
    
    x_ticks: should be a 3 item list of the form:
    [lower limit, upper limit, spacing]
    
    figsize: size of the returned plot (list of [width,height]).
    --------------------------------------------------------------------------
    Returns:
    
    f: the figure.
    axarr: the subplots (ravelled to a 5x1 array, rather than  2x3).
    --------------------------------------------------------------------------
    '''
    
    # Make the axes:
    f,axarr = plt.subplots(5,1,sharex=True,sharey=False
        ,figsize=(figsize[0],figsize[1]))
    axarr = np.ravel(axarr)
    f.subplots_adjust(hspace=0,wspace=0)

    # Set the labels, limits, etc.
    axarr[4].set_xlabel(x_label)
    axarr[4].set_xlabel(x_label)
    axarr[3].set_ylabel(y_label)

    axarr[0].set_xlim(x_limits)

    axarr[0].set_xticks(np.arange(x_ticks[0],x_ticks[1]+10**(-5),x_ticks[2]))
    
    if title == True:
        for m in range(5):
            axarr[0].text(title_position[0],title_position[1]
	        ,r"$m={}$".format(titles[m]),family="serif"
	        ,horizontalalignment='left',verticalalignment='top'
                ,transform = axarr[m].transAxes,size=title_position[2])
    
    return f,axarr

##############################################################################
##############################################################################

def histogram(cx,Nb,bin_extent,axarr,full_hist,style):
    '''
    Plot histograms-----------------------------------------------------------
    --------------------------------------------------------------------------
    Arguments:

    cx: The data you want histogrammed (ie. a colour, mass etc.)
    
    Nb: Number of bins.
    
    bin_extent: List of form [lower bound,upper bound]
    
    axarr: the plotting array as from make_grid or make_stack.
    
    full_hist: set as "all" for all galaxies, "all spirals" for all spiral 
    galaxies, or "assigned spirals" for all spirals that meet the threshold to
    be classified as having a particular arm number.
    
    style: histogram line style eg. "solid"
    --------------------------------------------------------------------------
    '''
  
    # Load all of the data and assign arms to each:
    table,full_table = load_data.load(cx=cx,cy=["REDSHIFT_1"]
        ,p_th=0.5,N_th=5,norm=False,p_values="d")
    bins,table = load_data.assign(table=table,Nb=20,th=0.5,
        bin_type="equal samples",redistribute=False,rd_th=0,ct_th=0
        ,print_sizes=False)
    # Define histogram bins:
    
    bin_values=np.linspace(bin_extent[0],bin_extent[1],Nb+1)
    
    for m in range(5):    
        t_select=table[bins[:,1] == m]
        # Reference histograms:
        if full_hist == "all":
            axarr[m].hist(table[:,-1],bins=bin_values,normed=True
	        ,histtype="step",linewidth=2, color="k",alpha=0.75)
	    
        elif full_hist == "all spirals":
            axarr[m].hist(table[:,-1],bins=bin_values,normed=True
	        ,histtype="step",linewidth=2, color="k",alpha=0.75)
	    
        else:
            sel = table[(bins[:,1] != -999) & (bins[:,1] != 5)]
            axarr[m].hist(sel[:,-1],bins=bin_values,normed=True
                ,histtype="step",linewidth=2, color="k",alpha=0.75)
	    
            if full_hist != "assigned spirals":
                print("Invalid full_hist value; using 'assigned spirals'")
	  
        # Plot histograms.
        axarr[m].hist(t_select[:,-1],bins=bin_values,normed=True
            ,histtype="step",linewidth=2,color=colours[m],linestyle=style)

    return None
  

def plot_data(cx,axarr,Nb,equal_samples,style,errors,data_type):
    '''
    Plot data-----------------------------------------------------------------
    --------------------------------------------------------------------------
    Arguments:

    cx: The data you want plotted (ie. a colour, mass etc.)
    
    axarr: the plotting array as from make_grid or make_stack.
    
    Nb: Number of bins.
    
    equal samples: if True, bin into equally sized samples. If False, bin 
    into equally spaced bins.
    
    style: linestyle eg. "dotted" or "solid"
    
    errors: if True, plot errors from Cameron et al.
    
    data_type: have either "d" for debiased, "r" for raw, or "w" for debiased
    from Willet et al.
    --------------------------------------------------------------------------
    '''
  
    table,full_table = load_data.load(cx=cx,cy=["REDSHIFT_1"],p_th=0.5,
        N_th=10,norm=False,p_values=data_type)
    bins,table = load_data.assign(table=table,Nb=Nb,th=0.5,
        equal_samples=equal_samples,redistribute=False,rd_th=0,ct_th=0,
        print_sizes=False)

    for m in range(5):
    
        fracs=load_data.get_fracs(table=table,bins=bins,m=m,c=0.683
            ,full_data="assigned spirals")    
        axarr[m].plot(fracs[:,0],fracs[:,1]/fracs[:,2],color=colours[m],
            linestyle=style,linewidth=2)
        
        if errors == True:
            axarr[m].fill_between(fracs[:,0],fracs[:,3],fracs[:,4]
                ,color=colours[m],alpha=0.3)
    
    return None
  
######################################################################################################
######################################################################################################

def contour(table,xlims,ylims,grid_spacing,contour_type,ax,alpha,colour,levels):
    
    xedges=np.linspace(xlims[0],xlims[1],grid_spacing+1)
    yedges=np.linspace(ylims[0],ylims[1],grid_spacing+1) 
    extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]] # Define the histogram range.

    H,xi,yi=np.histogram2d(table[:,-2],table[:,-1], bins=(yedges, xedges),normed=True) # 2D histogram values
    H=scipy.ndimage.filters.gaussian_filter(input=H,sigma=5,order=0) # Gaussian smoothing.
    
    if contour_type == "contour":
        ax.contour(H,extent=extent,colors=colour[0],alpha=alpha,levels=levels,linewidths=2) 
        
    elif contour_type == "contourf":
        ax.contourf(H,extent=extent,cmap=colour[1],alpha=alpha,levels=np.concatenate([levels.T,[np.max(H)]]))
    
    elif contour_type == "contourf+line":
        ax.contourf(H,extent=extent,cmap=colour[1],alpha=alpha,levels=np.concatenate([levels.T,[np.max(H)]])) 
        ax.contour(H,extent=extent,colors=colour[0],alpha=1,levels=levels,linewidths=1) 
        
    else:
        ax.hist2d(table[:,-1],table[:,-2],bins=[xedges,yedges],cmap=colour[1],alpha=alpha)

    return None

def contour_plots(cx,cy,grid_spacing,axarr,xlims,ylims,reference_plot,
    m_plot,reference,levels,alphas):
  
    '''
    Plot contours ------------------------------------------------------------
    --------------------------------------------------------------------------
    Arguments:

    cx,cy: x+y axis columns for contour (refer to the same cx+cy values in 
    load_data.load).
    
    grid_spacing: Number of cells to divide in to for the 2d histogram.
    
    axarr: the plotting array as from make_grid or make_stack.
    
    xlims,ylims: Lists of the form [lower bound, upper bound] for the
    x+y histogram limits.
    
    reference_plot,m_plot: Plot style. Can have the following values:
    
    -------
    "contour": pure line contour.
    "contourf": pure filled contour.
    "contourf+line": lines+filled contour.
    "hist": 2D shaded histogram.
     -------
   
    reference: Can set as "all galaxies" to include all of the volume-limited
    sample or "all spirals" to only show the spiral sample.
    
    levels: Set the contour levels manually here.
    
    alphas: 2 item list with [reference transparency, arm sample transparency]
    values.
    --------------------------------------------------------------------------
    '''
    
    table,full_table = load_data.load(cx=cx,cy=cy,p_th=0.5,N_th=10,norm=False,p_values="d")
    bins,table = load_data.assign(table=table,Nb=20,th=0.5,equal_samples=True,
        redistribute=False,rd_th=0,ct_th=0,print_sizes=False)
    
    if reference == "all galaxies":
        reference_table = full_table
        
    elif reference == "all spirals":
        reference_table = table
    
    for m in range(5):
        
        t_sel = table[bins[:,1] == m]
        
        if (reference_plot is not None) & ((reference == "all galaxies") or (reference == "all spirals")): 
            contour(table=reference_table,xlims=xlims,ylims=ylims,grid_spacing=grid_spacing
            ,contour_type=reference_plot,ax=axarr[m],alpha=alphas[0],colour=["k","Greys"],levels=levels)
        
        contour(table=t_sel,xlims=xlims,ylims=ylims,grid_spacing=grid_spacing
            ,contour_type=m_plot,ax=axarr[m],alpha=alphas[1],colour=[colours[m],cmaps[m]],levels=levels)
        
    return None
  

# CBAR STUFF
            
    # Add a colourbar        
    #extent_c=[0.9, 0.08, 0.02, 0.9]
    
    #f.subplots_adjust(right=extent_c[0]-0.01)
    #plt.hist2d(full_table[:,-1], full_table[:,-2], bins=125,cmap="Greys",range=rng,norm=LogNorm())
    #cbar_ax = f.add_axes(extent_c)  
    #plt.colorbar(cax=cbar_ax) 
    
   # f.text(0.98,(extent_c[1]+extent_c[3])-(extent_c[3]/2), "$N_{gal}$", 
           #ha='center', va='center', rotation='vertical',size=f_size)

   # for a in range(6):
        #ax.set_xlim(xlims)
       # ax.set_ylim(ylims)
    
   # return None