# Put all of the 'shared' plotting functions in here.

from  matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.ndimage
from astropy.io import fits
import numpy as np
import scipy.stats.distributions as dist
import math
import load_data

from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter

c = 0.683

titles = ["1","2","3","4","5+","??"]
#colours = ["purple","red","magenta","green","blue","orange"]
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
    axarr[2].set_ylabel(y_label)

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
        equal_samples=True,redistribute=False,rd_th=0,ct_th=0
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
                ,color=colours[m],alpha=0.3,linestyle="dashed",hatch="/")
    
    return None
  
######################################################################################################
######################################################################################################

def contour(table,xlims,ylims,grid_spacing,contour_type,ax,alpha,colour,levels,sigma):
    
    # Define the histogram range.
    
    if sigma == True:
        xi,yi,H,V=sigma_hist(x=table[:,-1],y=table[:,-2],bins=grid_spacing,
            levels=levels,range=[[xlims[0], xlims[1]], [ylims[0], ylims[1]]])

        H=scipy.ndimage.filters.gaussian_filter(input=H,sigma=1.5,order=0) # Gaussian smoothing.
    else:
        xedges=np.linspace(xlims[0],xlims[1],grid_spacing+1)
        yedges=np.linspace(ylims[0],ylims[1],grid_spacing+1) 
        
        H,Y,X=np.histogram2d(x=table[:,-2],y=table[:,-1], bins=(yedges, xedges)) # 2D histogram values
        H=H/np.sum(H)
        xi,yi = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
        V=levels
        
        H=scipy.ndimage.filters.gaussian_filter(input=H,sigma=5,order=0) # Gaussian smoothing.
    
    if contour_type == "contour":
        ax.contour(xi,yi,H,V,colors=colour[0])
    elif contour_type == "contourf":
        ax.contourf(xi,yi,H,levels=np.concatenate([V.T,[np.max(H)]]),cmap=colour[1],alpha=alpha)
    elif contour_type == "contourf+line":
        ax.contourf(xi,yi,H,levels=np.concatenate([V.T,[np.max(H)]]),cmap=colour[1],alpha=alpha)
        ax.contour(xi,yi,H,levels=np.concatenate([V.T,[np.max(H)]]),colors=colour[0],linewidths=1)
    else:
        ax.hist2d(table[:,-1],table[:,-2],bins=[xedges,yedges],cmap=colour[1],alpha=alpha)

    return None
  

def contour_plots(cx,cy,grid_spacing,axarr,xlims,ylims,reference_plot,
    m_plot,reference,levels,alphas,sigma):
  
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
    
    #table[:,-2:] = np.random.randn(len(table),2)
    #full_table[:,-2:] = np.random.randn(len(full_table),2)
    
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
            ,contour_type=reference_plot,ax=axarr[m],alpha=alphas[0],colour=["k","Greys"],levels=levels
            ,sigma=sigma)
	    
        if m_plot != None:
        
            contour(table=t_sel,xlims=xlims,ylims=ylims,grid_spacing=grid_spacing
                ,contour_type=m_plot,ax=axarr[m],alpha=alphas[1],colour=[colours[m],cmaps[m]],levels=levels
                ,sigma=sigma)

    axarr[0].set_xlim(xlims)
    axarr[0].set_ylim(ylims)
        
    return None


def xy_plot(cx,cy,Nb,equal_samples,reference,style,axarr,standard_dev):
    '''
    Plot binned data ---------------------------------------------------------
    --------------------------------------------------------------------------
    Arguments:

    cx,cy: x+y axis columns for contour (refer to the same cx+cy values in 
    load_data.load).
    
    Nb: Number of bins to plot.
    
    equal_samples: if set as True, all bins will have the same number of
    galaxies.
    
    reference: Can be set as 'all galaxies', 'all spirals' or None.
    
    style: linestyle to plot eg. 'dashed' or 'solid'
    
    reference_plot,m_plot: Plot style. Can have the following values:
    
    axarr: plot array as from make_grid or make_stack.
    
    standard_dev: if True, the standard deviation is plotted as a filled 
    contour.
    --------------------------------------------------------------------------
    '''
  
    table,full_table = load_data.load(cx=cx,cy=cy,p_th=0.5,N_th=10,norm=False,p_values="d")
    
    bins,table = load_data.assign(table=table,Nb=Nb,th=0.5,equal_samples=equal_samples,
        redistribute=False,rd_th=0,ct_th=0,print_sizes=False)
    full_bins,full_table = load_data.assign(table=full_table,Nb=Nb,th=0.5,equal_samples=equal_samples,
        redistribute=False,rd_th=0,ct_th=0,print_sizes=False)
    
    if reference == "all galaxies":
        reference_table,reference_bins = full_table,full_bins
        
    else:
        reference_table,reference_bins = table,bins
        
        if (reference != "all spirals") and (reference != None):
            print("Invalid 'reference' value; using 'all spirals'")
        
    xy_r = load_data.get_xy_binned(reference_table,reference_bins)
    
    for m in range(5):
      
        t_sel = table[bins[:,1] == m]
        b_sel = bins[bins[:,1] == m]
        
        xy = load_data.get_xy_binned(t_sel,b_sel)
        
        if reference != None:
            axarr[m].plot(xy_r[:,0],xy_r[:,2],color="k",linewidth=2,linestyle=style)
            if standard_dev == True:
                axarr[m].fill_between(xy_r[:,0],xy_r[:,2]+xy_r[:,3],xy_r[:,2]-xy_r[:,3],color="k",alpha=0.5)

        axarr[m].plot(xy[:,0],xy[:,2],color=colours[m],linewidth=2,linestyle=style)
        if standard_dev == True:
            axarr[m].fill_between(xy[:,0],xy[:,2]+xy[:,3],xy[:,2]-xy[:,3],color=colours[m],alpha=0.5)

    return None
  
def line_fit(cx,cy,x_range,curve,style,reference,reference_plot,m_plot,axarr):
  
    x_guide = np.linspace(x_range[0],x_range[1],100)
  
    table,full_table = load_data.load(cx=cx,cy=cy,p_th=0.5,N_th=10,norm=False,p_values="d")
    bins,table = load_data.assign(table=table,Nb=20,th=0.5,equal_samples=True,
        redistribute=False,rd_th=0,ct_th=0,print_sizes=False)
    
    if reference == "all galaxies":
        reference_table = full_table
    else:
        reference_table = table
  
    if curve == True:  
        def f(x,k,c1,c2):
            return k**(-x + c1) + c2
    else:
        def f(x,k,c1,c2):
            return k*x + c1 
    
    if reference_plot == True:
        p_r,c_r = curve_fit(f,reference_table[:,-1],reference_table[:,-2],maxfev=10000)
        
    for m in range(5):
        
        if reference_plot == True:
            axarr[m].plot(x_guide,f(x_guide,p_r[0],p_r[1],p_r[2]),color="k",linewidth=2,linestyle=style)
            
        if m_plot == True:
            t_m = table[bins[:,1] == m]
            p,c = curve_fit(f,t_m[:,-1],t_m[:,-2],maxfev=10000)
            axarr[m].plot(x_guide,f(x_guide,p[0],p[1],p[2]),color=colours[m],linewidth=2,linestyle=style)
            
    return None


def sigma_hist(x,y,bins,levels,range):
  
    if range == None:
        range = [[x.min(), x.max()], [y.min(), y.max()]]
  
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=range)
    
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])
    
    H2 = H2.T
    
    return X2,Y2,H2,V
  
  
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