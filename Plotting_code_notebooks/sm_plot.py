#Plotting SFH colour lines.

import params
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load a set of values: ----------------------------------------------------
values = pickle.load(open(params.sf_dir + "values.p","rb"),encoding='latin1')
mags_all = np.load(params.sf_dir + params.sf_mags)
axes = ['filter', 'redshift', 'tg', 'Av', 'tau', 'metallicity']
axes = dict(zip(axes, range(len(axes))))

#----------------------------------------------------------------------------
def take_nearest(axisname, val, array, values=values):
  
    # Tool for selecting nearest value in your array.
    # Axis name: name of column to match.
    # val: find nearest value to val.
    # array: input array
    idx = (np.abs(values[axisname] - val)).argmin()
    #print('Found {} = {}'.format(axisname, values[axisname][idx]))
    return array.take([idx], axis=axes[axisname])
  
#-----------------------------------------------------------------------------
#get magnitude
mags_all = mags_all
# restframe (redshift = 0)
mags_all = take_nearest('redshift', 0.0, mags_all)
# solar metallicity
mags_all = take_nearest('metallicity', 0.02, mags_all)

colours = ["red","green","blue","yellow","orange","magenta","purple","black"]
  
#----------------------------------------------------------------------------
def age_line(x, y, val, color, label=r'$\tau$', lw=3):
  
  # x,y: x,y values eg. u-r,r-z
  # color: line colour to plot
  
    plt.plot(x, y, '-', color=color, lw=lw,
             label=r'{} = {}'.format(label, val),
             zorder=1)
    plt.scatter(x, y, c=values['tg'], s=50, zorder=2)
    
    return None


#-----------------------------------------------------------------------------
def modify_string_name(s):
  
    if s == "u":
        s = b"u"
    elif s == "g":
        s = b"g"
    elif s == "r":
        s = b"r"
    elif s == "i":
        s = b"i"
    elif s == "z":
        s = b"z"
        
    return s
  
def string_to_var(s,u,g,r,i,z):
    
    if s == "u":
        n = u
    elif s == "g":
        n = g
    elif s == "r":
        n = r
    elif s == "i":
        n = i 
    elif s == "z":
        n = z
        
    return n 
#-----------------------------------------------------------------------------
def color_color_age_line_par(par,val,mags,colour1,colour2,linecolor='k',label=r'$\tau$',
    values=values):
  
    colxb = modify_string_name(colour1[0])
    colxr = modify_string_name(colour1[1])
    colyb = modify_string_name(colour2[0])
    colyr = modify_string_name(colour2[1])
    
    mags = take_nearest(par, val, mags)
    
    #print(mags)
    f = values['filter']
    colx = mags[f == colxb] - mags[f == colxr]
    coly = mags[f == colyb] - mags[f == colyr]
    colx = colx.squeeze()
    coly = coly.squeeze()
    age_line(colx, coly, val, linecolor, label) 
    
#-----------------------------------------------------------------------------

def dust_arrow_plot(val,mags,colour1,colour2):
  
    mags_Av0 = take_nearest('Av', 0.0, mags)
    mags_Av1 = take_nearest('Av', val, mags)
    dmag = mags_Av1 - mags_Av0
    dmag = dmag.squeeze()
    # get rid of nans
    dmag = dmag[:, ~np.isnan(dmag[0])]
    u, g, r, i, z = dmag.mean(axis=1)
    grAv1mag = (string_to_var(s=colour1[0],u=u,g=g,r=r,i=i,z=z) 
        - string_to_var(s=colour1[1],u=u,g=g,r=r,i=i,z=z))
    riAv1mag = (string_to_var(s=colour2[0],u=u,g=g,r=r,i=i,z=z) 
        - string_to_var(s=colour2[1],u=u,g=g,r=r,i=i,z=z))
    
    xy0 = (0.35, 0.25)
    xy1 = (0.35 + grAv1mag, 0.25 + riAv1mag)
    
    arrow = plt.annotate("", xy=xy1, xytext=xy0,
        arrowprops=dict(frac=0.1, width=2))
    plt.text(xy1[0], xy1[1], '$A_v = {{{}}}$ mag'.format(str(val)), ha='left',# va='bottom',
        fontsize='small')
    
    return None

#-----------------------------------------------------------------------------
def plot_line(vary,taus,Avs,dust_arrow,colour1,colour2):

    c = -1
        
    if vary == "Av":
        mags = take_nearest("tau",taus[0],mags_all)
        label = "$A_v$"
        
        for v in Avs:
            c=c+1
            color_color_age_line_par("Av",Avs[c], mags
                ,colour1=colour1,colour2=colour2,linecolor=colours[c],label=label)
            
    else:
        mags = take_nearest("Av",Avs[0],mags_all)
        label = r"$\tau$"
        
        if vary != "tau":
            print("Invalid 'vary' value; using tau")
            
        for v in taus:
            c=c+1
            color_color_age_line_par("tau",taus[c], mags
                ,colour1=colour1,colour2=colour2,linecolor=colours[c],label=label)
            
    cb = plt.colorbar()
    cb.ax.set_ylabel('age [Gyr]')
    plt.legend(loc = 'upper left', fontsize='small')
            
    if dust_arrow == True:
        dust_arrow_plot(1,mags=mags_all,colour1=colour1,colour2=colour2)
            
    return None
            
    
            
        