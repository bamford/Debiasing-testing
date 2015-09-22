#github.com/rjsmethurst/prefig
"""
    An awesome plotting object to make any plot poster or presentation ready in the colour of your choice!
    
    R. J. Smethurst 
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter

class Prefig(plt.Figure):
    """
        A class that can replace the 'plt.figure' python plotting command to create poster and presentaiton ready plots instantly. Plots will be initialised with a transparent background. The font colour and axes colour must be specified - e.g. 'white' for a black background poster. Colours are inverted if 'white' is specified as the colour. Size of fonts changed appropriately according to figure size specified. 
        
        :axcol:
            The colour of the axes of the plot (optional). Default is 'black' or 'k'.
        
        :fontcol:
            The colour of the axes labels and tick labels of the plot (optional). Default is 'black' or 'k'.
        
        :figsize:
            tuple of integers (optional). (width, height) in inches. Default is (16,12).
        :font: 
            font family (optional). Default from rc.font file. Can be specified within plot commands instead. 
        
        """
    def __init__(self, axcol='k', fontcol='k', size=(16,12), font='serif'):
        self.axcol = axcol
        self.fontcol = fontcol
        self.size = size
        self.font = font
        
        if axcol == 'w':
            col = colorConverter.to_rgba_array(plt.rcParams['axes.color_cycle'])[:,:-1]
            inv_col = (256-(256*col))/256.0
            plt.rc('axes', color_cycle = list(inv_col))
        
        plt.rc('figure', figsize=size, facecolor='w', edgecolor='none')
        plt.rc('savefig', dpi=300, facecolor='w', edgecolor='none', frameon='False')
        f = {'family':font, 'size':20}
        plt.rc('font', **f)
        plt.rc('text', color=fontcol)
        plt.rc('axes', labelsize='x-large', edgecolor=axcol, labelcolor=fontcol, facecolor='none', linewidth=2)
        plt.rc('xtick', labelsize='18', color=fontcol)
        plt.rc('ytick', labelsize='18', color=fontcol)
        plt.rc('lines', markersize=8, linewidth=2)


        




