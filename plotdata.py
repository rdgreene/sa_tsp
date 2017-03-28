#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:05:31 2017

@author: miguelesteras
"""

def plotline(mean_costs_matrix,variable,n,baseline,title):
    
    import matplotlib.gridspec as gridspec

    window_ave = np.zeros_like(mean_costs_matrix)
    for k in range(0,int(size(mean_costs_matrix,1))):
        for i in range(1,int(size(mean_costs_matrix[:,k])+1)):
            if i<n-1:
                window_ave[i-1,k] = (np.mean(mean_costs_matrix[:,k][0:i]))
            else:
                window_ave[i-1,k] = (np.mean(mean_costs_matrix[:,k][i-(n-1):i]))
        
    
    axis_font = {'color':  'black',         # define font for axis labels
            'weight': 500,
            'size': 14 }
    title_font = {'color':  'black',        # define font for tittle
            'weight': 500,
            'size': 16 }
    legend_font = {'color':  'black',        # define font for legend
            'weight': 500,
            'size': 12 }
    
    gs = gridspec.GridSpec(6, 3)            # set up graph size
    gs.update(hspace=0.3)                   # gap between graphs
    ax = plt.subplot(gs[:-1, :])            # size top graph
    ax_base = plt.subplot(gs[-1,:])         # size botton graph
    ax_base.plot(np.zeros_like(window_ave[:,1])+baseline,       # plot optimum performance
                               c='tomato', label='baseline')   
    for i in range(0,int(size(variable))):                      # plot learning data
        ax.plot(window_ave[:,i], label=str(variable[i]))
    
    # limit the view of the graphs
    ax.set_ylim(amin(window_ave)-1, amax(window_ave))
    ax_base.set_ylim(baseline-0.5, baseline+0.5)
    
    # hide the spines between ax and ax_base
    ax.spines['bottom'].set_visible(False)
    ax_base.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')
    ax_base.xaxis.tick_bottom()
    ax_base.yaxis.set_ticks(np.arange(baseline-1, baseline+1, 1))
    
    # add diagonal line in y-axis to separete ax and ax_base
    d = .015                                        # size of diagonal lines
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)           # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)     # top-right diagonal
    kwargs.update(transform=ax_base.transAxes)      # switch to the bottom axes
    ax_base.plot((-d, +d), (1 - d, 1 + d), **kwargs)# bottom-left diagonal
    ax_base.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    # Add a grid, axes labels and tittle
    ax.grid(b=True, which='major', color='lightgray', linestyle='-')    # grid on
    ax_base.set_xlabel('Epochs', fontdict=axis_font)                    # x-label
    ax.set_ylabel('Cost', fontdict=axis_font)                           # y-label
    ax.set_title(title, y=1.1, fontdict=title_font) # title
    
    # Add a legend
    legend = ax.legend(loc='upper right', shadow=False,fontsize= 10)
    legend.get_frame().set_facecolor('white')   # legend background
    legend.get_frame().set_edgecolor('None')    # legend edge color
    for text in legend.get_texts():             # text in legend
        plt.setp(text)
    plt.show()