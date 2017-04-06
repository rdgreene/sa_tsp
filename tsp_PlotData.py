# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:05:31 2017

@author: miguelesteras
"""

# Import functions and dependencies
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import collections
import numpy as np
import matplotlib.pyplot as plt
from tspFunctions import loadCoordenates

def heatmap(grid,a,b):
      
    # define font style for axes and title
    axis_font = {'color':  'black',         
                    'weight': 500,
                    'size': 14 }
    title_font = {'color':  'black',        
                'weight': 500,
                'size': 16 }    
    
    plt.imshow(grid, interpolation='lanczos', cmap='plasma')
    plt.title('Grid search of '+ a +' vs '+ b, y=1.05,fontdict=title_font)
    plt.ylabel(a, fontdict=axis_font)
    plt.xlabel(b, fontdict=axis_font)
    
    plt.show()




def plotFewRoutes(seqs,file_xy,variable):

    axis_font   = {'color':  'black',        # define font for axis labels
                   'weight': 500,
                   'size': 14 }
    title_font  = {'color':  'black',        # define font for tittle
                   'weight': 500,
                   'size': 16 }
    nodes_font  = {'family': 'fantasy',      # define font for nodes labels
                   'color':  'black',
                   'weight': 400,
                   'size': 14 }

    for i in range(0,np.size(variable)):
        
        # routes examples per value in variable
        n = int(np.size(seqs,0)/np.size(variable))
        # transform sequeces of states to an array
        d = collections.OrderedDict()
        for a in np.asarray(seqs)[i*n:(i*n)+n,:]:
            t = tuple(a)
            if t in d:
                d[t] += 1
            else:
                d[t] = 1
        
        transition_summary = []
        for (key, value) in d.items():
            transition_summary.append(list(key) + [value])
        
        paths = np.asarray(transition_summary).T
        idx = np.argsort(paths[-1,:])       # idx = index of ordered (min->max) path iterations 
        paths = paths[:,np.flipud(idx)]     # order paths according to iterations (max->min)  
        
        # load coordenates for nodes/locations
        coordenates = loadCoordenates(file_xy)   
        
        # Graph configurations
        fig, mapa = plt.subplots()
        mapa.patch.set_facecolor('whitesmoke')
        plt.grid(b=False)
        plt.title(str(variable), y=1.01,fontdict=title_font)
        plt.ylabel('latitude', fontdict=axis_font)
        plt.xlabel('longitude', fontdict=axis_font)
        plt.xticks([], [])
        plt.yticks([], [])
                
        # plot preferred path
        path = np.asarray([coordenates[x,:] for x in paths[0:-1,0]])
        plt.plot(path[:,0],path[:,1], linestyle = '-', c='orange', 
                 linewidth=8, alpha=0.9, zorder=2, label='best path ' + np.array_str(paths[0:-1,0]))
        # plot other paths
        for k in range(1,np.size(paths,1)):                         
            path = np.asarray([coordenates[x,:] for x in paths[0:-1,k]])
            width = 40*(paths[-1,k]/max(paths[-1,:]))
            plt.plot(path[:,0],path[:,1], linestyle = '-', c='gray', 
                     linewidth=width, alpha=0.9, zorder=2, label ='other path ' + np.array_str(paths[0:-1,k]))
        
       # Add a legend
        legend = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', shadow=False,fontsize= 10)
        legend.get_frame().set_facecolor('white')   # legend background
        legend.get_frame().set_edgecolor('black')    # legend edge color
        legtext = legend.get_texts()
        for text in legtext[0:-4]:             # text in legend
            plt.setp(text)
        
        # plot states/cities
        mapa.scatter(coordenates[0:-1,0],coordenates[0:-1,1],
                     c='white', s=800, 
                     label='white',alpha=1, 
                     edgecolors='black', zorder=3)
        
        # plot names/numbers of states
        for j in range(0,np.size(coordenates,0)-1):                        
            plt.text(coordenates[j,0], coordenates[j,1], str(j),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontdict=nodes_font)
        
        plt.show()                      
    



def plotManyRoutes(seqs,file_xy,variable):

    axis_font   = {'color':  'black',        # define font for axis labels
                   'weight': 500,
                   'size': 18 }
    title_font  = {'color':  'black',        # define font for tittle
                   'weight': 500,
                   'size': 22 }
    nodes_font  = {'family': 'fantasy',      # define font for nodes labels
                   'color':  'black',
                   'weight': 400,
                   'size': 12 }

    for i in range(0,np.size(variable)):
        
        # routes examples per value in variable
        n = int(np.size(seqs,0)/np.size(variable))
        # transform sequeces of states to an array
        d = collections.OrderedDict()
        for a in np.asarray(seqs)[i*n:(i*n)+n,:]:
            t = tuple(a)
            if t in d:
                d[t] += 1
            else:
                d[t] = 1
        
        transition_summary = []
        for (key, value) in d.items():
            transition_summary.append(list(key) + [value])
        
        paths = np.asarray(transition_summary).T
        idx = np.argsort(paths[-1,:])       # idx = index of ordered (min->max) path iterations 
        paths = paths[:,np.flipud(idx)]     # order paths according to iterations (max->min)  
        
        # load coordenates for nodes/locations
        coordenates = loadCoordenates(file_xy)   
        
        # Graph configurations
        plt.figure(figsize=(17, 13), facecolor='whitesmoke')
        plt.grid(b=False)
        plt.title(str(variable[i]), y=1.05,fontdict=title_font)
        plt.ylabel('latitude', fontdict=axis_font)
        plt.xlabel('longitude', fontdict=axis_font)
        plt.xticks([], [])
        plt.yticks([], [])
                
       # plot second most common paths
        path = np.asarray([coordenates[x,:] for x in paths[0:-1,1]])
        width = 10*(paths[-1,1]/max(paths[-1,:]))
        plt.plot(path[:,0],path[:,1], linestyle = '-', c='gray', 
                linewidth=width, alpha=0.2, zorder=2, label ='other paths')
        
        # plot other paths
        for k in range(2,np.size(paths,1)):                         
            path = np.asarray([coordenates[x,:] for x in paths[0:-1,k]])
            width = 30*(paths[-1,k]/max(paths[-1,:]))
            plt.plot(path[:,0],path[:,1], linestyle = '-', c='gray', 
                     linewidth=width, alpha=0.2, zorder=2, label ='_nolegend_')

       # plot most common path
        path = np.asarray([coordenates[x,:] for x in paths[0:-1,0]])
        plt.plot(path[:,0],path[:,1], linestyle = '-', c='orange', 
                 linewidth=8, alpha=0.8, zorder=2, label='best path')
        
        # Add a legend
        legend = plt.legend(loc='upper left', shadow=False,fontsize= 20)
        legend.get_frame().set_facecolor('white')  # legend background
        legend.get_frame().set_edgecolor('black')   # legend edge color
        for text in legend.get_texts():                 # text in legend
            plt.setp(text)

        # set the linewidth of each legend object
        for legobj in legend.legendHandles:
            legobj.set_linewidth(4)

        # plot states/cities
        plt.scatter(coordenates[0:-1,0],coordenates[0:-1,1],
                     c='white', s=800, 
                     label='white',alpha=1, 
                     edgecolors='black', zorder=3)
        
        # plot names/numbers of states
        for j in range(0,np.size(coordenates,0)-1):                        
            plt.text(coordenates[j,0], coordenates[j,1], str(j),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontdict=nodes_font)
        
        plt.show()                      
    
    
    
    
def plotBrokenLines(matrix,variable,baseline,title):
    
    axis_font = {'color':  'black',         # define font for axis labels
            'weight': 500,
            'size': 14 }
    title_font = {'color':  'black',        # define font for tittle
            'weight': 500,
            'size': 16 }
      
    gs = gridspec.GridSpec(12, 6)           # set up graph size
    gs.update(hspace=0.3)                   # gap between graphs
    ax = plt.subplot(gs[:-1, :])            # size top graph
    ax_base = plt.subplot(gs[-1,:])         # size botton graph
    ax_base.plot(np.zeros_like(matrix[:,0])+baseline,       # plot optimum performance
                               c='tomato', label='baseline')   
    for i in range(0,int(np.size(variable))):                      # plot learning data
        ax.plot(matrix[:,i], label=str(variable[i]))
    
    # limit the view of the graphs
    ax.set_ylim(np.amin(matrix)-1, np.amax(matrix))
    ax_base.set_ylim(baseline-(baseline/500), baseline+(baseline/500))
    
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
    ax.set_ylabel('Cost Ratio (agent/optimal)', fontdict=axis_font)                           # y-label
    ax.set_title(title, y=1.1, fontdict=title_font) # title
    
    # Add a legend
    legend = ax.legend(loc='upper right', shadow=False,fontsize= 14)
    legend.get_frame().set_facecolor('white')   # legend background
    legend.get_frame().set_edgecolor('gray')    # legend edge color
    for text in legend.get_texts():             # text in legend
        plt.setp(text)
    
    # set the linewidth of each legend object
    for legobj in legend.legendHandles:
        legobj.set_linewidth(4)
    
    plt.show() 


def plotLines(matrix,variable,title):
    
    axis_font = {'color':  'black',         # define font for axis labels
            'weight': 500,
            'size': 16 }
    
    plt.figure( figsize=(10, 7))
    
    for i in range(0,int(np.size(variable))):          # plot learning data
        plt.plot(matrix[:,i], label=str(variable[i]))
        
    # limit the view of the graphs
    plt.ylim(1, np.amax(matrix))

    # Add a grid, axes labels and tittle
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')   # grid on
    plt.xlabel('Epochs', fontdict=axis_font)                            # x-label
    plt.ylabel('Cost Ratio (agent/optimal)', fontdict=axis_font)        # y-label
    plt.suptitle(str(title), y=1, fontsize = 20, style = 'normal', weight=500) # title
    
    # Add a legend
    legend = plt.legend(loc='upper right', shadow=False,fontsize= 16)
    legend.get_frame().set_facecolor('white')   # legend background
    legend.get_frame().set_edgecolor('black')    # legend edge color
    for text in legend.get_texts():             # text in legend
        plt.setp(text)
    
    # set the linewidth of each legend object
    for legobj in legend.legendHandles:
        legobj.set_linewidth(5)
    
    plt.show()



def plotLines_base(matrix,variable,baseline,title):

    axis_font = {'color':  'black',         # define font for axis labels
            'weight': 500,
            'size': 16 }
    
    plt.figure( figsize=(10, 7))
    
    for i in range(0,int(np.size(variable))):          # plot learning data
        plt.plot(matrix[:,i], label=str(variable[i]))
    
    plt.plot(np.zeros_like(matrix[:,0])+baseline,       # plot optimum performance
                          c='tomato', label='baseline')   
    
    # limit the view of the graphs
    plt.ylim(baseline-(baseline/20), np.amax(matrix))

    # Add a grid, axes labels and tittle
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')    # grid on
    plt.xlabel('Epochs', fontdict=axis_font)                         # x-label
    plt.ylabel('Cost', fontdict=axis_font)                           # y-label
    plt.suptitle(title, y=1, fontsize = 20, style = 'normal', weight=500) # title
    
    # Add a legend
    legend = plt.legend(loc='upper right', shadow=False,fontsize= 16)
    legend.get_frame().set_facecolor('whitesmoke')   # legend background
    legend.get_frame().set_edgecolor('lightgray')    # legend edge color
    for text in legend.get_texts():             # text in legend
        plt.setp(text)
    
    # set the linewidth of each legend object
    for legobj in legend.legendHandles:
        legobj.set_linewidth(5)
    
    plt.show()

def diagnosticsPlot(plotData, legendData, title, saveFile): 
    
    plt.figure(figsize=(12,8))
    plt.plot(plotData)
    plt.title(title[0], fontsize = 20, style = 'normal', fontname = 'cambria')
    plt.xlabel('Epochs', fontsize = 17, style = 'italic', fontname = 'cambria')
    plt.ylabel('Cost Difference vs Optimal Tour', fontsize = 17, style = 'italic', fontname = 'cambria')
    plt.ylim(ymin=1)
    plt.legend(legendData.values(), fontsize= 12)
    plt.grid()       

    if saveFile == True:
        plt.savefig('results/' + title[1])
                         