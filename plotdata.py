# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:05:31 2017

@author: miguelesteras
"""

def heatmap(grid,a,b):
    
    # import dependecies
    import numpy as np
    import matplotlib.pyplot as plt
    
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





def plotRoutes(seqs,file_xy,variable):

    # import dependencies
    import collections
    import numpy as np
    import matplotlib.pyplot as plt
    from loadCoordenates import loadCoordenates


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
        n = np.size(seqs,0)/np.size(variable)
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
        
        # Plot routes and nodes
        fig, mapa = plt.subplots()
        mapa.patch.set_facecolor('forestgreen')
        #mapa.set_ylim()
        #mapa.set_xlim()
        plt.grid(b=False)
        plt.title(str(variable[i]), y=1.05,fontdict=title_font)
        plt.ylabel('latitud', fontdict=axis_font)
        plt.xlabel('longitud', fontdict=axis_font)
        plt.xticks([], [])
        plt.yticks([], [])
                
        # load coordenates of states and plot states/cities
        coordenates = loadCoordenates(file_xy)      
        nodes = np.asarray([coordenates[x,:] for x in paths[0:-1,0]])
        mapa.scatter(nodes[:,0],nodes[:,1],
                     c='white', s=800, 
                     label='white',alpha=1, 
                     edgecolors='black', zorder=3)
        
        # plot names/numbers of states
        for j in range(0,np.size(nodes,0)):                        
            plt.text(nodes[j,0], nodes[j,1], str(paths[j,0]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontdict=nodes_font)
                    
        # plot paths
        for k in range(0,np.size(paths,1)):                         
            path = np.asarray([coordenates[x,:] for x in paths[0:-1,k]])
            width = 5*(paths[-1,k]/max(paths[-1,:]))
            plt.plot(path[:,0],path[:,1], 
                     linestyle = '-', c='yellow', 
                     linewidth=width, alpha=0.8, zorder=2)
        plt.show()                      
    
    
    
def plotBrokenLines(matrix,variable,baseline,title):
    
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib.pyplot as plt

    axis_font = {'color':  'black',         # define font for axis labels
            'weight': 500,
            'size': 14 }
    title_font = {'color':  'black',        # define font for tittle
            'weight': 500,
            'size': 16 }
      
    gs = gridspec.GridSpec(6, 3)            # set up graph size
    gs.update(hspace=0.3)                   # gap between graphs
    ax = plt.subplot(gs[:-1, :])            # size top graph
    ax_base = plt.subplot(gs[-1,:])         # size botton graph
    ax_base.plot(np.zeros_like(matrix[:,1])+baseline,       # plot optimum performance
                               c='tomato', label='baseline')   
    for i in range(0,int(np.size(variable))):                      # plot learning data
        ax.plot(matrix[:,i], label=str(variable[i]))
    
    # limit the view of the graphs
    ax.set_ylim(np.amin(matrix)-1, np.amax(matrix))
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


def plotLines(matrix,variable,baseline,title):
    
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib.pyplot as plt

    axis_font = {'color':  'black',         # define font for axis labels
            'weight': 500,
            'size': 14 }
    title_font = {'color':  'black',        # define font for tittle
            'weight': 500,
            'size': 16 }
    fig = plt.figure()
    ax = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=5)  # size graph
    
    for i in range(0,int(np.size(variable))):          # plot learning data
        ax.plot(matrix[:,i], label=str(variable[i]))
    
    ax.plot(np.zeros_like(matrix[:,1])+baseline,       # plot optimum performance
                          c='tomato', label='baseline')   
    
    # limit the view of the graphs
    ax.set_ylim(baseline-1, np.amax(matrix))
    
    # hide the spines between ax and ax_base
    # ax_base.yaxis.set_ticks(np.arange(baseline-1, baseline+1, 1))
    
    # Add a grid, axes labels and tittle
    ax.grid(b=True, which='major', color='lightgray', linestyle='-')    # grid on
    ax.set_xlabel('Epochs', fontdict=axis_font)                         # x-label
    ax.set_ylabel('Cost', fontdict=axis_font)                           # y-label
    ax.set_title(title, y=1.1, fontdict=title_font) # title
    
    # Add a legend
    legend = ax.legend(loc='upper right', shadow=False,fontsize= 10)
    legend.get_frame().set_facecolor('white')   # legend background
    legend.get_frame().set_edgecolor('None')    # legend edge color
    for text in legend.get_texts():             # text in legend
        plt.setp(text)
    plt.show()                                