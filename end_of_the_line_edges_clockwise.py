import numpy as np
import sys
import matplotlib.pyplot as plt
import nash_to_brouwer as ntb


def plot_vector_filed(func):
    N, delta = 30, 0.01
    plt.figure(figsize = (10, 10))
    plt.title(ntb.name_game)

    #prepare grid
    xarr = np.linspace(delta, 1-delta, N)
    yarr = np.linspace(delta, 1-delta, N)
    X, Y = np.meshgrid(xarr, yarr)
    Xflat, Yflat = np.concatenate(X), np.concatenate(Y)
    D = np.empty((len(Xflat), 2))
    r = np.empty(2)
    colorarr = []#np.empty(len(Xflat), dtype = str)
    ryclockwise = [] #PB: the list with the vertices of red-yellow horizontal edges
    yranticlockwise = [] #PB: the list with the vertices of red-yellow vertical edges
    
    #calculate displacements and convert to colors
    for i in range(len(Xflat)):
        r[0], r[1] = Xflat[i], Yflat[i]
        D[i] = func(r) - r  #displacement of R after the mapping
        D[i] /= np.sqrt(np.sum(np.square(D[i])))
        angle = np.arctan2(D[i, 1], D[i, 0])*2/np.pi
        if angle >= 0 and angle< 1:
            colorarr.append('gold')
        elif angle < 0 and angle > - 1.5:
            colorarr.append('red')            
        else:
            colorarr.append('blue')
            
    #PB: Here we search for yellow-red and red-yellow horizontal edges and append the coordinate list ryhor 
    #PB: Note that the modulo condition has to be added to make sure we don't count edges that "skip the boundaries"
    for i in range(len(Xflat)):
        if colorarr[i] == 'red' and colorarr[i+1] == 'gold' and (i+1) % N != 0:
            yranticlockwise.append([[Xflat[i],Yflat[i]],[Xflat[i+1],Yflat[i+1]]])
        elif colorarr[i] == 'gold' and colorarr[i+1] == 'red' and (i+1) % N != 0:
            ryclockwise.append([[Xflat[i],Yflat[i]],[Xflat[i+1],Yflat[i+1]]])
            
    #PB:Here we search for yellow-red and red-yellow diagonal edges and append the coordinate list rydiag
    #PB:Note that the modulo condition has to be added to make sure we don't count edges that "skip the boundaries"
    #PB:Note also, that we don't scan the N-th latitudonal line hence we reduce the length of our "for" loop
    for i in range(len(Xflat)-N):
        if colorarr[i] == 'red' and colorarr[i+N-1] == 'gold' and i % N != 0:
            yranticlockwise.append([[Xflat[i],Yflat[i]],[Xflat[i+N-1],Yflat[i+N-1]]])
        elif colorarr[i] == 'gold' and colorarr[i+N-1] == 'red' and i % N != 0:
            ryclockwise.append([[Xflat[i],Yflat[i]],[Xflat[i+N-1],Yflat[i+N-1]]])
    #PB:Here we search for yellow-red and red-yellow diagonal edges and append the coordinate list rydiag
    #PB:Here we do not risk skipping the boundaries
    #PB:Note also, that we don't scan the N-th latitudonal line hence we reduce the length of our "for" loop
    for i in range(len(Xflat)-N):
        if colorarr[i] == 'red' and colorarr[i+N] == 'gold':
            ryclockwise.append([[Xflat[i],Yflat[i]],[Xflat[i+N-1],Yflat[i+N-1]]])
        elif colorarr[i] == 'gold' and colorarr[i+N] == 'red':
            yranticlockwise.append([[Xflat[i],Yflat[i]],[Xflat[i+N],Yflat[i+N]]])
            
            
            
    plt.quiver(Xflat, Yflat, D[:, 0], D[:, 1], color = colorarr)

    #plot triangles
    lwmy = 0.5 #width of grid lines
    for i in range(N):
        plt.plot([xarr[i], xarr[i]], [yarr[0], yarr[-1]], lw = lwmy, color = 'grey') #vertical lines
        plt.plot([xarr[0], xarr[-1]], [yarr[i], yarr[i]], lw = lwmy, color = 'grey') #horizontal lines
        plt.plot([xarr[0], xarr[i]], [yarr[i], yarr[0]], lw = lwmy, color = 'grey') #diagonal down lines
        plt.plot([xarr[i], xarr[-1]], [yarr[-1], yarr[i]], lw = lwmy, color = 'grey') #diagonal down lines
    
       #PB: Now we draw the yellow-red edges
    for i in range(len(ryclockwise)):
        x_values = [ryclockwise[i][0][0],ryclockwise[i][1][0]]
        y_values = [ryclockwise[i][0][1],ryclockwise[i][1][1]]
        plt.plot(x_values,y_values, color = 'green')
        
    for i in range(len(yranticlockwise)):
        x_values = [yranticlockwise[i][0][0],yranticlockwise[i][1][0]]
        y_values = [yranticlockwise[i][0][1],yranticlockwise[i][1][1]]
        plt.plot(x_values,y_values, color = 'green')
        

    

    
    plt.scatter(ntb.Nash_equilibria[:, 0], ntb.Nash_equilibria[:, 1], s = 80, color = 'black')

    plt.savefig('brouwer_to_end_of_the_line' + ntb.name_game)
    plt.show()


plot_vector_filed(ntb.f)

#test test test test