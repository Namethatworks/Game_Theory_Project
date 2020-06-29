# setup:
# we consider 2 player game with 2 pure strategies for each player and payoffs
# utilities[i, j, k] = what a player i gets if he plays j and the other player plays k
# all i, j, k are elements of {0, 1}
#for definition of a, A, ... see article
# player 1 plays option 0 with probability alfa in [0, 1]
# player 2 plays option 0 with probability beta in [0, 1]
# strategies = [alfa, beta]

# what we do:
# we convert this problem to a problem of finding a fixed point of a function f: [0, 1]^2 -> [0, 1]^2 , where f is as given in the proof of Nash's theorem. We visualize f.

# usefull links:
# for definitions of g and details see subsection Alternate proof using the Brouwer fixed-point theorem on link https://en.wikipedia.org/wiki/Nash_equilibrium#Proof_of_existence
# for some simple games Nash equilibria can be found by this link (if you are too lazy ;)) : https://demonstrations.wolfram.com/SetOfNashEquilibriaIn2x2MixedExtendedGames/

import numpy as np
import matplotlib.pyplot as plt

#choose a game (uncomment)
name_game = 'Cooperation game'
a, A, d, D, b, B, c, C = 4, 4, 2, 2, 0, 0, 0, 0

#name_game = "No pure Nash equilibrium"
#a, A, d, D, b, B, c, C = 0, 0, 1, 1, 2, -1, 3, -1

#name_game = "Prisoner's dilemma"
#a, A, d, D, b, B, c, C = -1, -1, -2, -2, -3, -3, 0, 0


#form suitable for further calculations
utilities = np.array([[[a, b], [c, d]],
                      [[A, B], [C, D]]])

print(utilities[0,1,0])

def find_NE(a, A, d, D, b, B, c, C):
    NE = []
    if a>c and A>C:
        NE.append([1, 1])
    if d>b and D>B:
        NE.append([0, 0])
    if b>d and C>A:
        NE.append([1, 0])
    if c>a and B>D:
        NE.append([0, 1])
    mixed_nash = [(D-B)/(A-C-B+D), (d-b)/(a-c-b+d)]
    if (mixed_nash[0] >0) and (mixed_nash[0] < 1) and (mixed_nash[1]>0) and (mixed_nash[1]<1):
        NE.append(mixed_nash)
    return np.array(NE)

Nash_equilibria = find_NE(a, A, d, D, b, B, c, C)
print('Nash equilibria:')
print(Nash_equilibria)

def g(strategies, player, option):
    other_player = (player+1)%2
    u_change_to_option = utilities[player, option, 0] * strategies[other_player] + utilities[player, option, 1] * (1-strategies[other_player])
    u_dont_change = (utilities[player, 0, 0]*strategies[player] + utilities[player, 1, 0] * (1-strategies[player])) * strategies[other_player] \
                    + (utilities[player, 0, 1]*strategies[player] + utilities[player, 1, 1] * (1-strategies[player])) * (1 - strategies[other_player])

    Gain = 0 if (u_dont_change> u_change_to_option) else (u_change_to_option - u_dont_change)
    return Gain + strategies[player]*(1-2*option) + option #second part is just how likely is a player to play option. This way of writting works only for 2by2 game.

def f(strategies):
    """function mapping [0, 1]^2 -> [0, 1]^2. It's fixed points correspond to NE"""
    g00, g01, g10, g11 = g(strategies, 0, 0), g(strategies, 0, 1), g(strategies, 1, 0), g(strategies, 1, 1)
    return np.array([g00/(g00+g01), g10/(g10+g11)])


def plot_distance():
    """plots |f(x) - x| for x = [alfa, beta] representing all mixed strategies"""

    num_points = 100
    alfa = np.linspace(0, 1, num_points)
    beta = np.linspace(0, 1, num_points)
    ALFA, BETA = np.meshgrid(alfa, beta)
    Distance = np.empty((len(alfa), len(beta)))
    strategy = np.empty(2)
    for i in range(len(alfa)):
        for j in range(len(beta)):
            strategy[0], strategy[1] = ALFA[i, j], BETA[i, j]
            Distance[i, j] = np.sqrt(np.sum(np.square(f(strategy) - strategy)))

    plt.figure(figsize=(10, 13))
    plt.title(name_game)
    plt.contourf(ALFA, BETA, Distance, cmap='viridis')
    legend = plt.colorbar(orientation='horizontal')
    legend.set_label(r'$\vert f(\alpha, \beta) - (\alpha, \beta) \vert$')

    plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s = 80, color = 'red')

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.savefig(name_game + ' distance.png')
    plt.show()


def plot_line(fixed, value_of_fixed):
    """returns where a line is mapped. fixed is either 0 or 1, meaning vertical or horizontal lines, value_of_fixed \in [0, 1]"""

    not_fixed = (1+fixed)%2
    strategy = np.empty(2)
    strategy[fixed] = value_of_fixed
    running = np.linspace(0, 1, 50)
    X = np.empty(len(running))
    Y = np.empty(len(running))

    for i in range(len(running)):
        strategy[not_fixed] = running[i]
        X[i], Y[i] = f(strategy)

    return X, Y


def plot_line_compositions(xinput, yinput):
    """returns where a line is mapped. fixed is either 0 or 1, meaning vertical or horizontal lines, value_of_fixed \in [0, 1]"""
    strategy = np.empty(2)
    X = np.empty(len(xinput))
    Y = np.empty(len(yinput))

    for i in range(len(xinput)):
        strategy[0], strategy[1] = xinput[i], yinput[i]
        X[i], Y[i] = f(strategy)
    return X, Y

def plot_distortion():
    """plots grid distortion"""
    num_lines = 200
    value = np.linspace(0, 1, num_lines)
    ones = np.ones(num_lines)

    plt.figure(figsize = (10, 10))
    plt.title(name_game)
    #vertical lines
    for i in range(num_lines):
        X, Y = plot_line(0, value[i])
        plt.plot(value[i]*ones, value, ':', color = 'grey')
        plt.plot(X, Y, color = 'orange')

    #horizontal lines
    for i in range(num_lines):
        X, Y = plot_line(1, value[i])
        plt.plot(value, value[i] * ones, ':', color='grey')
        plt.plot(X, Y, color='orange')

    #plot Nash equilibria
    plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s = 80, color = 'red')

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')

    plt.savefig(name_game + ' distortion.png')
    plt.show()




def plot_composition():
    """plots grid distortion"""
    num_lines = 50
    value = np.linspace(0, 1, num_lines)
    ones = np.ones(num_lines)
    X1, X2, X3, X4, Y1, Y2, Y3, Y4 = np.empty(num_lines*num_lines), np.empty(num_lines*num_lines), np.empty(num_lines*num_lines), np.empty(num_lines*num_lines), np.empty(num_lines*num_lines), np.empty(num_lines*num_lines), np.empty(num_lines*num_lines), np.empty(num_lines*num_lines)

    square_gridX = np.empty((num_lines, num_lines))
    square_gridY = np.empty((num_lines, num_lines))

    #square grid
    for i in range(num_lines):
        square_gridX[i, :] = ones*value[i]
        square_gridY[i, :] = value
    flat_square_gridX = np.concatenate(square_gridX)
    flat_square_gridY = np.concatenate(square_gridY)
    pq = np.empty(2)
    N = Nash_equilibria[-1, :]

    #triangular regions
    # A1 = np.array([0, 0])
    # A2 = np.array([1, 0])
    # A3 = np.array([0, 1])
    # A4 = np.array([1, 1])
    #
    # for i in range(len(flat_square_gridX)):
    #     X1[i], Y1[i] = N + flat_square_gridX[i]*(A1-N)+flat_square_gridY[i]*(1-flat_square_gridX[i])*(A2-N)
    #     X2[i], Y2[i] = N + flat_square_gridX[i]*(A1-N)+flat_square_gridY[i]*(1-flat_square_gridX[i])*(A3-N)
    #     X3[i], Y3[i] = N + flat_square_gridX[i]*(A3-N)+flat_square_gridY[i]*(1-flat_square_gridX[i])*(A4-N)
    #     X4[i], Y4[i] = N + flat_square_gridX[i]*(A2-N)+flat_square_gridY[i]*(1-flat_square_gridX[i])*(A4-N)

    #square regions
    A1 = np.array([1-N[0], 0])
    A2 = np.array([0, 1-N[1]])
    A3 = np.array([-N[0], 0])
    A4 = np.array([0, -N[1]])

    for i in range(len(flat_square_gridX)):
        X1[i], Y1[i] = N + flat_square_gridX[i]*A1 + flat_square_gridY[i]*A2
        X2[i], Y2[i] = N + flat_square_gridX[i]*A2 + flat_square_gridY[i]*A3
        X3[i], Y3[i] = N + flat_square_gridX[i]*A3 + flat_square_gridY[i]*A4
        X4[i], Y4[i] = N + flat_square_gridX[i]*A4 + flat_square_gridY[i]*A1


    for num_composition in range(50):
        plt.title(name_game)
        plt.scatter(X1, Y1, color='red', s= 10)
        plt.scatter(X2, Y2, color='orange', s=10)
        plt.scatter(X3, Y3, color='green', s=10)
        plt.scatter(X4, Y4, color='blue', s=10)

        plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s=80, color='black')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.savefig(name_game + ' distortion.png')
        plt.savefig('half_colored_no_pure_NE/' + str(num_composition) + '.png')
        plt.close()

        for i in range(len(flat_square_gridX)):
            pq[0], pq[1] = X1[i], Y1[i]
            X1[i], Y1[i] = f(pq)

            pq[0], pq[1] = X2[i], Y2[i]
            X2[i], Y2[i] = f(pq)

            pq[0], pq[1] = X3[i], Y3[i]
            X3[i], Y3[i] = f(pq)

            pq[0], pq[1] = X4[i], Y4[i]
            X4[i], Y4[i] = f(pq)

def calculate_invariant_subset():
    """plots grid distortion"""
    num_lines = 50
    value = np.linspace(0, 1, num_lines)
    ones = np.ones(num_lines)
    X, Y = np.empty(num_lines*num_lines), np.empty(num_lines*num_lines)

    square_gridX = np.empty((num_lines, num_lines))
    square_gridY = np.empty((num_lines, num_lines))

    #square grid
    for i in range(num_lines):
        square_gridX[i, :] = ones*value[i]
        square_gridY[i, :] = value
    X = np.concatenate(square_gridX)
    Y = np.concatenate(square_gridY)
    pq = np.empty(2)

    for num_composition in range(30):
        for i in range(num_lines*num_lines):
            pq[0], pq[1] = X[i], Y[i]
            X[i], Y[i] = f(pq)


    plt.title(name_game)
    plt.scatter(X, Y, color='black', s= 10)

    plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s=80, color='red')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.savefig(name_game + ' distortion.png')
    plt.show()


def derivative(f, x, eps = 1e-6):
    fx = f(x)
    return np.array([[f([x[0] +eps, x[1]])[0] - fx[0], f([x[0], x[1]+eps])[0] - fx[0]], [f([x[0] +eps, x[1]])[1] - fx[1], f([x[0], x[1]+eps])[1] - fx[1]]]) / eps

def jacobian_first_component(f,x,eps = 0.1):
    fx = f(x)
    f0_x0 = (f([x[0] +eps, x[1]])[0] - fx[0])/eps
    f0_x1 = (f([x[0], x[1]+eps])[0] - fx[0])/eps
    f1_x0 = (f([x[0] +eps, x[1]])[1] - fx[1])/eps
    f1_x1 = (f([x[0], x[1]+eps])[1] - fx[1])/eps


    return f0_x0

def jacobian_determinant(f,x,eps = 0.1):
    fx = f(x)
    #print('x0 ',x[0])
    f0_x0 = (f([x[0] +eps, x[1]])[0] - fx[0])/eps
    f0_x1 = (f([x[0], x[1]+eps])[0] - fx[0])/eps
    f1_x0 = (f([x[0] +eps, x[1]])[1] - fx[1])/eps
    f1_x1 = (f([x[0], x[1]+eps])[1] - fx[1])/eps


    return f0_x0*f1_x1-f0_x1*f1_x0

def monte_carlo_derivative(f,x,eps,max_iter,ignore_borders):
    import math
    fx = f(x)
    # sample points around x with distance < eps
    iter = 0
    f_x = 0
    f_y = 0
    f_xy = 0
    while(iter<max_iter):
        # generate pseudorandom numbers.
        xplush = []
        xminush = []
        yplush = []
        yminush = []
        bothplush = []
        bothminush = []
        diffa = []
        diffb = []
        while(1): # sorry for this
            uniform_rand_samp1 = np.random.random_sample()
            uniform_rand_samp2 = np.random.random_sample()
            # generate sample points: x+h,x-h,y+h,y-h
            xplush = np.array([x[0]+eps*uniform_rand_samp1,x[1]])
            xminush = np.array([x[0]-eps*uniform_rand_samp1,x[1]])
            yplush = np.array([x[0],x[1]+eps*uniform_rand_samp2])
            yminush = np.array([x[0],x[1]-eps*uniform_rand_samp2])
            bothplush = np.array([x[0]+eps*uniform_rand_samp1,x[1]+uniform_rand_samp2*eps])
            bothminush = np.array([x[0]-eps*uniform_rand_samp1,x[1]-uniform_rand_samp2*eps])
            diffa = np.array([x[0]+eps*uniform_rand_samp1,x[1]-uniform_rand_samp2*eps])
            diffb = np.array([x[0]-eps*uniform_rand_samp1,x[1]+uniform_rand_samp2*eps])
            # make sure its in the bounding box. Can be a problem in the edges.
            if ((xminush[0]>0) and yminush[1]>0 and 1 > yplush[1] and 1 > xplush[0]):
                break
            elif (ignore_borders is True):
                break
        f_x = f_x + (f(xplush)-f(xminush))/((xplush[0]-xminush[0])*2) #finite differences: fx = (f(x+h,y)-f(x-h,y))/(2h) -
        f_y = f_y + (f(yplush)-f(yminush))/((yplush[1]-yminush[1])*2) #finite differences: fy = (f(x,y+h)-f(x,y-h))/(2h)
        # lets also use the sampled points to get the mixed derivative:
        f_xy = f_xy + (f(bothplush)-f(diffa)-f(diffb)+f(bothminush))/(4*(yplush[1]-yminush[1])*(xplush[0]-xminush[0])) #same principle, finite difference mixed derivative.


        iter = iter+1
    # average over iterations:
    f_x = f_x/max_iter
    f_y = f_y/max_iter
    f_xy = f_xy/max_iter

    # some housekeeping
    determinant = f_x[0]*f_y[1]-f_x[1]*f_y[0]
    rot = f_xy[0]-f_xy[1]
    return np.array([determinant,f_x[0],rot])

def plot_determinant():
    """plots determinant of jacobian of f for x = [alfa, beta] representing all mixed strategies"""

    num_points = 100
    alfa = np.linspace(0, 1, num_points)
    beta = np.linspace(0, 1, num_points)
    ALFA, BETA = np.meshgrid(alfa, beta)
    determin = np.empty((len(alfa), len(beta)))
    strategy = np.empty(2)
    for i in range(len(alfa)):
        for j in range(len(beta)):
            strategy[0], strategy[1] = ALFA[i, j], BETA[i, j]
            determin[i, j] = monte_carlo_derivative(f,strategy,1e-2,100,True)[0]

    plt.figure(figsize=(10, 13))
    plt.title(name_game)
    plt.contourf(ALFA, BETA, determin, cmap='viridis')
    legend = plt.colorbar(orientation='horizontal')
    legend.set_label(r'$\vert f(\alpha, \beta) - (\alpha, \beta) \vert$')

    plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s = 80, color = 'red')

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.savefig(name_game + ' determinant.png')
    #plt.show()

def plot_rotation():
    """plots 3rd component of curl for f for x = [alfa, beta] representing all mixed strategies"""

    num_points = 100
    alfa = np.linspace(0, 1, num_points)
    beta = np.linspace(0, 1, num_points)
    ALFA, BETA = np.meshgrid(alfa, beta)
    determin = np.empty((len(alfa), len(beta)))
    strategy = np.empty(2)
    for i in range(len(alfa)):
        for j in range(len(beta)):
            strategy[0], strategy[1] = ALFA[i, j], BETA[i, j]
            determin[i, j] = monte_carlo_derivative(f,strategy,1e-2,100,True)[2]

    plt.figure(figsize=(10, 13))
    plt.title(name_game)
    plt.contourf(ALFA, BETA, determin, cmap='viridis')
    legend = plt.colorbar(orientation='horizontal')
    legend.set_label(r'$\vert f(\alpha, \beta) - (\alpha, \beta) \vert$')

    plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s = 80, color = 'red')

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.savefig(name_game + ' rotation.png')
    #plt.show()

def plot_first_component():
    """plots first component of jacobian of f for x = [alfa, beta] representing all mixed strategies"""

    num_points = 100
    alfa = np.linspace(0, 1, num_points)
    beta = np.linspace(0, 1, num_points)
    ALFA, BETA = np.meshgrid(alfa, beta)
    determin = np.empty((len(alfa), len(beta)))
    strategy = np.empty(2)
    for i in range(len(alfa)):
        for j in range(len(beta)):
            strategy[0], strategy[1] = ALFA[i, j], BETA[i, j]
            determin[i, j] = monte_carlo_derivative(f,strategy,1e-2,100,True)[1]

    plt.figure(figsize=(10, 13))
    plt.title(name_game)
    plt.contourf(ALFA, BETA, determin, cmap='viridis')
    legend = plt.colorbar(orientation='horizontal')
    legend.set_label(r'$\vert f(\alpha, \beta) - (\alpha, \beta) \vert$')

    plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s = 80, color = 'red')

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.savefig(name_game + ' first_component.png')
    #plt.show()


if __name__ == '__main__':
    # print('right jacobian')
    print('derivative: ',derivative(f, Nash_equilibria[-1], eps = 1e-6))
    print('first_component: ',jacobian_first_component(f,Nash_equilibria[-1],eps = 0.01))
    print('determinant: ',jacobian_determinant(f,Nash_equilibria[-1],eps = 0.01))
    print('Nash_equilibiria: ', Nash_equilibria)
    print('Monte carlo: ', monte_carlo_derivative(f,Nash_equilibria[-1],1e-6,1000,True))
    plot_determinant()
    plot_rotation()
    plot_first_component()
    # print('left jacobian')
    # print(derivative(f, Nash_equilibria[-1], eps = -1e-6))

    #plot_composition()

    # plot_distance()
    #plot_distortion()
