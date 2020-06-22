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
# name_game = 'Cooperation game'
# a, A, d, D, b, B, c, C = 4, 4, 2, 2, 0, 0, 0, 0

name_game = "No pure Nash equilibrium"
a, A, d, D, b, B, c, C = 0, 0, 1, 1, 2, -1, 3, -1

# name_game = "Prisoner's dilemma"
# a, A, d, D, b, B, c, C = -1, -1, -2, -2, -3, -3, 0, 0


#form suitable for further calculations
utilities = np.array([[[a, b], [c, d]],
                      [[A, B], [C, D]]])


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
    num_lines = 100
    value = np.linspace(0, 1, num_lines)
    ones = np.ones(num_lines)
    X = np.empty((num_lines, num_lines))
    Y = np.empty((num_lines, num_lines))

    for i in range(num_lines):
        #vertical lines
        X[i, :] = ones*value[i]
        Y[i, :] = value

    plt.figure(figsize=(10, 10))

    for num_composition in range(50):
        plt.title(name_game)

        for i in range(num_lines):
            X[i], Y[i] = plot_line_compositions(X[i], Y[i])
            plt.scatter(X[i], Y[i], color='orange')

        plt.scatter(Nash_equilibria[:, 0], Nash_equilibria[:, 1], s=80, color='red')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.savefig(name_game + ' distortion.png')
        plt.savefig('iteration_grid_no_pure_NE/' + str(num_composition) + '.png')
        plt.close()

def derivative(f, x, eps = 1e-6):
    fx = f(x)
    return np.array([[f([x[0] +eps, x[1]])[0] - fx[0], f([x[0], x[1]+eps])[0] - fx[0]], [f([x[0] +eps, x[1]])[1] - fx[1], f([x[0], x[1]+eps])[1] - fx[1]]]) / eps

def gg(r):
    x, y = r[0], r[1]
    return [x**2 + y, 3*y + x]


if __name__ == '__main__':
    # print('right jacobian')
    # print(derivative(f, Nash_equilibria[-1], eps = 1e-6))
    # print('left jacobian')
    # print(derivative(f, Nash_equilibria[-1], eps = -1e-6))


    #plot_composition()

    # plot_distance()
    plot_distortion()
