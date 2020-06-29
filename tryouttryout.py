# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 01:20:38 2020

@author: Pieter-Bart Peters
"""

import numpy as np
import matplotlib.pyplot as plt

#choose a game (uncomment)
# name_game = 'Cooperation game'
# a, A, d, D, b, B, c, C = 4, 4, 2, 2, 0, 0, 0, 0

name_game = "No pure Nash equilibrium"
a1, a2, d1, d2, b1, b2, c1, c2 = 0, 0, 1, 1, 2, -1, 3, -1

# name_game = "Prisoner's dilemma"
# a, A, d, D, b, B, c, C = -1, -1, -2, -2, -3, -3, 0, 0


utilities = np.array([[[a, b], [c, d]],
                      [[A, B], [C, D]]])

