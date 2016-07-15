from . import Datum

import numpy as np
import sys

BOARD_SIZE = 10

def load_batch(n_batch):
    data = []
    for i_datum in range(n_batch):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        c_wall = np.random.randint(BOARD_SIZE / 2) + BOARD_SIZE / 4
        r_hole = np.random.randint(BOARD_SIZE)
        board[:, c_wall] = 1
        board[r_hole, c_wall] = 0

        init = (np.random.randint(BOARD_SIZE), 0)
        goal = (BOARD_SIZE - 1, BOARD_SIZE - 1)
        features = np.concatenate((board.ravel(), init))
        demonstration = [init, (r_hole, 0), (r_hole, BOARD_SIZE-1), goal]

        datum = Datum(features, init, goal, demonstration)
        data.append(datum)
    return data

