import math
import torch

import torch.nn as nn
from Simulator.Task import generate_weighted_function


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class Task_Query_Loss(nn.Module):

    def __init__(self, mode, constraints):

        super(Task_Query_Loss, self).__init__()
        self.mode = mode
        self.constraints = constraints
        self.density = DENSITY
        self.grid_task_index = generate_weighted_function()
        self.delta = 1

    def forward(self, trajectory1, trajectory2):
        size = trajectory1.size()

        for i in range(size[1]):
            track1 = trajectory2[:, i, :].tolist()
            lat_grid, long_grid = nor_to_real_grid(track1[0][0], track1[0][1])
            weight = self.grid_task_index[lat_grid][long_grid] / self.density * 0.0001 + self.delta
            weight = math.sqrt(weight)

            trajectory2[:, i, :] = torch.mul(trajectory2[:, i, :], weight)
            trajectory1[:, i, :] = torch.mul(trajectory1[:, i, :], weight)

        this_criterion = nn.MSELoss()

        return this_criterion(trajectory1, trajectory2)
