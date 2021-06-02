"""
This module contains functions for generating co-ordinate values, which are 
used to test the Delaunay triangulation algorithm. 
"""

# Standard library imports
import sys
import numpy as np
import json
from numpy.random import default_rng
from math import sqrt, ceil
from solver import World, Input

# ----------------------------- Points generators -----------------------------

"""
The following functions are used to generate the inital values for the 
position of each point.
"""

def random(num_points, span):
    """
    This function generates a set of random x and y coordinates using the 
    numpy uniform random number generator 'numpy.random.default_rng().uniform'.
    Parameters
    ----------
    num_points : int
        The number of points to generate
    span : World class
        The world defines the range of values the coordinates can have
    Returns
    -------
    pts : list
        A list of length num_points, where each element is a point
        e.g. [ [x1, y1], [x2, y2], ... [xn, yn] ]
    """
    x_vals = default_rng().uniform(span.x_min, span.x_max, num_points)
    y_vals = default_rng().uniform(span.y_min, span.y_max, num_points)
    pts = [list(i) for i in zip(x_vals.tolist(), y_vals.tolist())]
    
    # Alternative version to return numpy array
    # pts = np.concatenate((x_vals, y_vals)).reshape(-1, 2)
    return pts

def lattice(num_points, span):
    """
    This function generates a set of points which are set on a grid. The points
    are spaced equally in x and y using the numpy.linspace function. To have
    a point at every place on the grid requires that the number of points is a
    square number. If the number of points is not a square number, points are
    removed randomly until there are 'num_points' many points returned.
    Parameters
    ----------
    num_points : int
        The number of points to generate
    span : World class
        The world defines the range of values the coordinates can have
    Returns
    -------
    pts : list
        A list of length num_points, where each element is a point
        e.g. [ [x1, y1], [x2, y2], ... [xn, yn] ]
    """
    num_sqrt = ceil(sqrt(num_points))
    x_vals = np.linspace(span.x_min, span.x_max, num_sqrt)
    y_vals = np.linspace(span.y_min, span.y_max, num_sqrt)
    _X, _Y = np.meshgrid(x_vals, y_vals)
    pts = [list(i) for i in zip(_X.ravel().tolist(), _Y.ravel().tolist())]
    
    # If the number of points is not square, remove excess points randomly
    if not sqrt(num_points).is_integer():
        current_num = len(pts)
        to_remove = current_num - num_points
        indices = np.random.choice(current_num, to_remove, replace=False)
        pts = [i for j, i in enumerate(pts) if j not in indices]
    return pts

def generateConfig(num_points, world_width, workers_num = 1000):
    world_size = [0, world_width, 0, world_width]
    world = World(world_size)

    positions = random(num_points, world) 

    config = Input(world_size, positions, workers_num)
    return config

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception('Generation script require')

    num_points = int(sys.argv[1])
    world_width = int(sys.argv[2])

    config = generateConfig(num_points, world_width)

    filename = f"points-{num_points}-{world_width}.json"

    config.writeTo(filename)

    res = config.readFrom(filename)

    