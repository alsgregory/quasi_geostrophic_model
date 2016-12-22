""" quasi - geostrophic demo for single resolution, randomised const forcing, const i.c. """

from __future__ import division, absolute_import

from firedrake import *
from quasi_geostrophic_model import *

import numpy as np


# define mesh
mesh = UnitSquareMesh(30, 30)

# define initial condition ufl expression
x = SpatialCoordinate(mesh)
ufl_expression = (exp(-(pow(x[0] - 0.25, 2) / (2 * pow(0.1, 2)) +
                        pow(x[1] - 0.7, 2) / (2 * pow(0.1, 2)))) -
                  exp(-(pow(x[0] - 0.25, 2) / (2 * pow(0.1, 2)) +
                        pow(x[1] - 0.3, 2) / (2 * pow(0.1, 2)))) -
                  exp(-(pow(x[0] - 0.50, 2) / (2 * pow(0.1, 2)) +
                        pow(x[1] - 0.7, 2) / (2 * pow(0.1, 2)))) +
                  exp(-(pow(x[0] - 0.50, 2) / (2 * pow(0.1, 2)) +
                        pow(x[1] - 0.3, 2) / (2 * pow(0.1, 2)))) +
                  exp(-(pow(x[0] - 0.75, 2) / (2 * pow(0.1, 2)) +
                        pow(x[1] - 0.7, 2) / (2 * pow(0.1, 2)))) -
                  exp(-(pow(x[0] - 0.75, 2) / (2 * pow(0.1, 2)) +
                        pow(x[1] - 0.3, 2) / (2 * pow(0.1, 2)))))

# variance of random forcing
var = 2.0

# set-up QG class
dg_deg = 1
QG = quasi_geostrophic(mesh, var, dg_deg=dg_deg)

# initial condition
QG.initial_condition(ufl_expression)

# timestep
qFile = File("q.pvd")
Ts = np.linspace(0, 49, 50)
for i in range(len(Ts)):
    QG.timestepper(Ts[i])
    # write to file
    qFile.write(QG.q_)
