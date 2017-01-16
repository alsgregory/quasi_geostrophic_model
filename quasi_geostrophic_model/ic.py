""" Random inifinite dimensional initial condition for QG system """

from __future__ import division, absolute_import

from firedrake import *

from quasi_geostrophic_model import quasi_geostrophic


def random_ic(dg_fs, cg_fs):

    # find spatial coordinates of mesh and create fixed initial condition structure
    x = SpatialCoordinate(dg_fs.mesh())
    ufl_expression = (conditional(x[1] > 0.5 + (0.25 * sin(4 * pi * (x[0]))), 1.0, 0.0) +
                      conditional(x[1] < 0.5 + (0.25 * sin(4 * pi * (x[0]))), -1.0, 0.0))

    # create QG class
    var = 10.0
    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)

    # run one timestep
    QG.timestepper(QG.dt)

    return QG.q_
