""" Random infinite dimensional initial condition for QG system """

from __future__ import division, absolute_import

from firedrake import *

from quasi_geostrophic_model import quasi_geostrophic, two_level_quasi_geostrophic


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


def two_level_random_ic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f):

    # find spatial coordinates of mesh and create fixed initial condition structure
    x_c = SpatialCoordinate(dg_fs_c.mesh())
    ufl_expression_c = (conditional(x_c[1] > 0.5 + (0.25 * sin(4 * pi * (x_c[0]))), 1.0, 0.0) +
                        conditional(x_c[1] < 0.5 + (0.25 * sin(4 * pi * (x_c[0]))), -1.0, 0.0))
    x_f = SpatialCoordinate(dg_fs_f.mesh())
    ufl_expression_f = (conditional(x_f[1] > 0.5 + (0.25 * sin(4 * pi * (x_f[0]))), 1.0, 0.0) +
                        conditional(x_f[1] < 0.5 + (0.25 * sin(4 * pi * (x_f[0]))), -1.0, 0.0))

    # create QG class
    var = 10.0
    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)

    # run one timestep
    QG.timestepper(QG.qg_class_c.dt)

    return QG.q_
