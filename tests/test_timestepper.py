""" test timestepper """

from __future__ import division

import numpy as np

from quasi_geostrophic_model import *


def test_timestepper_final_time():

    mesh = UnitSquareMesh(2, 2)

    var = 0.0

    QG = quasi_geostrophic(mesh, var)
    QG.timestepper(2.0)

    assert QG.t == 2.0


def test_timestepper_double_final_time():

    mesh = UnitSquareMesh(2, 2)

    var = 0.0

    QG = quasi_geostrophic(mesh, var)

    # generate random end time that wouldn't be a factor of timestep
    T = np.random.uniform(0, 0.1, 1)[0]
    QG.timestepper(T)

    assert QG.t == T


def test_adaptive_timestepper():

    n = [2, 4]
    dt = np.zeros(2)

    for i in range(2):

        mesh = UnitSquareMesh(n[i], n[i])

        var = 0.0

        QG = quasi_geostrophic(mesh, var)
        dt[i] = QG.dt

    assert np.abs((2 * dt[1]) - dt[0]) < 1e-5


def test_zero_time():

    mesh = UnitSquareMesh(2, 2)

    var = 0.0

    V = FunctionSpace(mesh, 'DG', 0)
    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    f = Function(V).interpolate(ufl_expression)

    QG = quasi_geostrophic(mesh, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(0.0)

    assert QG.t == 0.0
    assert np.max(np.abs(QG.q_.dat.data - f.dat.data)) < 1e-5


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
