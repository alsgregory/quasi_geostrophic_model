""" test qg model """

from __future__ import division

import numpy as np

from quasi_geostrophic_model import *


def test_zero_solution_both_psi_and_q():

    mesh = UnitSquareMesh(2, 2)

    var = 0.0

    # no initial condition
    QG = quasi_geostrophic(mesh, var)
    QG.timestepper(1.0)

    assert np.all(QG.q_.dat.data[:] == 0.0)
    assert np.all(QG.psi_.dat.data[:] == 0.0)


def test_zero_solution_q():

    mesh = UnitSquareMesh(2, 2)

    var = 1.0

    # no initial condition
    QG = quasi_geostrophic(mesh, var)
    QG.timestepper(1.0)

    assert np.all(QG.q_.dat.data[:] == 0.0)
    assert np.any(QG.psi_.dat.data[:] != 0.0)


def test_random_q():

    mesh = UnitSquareMesh(2, 2)

    var = 1.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    QG = quasi_geostrophic(mesh, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    fs = QG.q_.function_space()
    f = Function(fs).assign(QG.q_)

    QG = quasi_geostrophic(mesh, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    g = Function(fs).project(QG.q_)

    assert norm(assemble(f - g)) > 0


def test_deterministic_q():

    mesh = UnitSquareMesh(2, 2)

    var = 0.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    QG = quasi_geostrophic(mesh, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    fs = QG.q_.function_space()
    f = Function(fs).assign(QG.q_)

    QG = quasi_geostrophic(mesh, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    g = Function(fs).project(QG.q_)

    assert norm(assemble(f - g)) == 0


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
