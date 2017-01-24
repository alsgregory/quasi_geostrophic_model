""" test timestepper """

from __future__ import division

import numpy as np

from quasi_geostrophic_model import *


def test_timestepper_final_time():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.timestepper(2.0)

    assert QG.t == 2.0


def test_timestepper_final_time_2():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 0.0

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.timestepper(2.0)

    assert QG.t == 2.0


def test_timestepper_double_final_time():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    QG = quasi_geostrophic(dg_fs, cg_fs, var)

    # generate random end time that wouldn't be a factor of timestep
    T = np.random.uniform(0, 0.1, 1)[0]
    QG.timestepper(T)

    assert QG.t == T


def test_timestepper_double_final_time_2():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 0.0

    # generate random end time that wouldn't be a factor of timestep
    T = np.random.uniform(0, 0.1, 1)[0]
    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.timestepper(T)

    assert QG.t == T


def test_adaptive_timestepper():

    n = [2, 4]
    dt = np.zeros(2)

    for i in range(2):

        mesh = UnitSquareMesh(n[i], n[i])
        dg_fs = FunctionSpace(mesh, 'DG', 0)
        cg_fs = FunctionSpace(mesh, 'CG', 1)

        var = 0.0

        QG = quasi_geostrophic(dg_fs, cg_fs, var)
        dt[i] = QG.dt

    assert np.abs((2 * dt[1]) - dt[0]) < 1e-5


def test_adaptive_timestepper_2():

    n = [2, 4]
    dt_c = np.zeros(2)
    dt_f = np.zeros(2)

    for i in range(2):

        mesh = UnitSquareMesh(n[i], n[i])
        mesh_h = MeshHierarchy(mesh, 1)
        dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
        cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
        dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
        cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

        var = 0.0

        QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
        dt_c[i] = QG.dt_c
        dt_f[i] = QG.dt_f

    assert np.abs((2 * dt_c[1]) - dt_c[0]) < 1e-5
    assert np.abs((2 * dt_f[1]) - dt_f[0]) < 1e-5
    assert np.abs((2 * dt_f[0]) - dt_c[0]) < 1e-5
    assert np.abs((2 * dt_f[1]) - dt_c[1]) < 1e-5


def test_zero_time():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    f = Function(dg_fs).interpolate(ufl_expression)

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(0.0)

    assert QG.t == 0.0
    assert np.max(np.abs(QG.q_.dat.data - f.dat.data)) < 1e-5


def test_timestep_ic_solve():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression, 1.0)

    assert QG.qg_class.const_dt.dat.data[0] == 1e-3


def test_timestep_ic_solve_two():

    mesh = UnitSquareMesh(2, 2)
    mesh_hierarchy = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_hierarchy[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_hierarchy[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_hierarchy[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_hierarchy[1], 'CG', 1)

    var = 0.0

    x = SpatialCoordinate(mesh_hierarchy[0])
    ufl_expression_c = sin(2 * pi * x[0])
    x = SpatialCoordinate(mesh_hierarchy[1])
    ufl_expression_f = sin(2 * pi * x[0])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f, 1.0)

    assert QG.qg_class_c.const_dt.dat.data[0] == 1e-3
    assert QG.qg_class_f.const_dt.dat.data[0] == 1e-3


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
