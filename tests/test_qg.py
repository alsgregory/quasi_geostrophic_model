""" test qg model """

from __future__ import division

import numpy as np

from quasi_geostrophic_model import *


def test_two_level_random_u_c():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 2.0

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)

    ts = np.linspace(1, 3, 3)
    for i in range(len(ts)):
        QG.timestepper(ts[i])
        assert np.any(QG.qg_class_c.psi_forced.dat.data !=
                      np.zeros(len(QG.qg_class_c.psi_forced.dat.data)))
        if i > 0:
            assert np.any(QG.qg_class_c.psi_forced.dat.data != u_)
        u_ = np.copy(QG.qg_class_c.psi_forced.dat.data)


def test_zero_solution_both_psi_and_q():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    # no initial condition
    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.timestepper(1.0)

    assert np.all(QG.q_.dat.data[:] == 0.0)
    assert np.all(QG.psi_.dat.data[:] == 0.0)


def test_zero_solution_both_psi_and_q_2():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 0.0

    # no initial condition
    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.timestepper(1.0)

    assert np.all(QG.q_[0].dat.data[:] == 0.0)
    assert np.all(QG.psi_[0].dat.data[:] == 0.0)

    assert np.all(QG.q_[1].dat.data[:] == 0.0)
    assert np.all(QG.psi_[1].dat.data[:] == 0.0)


def test_zero_solution_q():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 1.0

    # no initial condition
    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.timestepper(1.0)

    assert np.all(QG.q_.dat.data[:] == 0.0)
    assert np.any(QG.psi_.dat.data[:] != 0.0)


def test_zero_solution_q_2():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 1.0

    # no initial condition
    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.timestepper(1.0)

    assert np.all(QG.q_[0].dat.data[:] == 0.0)
    assert np.any(QG.psi_[0].dat.data[:] != 0.0)

    assert np.all(QG.q_[1].dat.data[:] == 0.0)
    assert np.any(QG.psi_[1].dat.data[:] != 0.0)


def test_correct_fs():

    deg = [0, 1, 2]
    mesh = UnitSquareMesh(2, 2)

    var = 1.0

    for i in range(len(deg)):

        dg_fs = FunctionSpace(mesh, 'DG', deg[i])
        cg_fs = FunctionSpace(mesh, 'CG', int(deg[i] + 1))

        QG = quasi_geostrophic(dg_fs, cg_fs, var)
        QG.timestepper(1.0)

        assert dg_fs == QG.q_.function_space()


def test_update_sigma():

    mesh = SquareMesh(1, 1, 2)

    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 1.0

    QG = quasi_geostrophic(dg_fs, cg_fs, var)

    n = 50
    s = 0.0
    for i in range(n):
        QG.timestepper(i + 1)

        assert np.abs(s - QG.qg_class.sigma) > 0.0
        s = float(QG.qg_class.sigma)


def test_zero_sigma():

    mesh = SquareMesh(1, 1, 2)

    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    QG = quasi_geostrophic(dg_fs, cg_fs, var)

    n = 50
    s = 0.0
    for i in range(n):
        QG.timestepper(i + 1)

        assert np.abs(s - QG.qg_class.sigma) < 1e-5
        s = float(QG.qg_class.sigma)


def test_random_q():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 1.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    f = Function(dg_fs).assign(QG.q_)

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    g = Function(dg_fs).project(QG.q_)

    assert norm(assemble(f - g)) > 0


def test_random_q_2():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 1.0

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)

    x_c = SpatialCoordinate(mesh_h[0])
    ufl_expression_c = sin(2 * pi * x_c[0])
    x_f = SpatialCoordinate(mesh_h[1])
    ufl_expression_f = sin(2 * pi * x_f[0])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)
    QG.timestepper(1.0)

    assert isinstance(QG.q_, tuple)

    f_c = Function(dg_fs_c).assign(QG.q_[0])
    f_f = Function(dg_fs_f).assign(QG.q_[1])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)
    QG.timestepper(1.0)

    g_c = Function(dg_fs_c).project(QG.q_[0])
    g_f = Function(dg_fs_f).project(QG.q_[1])

    assert norm(assemble(f_c - g_c)) > 0
    assert norm(assemble(f_f - g_f)) > 0


def test_deterministic_q():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    f = Function(dg_fs).assign(QG.q_)

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)
    QG.timestepper(1.0)

    g = Function(dg_fs).project(QG.q_)

    assert norm(assemble(f - g)) == 0


def test_deterministic_q_2():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 0.0

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)

    x_c = SpatialCoordinate(mesh_h[0])
    ufl_expression_c = sin(2 * pi * x_c[0])
    x_f = SpatialCoordinate(mesh_h[1])
    ufl_expression_f = sin(2 * pi * x_f[0])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)
    QG.timestepper(1.0)

    assert isinstance(QG.q_, tuple)

    f_c = Function(dg_fs_c).assign(QG.q_[0])
    f_f = Function(dg_fs_f).assign(QG.q_[1])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)
    QG.timestepper(1.0)

    g_c = Function(dg_fs_c).project(QG.q_[0])
    g_f = Function(dg_fs_f).project(QG.q_[1])

    assert norm(assemble(f_c - g_c)) == 0
    assert norm(assemble(f_f - g_f)) == 0


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
