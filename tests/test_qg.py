""" test qg model """

from __future__ import division

import numpy as np

from quasi_geostrophic_model import *


def test_delayed_start():

    mesh = UnitSquareMesh(10, 10)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)

    # get initial q and psi
    QG.timestepper(1.0)
    halfFpsi = Function(cg_fs).assign(QG.psi_)
    halfFq = Function(dg_fs).assign(QG.q_)

    # delayed start one
    QG1 = quasi_geostrophic(dg_fs, cg_fs, var, start_time=1.0, start_psi=halfFpsi, start_q=halfFq)
    QG1.timestepper(3.0)

    # carry on initial simulation
    QG.timestepper(3.0)
    Fpsi = Function(cg_fs).assign(QG.psi_)
    Fq = Function(dg_fs).assign(QG.q_)

    # compare
    assert norm(assemble(QG1.q_ - Fq)) < 1e-10
    assert norm(assemble(QG1.psi_ - Fpsi)) < 1e-10


def test_delayed_start_two_level():

    mesh = UnitSquareMesh(10, 10)
    mesh_hierarchy = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_hierarchy[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_hierarchy[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_hierarchy[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_hierarchy[1], 'CG', 1)

    var = 0.0

    xc = SpatialCoordinate(mesh_hierarchy[0])
    ufl_expression_c = sin(2 * pi * xc[0])
    xf = SpatialCoordinate(mesh_hierarchy[1])
    ufl_expression_f = sin(2 * pi * xf[0])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)

    # get initial q and psi
    QG.timestepper(1.0)
    halfFpsi_c = Function(cg_fs_c).assign(QG.psi_[0])
    halfFq_c = Function(dg_fs_c).assign(QG.q_[0])
    halfFpsi_f = Function(cg_fs_f).assign(QG.psi_[1])
    halfFq_f = Function(dg_fs_f).assign(QG.q_[1])

    # delayed start one
    QG1 = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var,
                                      start_time=1.0, start_psi_c=halfFpsi_c, start_q_c=halfFq_c,
                                      start_psi_f=halfFpsi_f, start_q_f=halfFq_f)
    QG1.timestepper(3.0)

    # carry on initial simulation
    QG.timestepper(3.0)
    Fpsi_c = Function(cg_fs_c).assign(QG.psi_[0])
    Fq_c = Function(dg_fs_c).assign(QG.q_[0])
    Fpsi_f = Function(cg_fs_f).assign(QG.psi_[1])
    Fq_f = Function(dg_fs_f).assign(QG.q_[1])

    # compare
    assert norm(assemble(QG1.q_[0] - Fq_c)) < 1e-10
    assert norm(assemble(QG1.psi_[0] - Fpsi_c)) < 1e-10
    assert norm(assemble(QG1.q_[1] - Fq_f)) < 1e-10
    assert norm(assemble(QG1.psi_[1] - Fpsi_f)) < 1e-10


def test_fixed_ic():

    mesh = UnitSquareMesh(2, 2)
    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    x = SpatialCoordinate(mesh)
    ufl_expression = sin(2 * pi * x[0])

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)
    G = Function(dg_fs).assign(QG.q_)

    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression)
    assert norm(assemble(G - QG.q_)) == 0


def test_fixed_ic_two_level():

    mesh = UnitSquareMesh(2, 2)
    mesh_h = MeshHierarchy(mesh, 1)
    dg_fs_c = FunctionSpace(mesh_h[0], 'DG', 0)
    cg_fs_c = FunctionSpace(mesh_h[0], 'CG', 1)
    dg_fs_f = FunctionSpace(mesh_h[1], 'DG', 0)
    cg_fs_f = FunctionSpace(mesh_h[1], 'CG', 1)

    var = 0.0

    x_c = SpatialCoordinate(mesh_h[0])
    x_f = SpatialCoordinate(mesh_h[1])
    ufl_expression_c = sin(2 * pi * x_c[0])
    ufl_expression_f = sin(2 * pi * x_f[0])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)
    G_c = Function(dg_fs_c).assign(QG.q_[0])
    G_f = Function(dg_fs_f).assign(QG.q_[1])

    QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)
    QG.initial_condition(ufl_expression_c, ufl_expression_f)
    assert norm(assemble(G_c - QG.q_[0])) == 0
    assert norm(assemble(G_f - QG.q_[1])) == 0


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
    assert np.any(QG.psi_.dat.data[:] == 0.0)


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
    assert np.any(QG.psi_[0].dat.data[:] == 0.0)

    assert np.all(QG.q_[1].dat.data[:] == 0.0)
    assert np.any(QG.psi_[1].dat.data[:] == 0.0)


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


def test_one_sigma():

    mesh = SquareMesh(1, 1, 2)

    dg_fs = FunctionSpace(mesh, 'DG', 0)
    cg_fs = FunctionSpace(mesh, 'CG', 1)

    var = 0.0

    QG = quasi_geostrophic(dg_fs, cg_fs, var)

    n = 50
    s = 1.0
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
