""" quasi - geostrophic demo for two resolution levels, randomised const forcing, const i.c. """

from __future__ import division, absolute_import

from firedrake import *
from quasi_geostrophic_model import *

import numpy as np


# define mesh
mesh = UnitSquareMesh(30, 30)

# define mesh hierarchy
mesh_hierarchy = MeshHierarchy(mesh, 1)

# define function spaces
dg_fs_c = FunctionSpace(mesh_hierarchy[0], 'DG', 1)
cg_fs_c = FunctionSpace(mesh_hierarchy[0], 'CG', 1)
dg_fs_f = FunctionSpace(mesh_hierarchy[1], 'DG', 1)
cg_fs_f = FunctionSpace(mesh_hierarchy[1], 'CG', 1)

# define initial condition ufl expression
x_c = SpatialCoordinate(mesh_hierarchy[0])
x_f = SpatialCoordinate(mesh_hierarchy[1])
ufl_expression_c = (exp(-(pow(x_c[0] - 0.25, 2) / (2 * pow(0.1, 2)) +
                          pow(x_c[1] - 0.7, 2) / (2 * pow(0.1, 2)))) -
                    exp(-(pow(x_c[0] - 0.25, 2) / (2 * pow(0.1, 2)) +
                          pow(x_c[1] - 0.3, 2) / (2 * pow(0.1, 2)))) -
                    exp(-(pow(x_c[0] - 0.50, 2) / (2 * pow(0.1, 2)) +
                          pow(x_c[1] - 0.7, 2) / (2 * pow(0.1, 2)))) +
                    exp(-(pow(x_c[0] - 0.50, 2) / (2 * pow(0.1, 2)) +
                          pow(x_c[1] - 0.3, 2) / (2 * pow(0.1, 2)))) +
                    exp(-(pow(x_c[0] - 0.75, 2) / (2 * pow(0.1, 2)) +
                          pow(x_c[1] - 0.7, 2) / (2 * pow(0.1, 2)))) -
                    exp(-(pow(x_c[0] - 0.75, 2) / (2 * pow(0.1, 2)) +
                          pow(x_c[1] - 0.3, 2) / (2 * pow(0.1, 2)))))

ufl_expression_f = (exp(-(pow(x_f[0] - 0.25, 2) / (2 * pow(0.1, 2)) +
                          pow(x_f[1] - 0.7, 2) / (2 * pow(0.1, 2)))) -
                    exp(-(pow(x_f[0] - 0.25, 2) / (2 * pow(0.1, 2)) +
                          pow(x_f[1] - 0.3, 2) / (2 * pow(0.1, 2)))) -
                    exp(-(pow(x_f[0] - 0.50, 2) / (2 * pow(0.1, 2)) +
                          pow(x_f[1] - 0.7, 2) / (2 * pow(0.1, 2)))) +
                    exp(-(pow(x_f[0] - 0.50, 2) / (2 * pow(0.1, 2)) +
                          pow(x_f[1] - 0.3, 2) / (2 * pow(0.1, 2)))) +
                    exp(-(pow(x_f[0] - 0.75, 2) / (2 * pow(0.1, 2)) +
                          pow(x_f[1] - 0.7, 2) / (2 * pow(0.1, 2)))) -
                    exp(-(pow(x_f[0] - 0.75, 2) / (2 * pow(0.1, 2)) +
                          pow(x_f[1] - 0.3, 2) / (2 * pow(0.1, 2)))))

# variance of random forcing
var = 0.25

# set-up two level QG class
QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, var)

# initial condition
QG.initial_condition(ufl_expression_c, ufl_expression_f)

# timestep
qFile_c = File("q_c.pvd")
qFile_f = File("q_f.pvd")
psiFile_c = File("psi_c.pvd")
psiFile_f = File("psi_f.pvd")
Ts = np.linspace(0, 49, 50)
for i in range(len(Ts)):
    QG.timestepper(Ts[i])
    # write to file
    qFile_c.write(QG.q_[0])
    qFile_f.write(QG.q_[1])
    psiFile_c.write(QG.psi_[0])
    psiFile_f.write(QG.psi_[1])
