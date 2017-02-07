""" quasi - geostrophic demo of convergence for deterministic settings (single level) """


from __future__ import division, absolute_import

from firedrake import *

from quasi_geostrophic_model import *

import numpy as np

import matplotlib.pyplot as plot


# define mesh hierarchy
mesh = UnitSquareMesh(20, 20)
L = 4
mesh_hierarchy = MeshHierarchy(mesh, L)


# define initial condition ufl expression
def ic(mesh):
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
    return ufl_expression


# parameters
var = 0.0
T = 20

# find finest res solution
dg_fs_finest = FunctionSpace(mesh_hierarchy[-1], 'DG', 1)
cg_fs_finest = FunctionSpace(mesh_hierarchy[-1], 'CG', 1)
QG_finest = quasi_geostrophic(dg_fs_finest, cg_fs_finest, var)
QG_finest.initial_condition(ic(mesh_hierarchy[-1]))
QG_finest.timestepper(T)
comp_q = QG_finest.q_
comp_psi = QG_finest.psi_

# preallocate error and functions to project coarser solutions to
error_q = np.zeros(L)
error_psi = np.zeros(L)
q_func_finest = Function(dg_fs_finest)
psi_func_finest = Function(cg_fs_finest)

# preallocate dx array
dxs = np.zeros(L)

# iterate over resolutions
for l in range(L):

    # find dx for each level
    dxs[l] = norm(MinDx(mesh_hierarchy[l]))

    # define function spaces
    dg_fs = FunctionSpace(mesh_hierarchy[l], 'DG', 1)
    cg_fs = FunctionSpace(mesh_hierarchy[l], 'CG', 1)

    # simulate solution on resolution
    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ic(mesh_hierarchy[l]))
    QG.timestepper(T)

    # project solution to finest mesh and calculate error
    prolong(QG.q_, q_func_finest)
    prolong(QG.psi_, psi_func_finest)
    error_q[l] = norm(q_func_finest - comp_q)
    error_psi[l] = norm(psi_func_finest - comp_psi)

    print 'completed simulation on level: ', l

# plot convergence
plot.loglog(dxs, error_q, 'r*-')
plot.loglog(dxs, error_psi, 'y^-')
plot.loglog(dxs, 1e1 * (dxs ** 2), 'k--')
plot.xlabel('dx')
plot.ylabel('norm error')
plot.legend(['potential vorticity', 'streamfunction', 'quadratic decay'])
plot.show()
