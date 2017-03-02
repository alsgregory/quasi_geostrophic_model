""" quasi - geostrophic demo of convergence for deterministic settings (single level) """


from __future__ import division, absolute_import

from firedrake import *

from quasi_geostrophic_model import *

import numpy as np

import matplotlib.pyplot as plot


# define mesh hierarchy
mesh = UnitSquareMesh(5, 5)
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

# preallocate error
error_q = np.zeros((2, L))
error_psi = np.zeros((2, L))

# iterate over degree of potential vorticity
for k in range(2):

    # find finest res solution
    dg_fs_finest = FunctionSpace(mesh_hierarchy[-1], 'DG', k)
    cg_fs_finest = FunctionSpace(mesh_hierarchy[-1], 'CG', 1)
    QG_finest = quasi_geostrophic(dg_fs_finest, cg_fs_finest, var)
    QG_finest.initial_condition(ic(mesh_hierarchy[-1]))
    QG_finest.timestepper(T)
    comp_q = QG_finest.q_
    comp_psi = QG_finest.psi_

    # preallocate functions to project coarser solutions to
    q_func_finest = Function(dg_fs_finest)
    psi_func_finest = Function(cg_fs_finest)

    # preallocate dx array
    dxs = np.zeros(L)

    # iterate over resolutions
    for l in range(L):

        # find dx for each level
        dxs[l] = norm(MinDx(mesh_hierarchy[l]))

        # define function spaces
        dg_fs = FunctionSpace(mesh_hierarchy[l], 'DG', k)
        cg_fs = FunctionSpace(mesh_hierarchy[l], 'CG', 1)

        # simulate solution on resolution
        QG = quasi_geostrophic(dg_fs, cg_fs, var)
        QG.initial_condition(ic(mesh_hierarchy[l]))
        QG.timestepper(T)

        # project solution to finest mesh and calculate error
        prolong(QG.q_, q_func_finest)
        prolong(QG.psi_, psi_func_finest)
        error_q[k, l] = assemble(((q_func_finest - comp_q) ** 2) * dx)
        error_psi[k, l] = assemble(((psi_func_finest - comp_psi) ** 2) * dx)

        print 'completed simulation of PV on level: ', l, ' with degree: ', k

# plot convergence
plot.loglog(dxs, error_psi[0, :], 'y^-')
plot.loglog(dxs, error_q[0, :], 'r*-')
plot.loglog(dxs, error_q[1, :], 'bo-')
plot.loglog(dxs, 5e0 * (dxs ** 2), 'k')
plot.loglog(dxs, 1e0 * (dxs ** 4), 'k--')
plot.xlabel('dx')
plot.ylabel('\int (\psi_L - \psi)^2 dx')
plot.legend(['streamfunction CG1', 'potential vorticity DG0',
             'potential vorticity DG1',
             'linear decay', 'quadratic decay'])
plot.show()
