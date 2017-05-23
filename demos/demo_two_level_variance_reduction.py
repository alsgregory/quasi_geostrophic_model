""" sample variance decay of two level QG system """

from __future__ import division
from __future__ import absolute_import

from firedrake import *

from quasi_geostrophic_model import *

import numpy as np

import matplotlib.pyplot as plot


# define mesh hierarchy
mesh = UnitSquareMesh(5, 5)

L = 4

mesh_hierarchy = MeshHierarchy(mesh, L)

# define sample size
n = 10

# define variance
variance = 0.125


# define initial condition function
def ic(mesh, xp):

    x = SpatialCoordinate(mesh)
    ufl_expression = (exp(-(pow(x[0] - 0.5 + xp, 2) / (2 * pow(0.25, 2)) +
                            pow(x[1] - 0.7, 2) / (2 * pow(0.1, 2)))) -
                      exp(-(pow(x[0] - 0.5 + xp, 2) / (2 * pow(0.25, 2)) +
                            pow(x[1] - 0.3, 2) / (2 * pow(0.1, 2)))))

    return ufl_expression


sample_variances_difference = np.zeros(L)

finest_fs = FunctionSpace(mesh_hierarchy[-1], 'CG', 1)

for l in range(L):

    print 'level: ', l

    meshc = mesh_hierarchy[l]
    meshf = mesh_hierarchy[l + 1]

    # define fs
    dg_fs_c = FunctionSpace(meshc, 'DG', 1)
    cg_fs_c = FunctionSpace(meshc, 'CG', 1)
    dg_fs_f = FunctionSpace(meshf, 'DG', 1)
    cg_fs_f = FunctionSpace(meshf, 'CG', 1)

    m = Function(finest_fs)
    sq = Function(finest_fs)

    for j in range(n):

        print 'sample: ', j

        # set-up system
        QG = two_level_quasi_geostrophic(dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, variance)

        # fixed ic
        xp = 0
        QG.initial_condition(ic(meshc, xp), ic(meshf, xp))

        # time-step
        QG.timestepper(3.0)

        # prolong coarse and fine
        comp_c = Function(finest_fs)
        comp_f = Function(finest_fs)
        prolong(QG.psi_[0], comp_c)
        if l < L - 1:
            prolong(QG.psi_[1], comp_f)
        else:
            comp_f.assign(QG.psi_[1])

        m += assemble((comp_f - comp_c) * (1.0 / n))
        sq += assemble(((comp_f - comp_c) ** 2) * (1.0 / n))

    ff = Function(finest_fs).assign((sq - (m ** 2)))

    sample_variances_difference[l] = assemble(ff * dx)

dxf = 1.0 / 2 ** (np.linspace(1, L, L))

plot.loglog(dxf, sample_variances_difference)
plot.loglog(dxf, 1e-9 * dxf ** (4), 'k--')
plot.xlabel('normalized dx of coarse level')
plot.ylabel('sample variance difference')
plot.show()
