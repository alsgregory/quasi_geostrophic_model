""" qg model class """

from __future__ import division, absolute_import

from firedrake import *

from mpi4py import MPI

from quasi_geostrophic_model.min_dx import MinDx

import numpy as np


class quasi_geostrophic(object):

    def __init__(self, dg_fs, cg_fs, variance):

        """ class specifying a quasi geostrophic model

            Arguments:

            :arg dg_fs: The discontinuous :class:`FunctionSpace` used for Potential Vorticity
            :type dg_fs: :class:`FunctionSpace`

            :arg cg_fs: The ontinuous :class:`FunctionSpace` used for Potential Vorticity
            :type cg_fs: :class:`FunctionSpace`

            :arg variance: Variance of random forcing
            :type variance: int, if zero, forcing is off

        """

        # define function spaces
        self.Vdg = dg_fs
        self.Vcg = cg_fs

        self.mesh = dg_fs.mesh()

        # check that both function spaces have same mesh
        if self.mesh is not cg_fs.mesh():
            raise ValueError("both function spaces need to be on same mesh")

        # only allow 2D meshes
        if self.mesh.geometric_dimension() != 2:
            raise ValueError("model is only compatible with 2-dimensional meshes")

        # set-up adaptive time-step
        self.mdx = MinDx(self.mesh)
        self.dt = self.mdx.comm.allreduce(self.mdx.dat.data_ro.min(),
                                          MPI.MIN)
        self.const_dt = Constant(self.dt)

        # set-up noise variance
        self.variance = variance

        # define current time
        self.t = 0

        # define functions
        self.q_ = Function(self.Vdg)  # actual q
        self.psi_ = Function(self.Vcg)  # actual streamfunction
        self.q_old = Function(self.Vdg)  # last time-step q
        self.dq = Function(self.Vdg)  # intermediate q for inter - RK steps
        self.q_forced = Function(self.Vdg)  # forcing q to invert streamfunction
        self.forcing = Function(self.Vdg)  # forcing

        # solver functions
        self.psi = TrialFunction(self.Vcg)
        self.phi = TestFunction(self.Vcg)
        self.q = TrialFunction(self.Vdg)
        self.p = TestFunction(self.Vdg)

        # set-up constants
        self.F = Constant(1.0)
        self.beta = Constant(0.1)

        # define perpendicular grad
        self.gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

        # set-up solvers
        self.setup_solver()

        super(quasi_geostrophic, self).__init__()

    def initial_condition(self, ufl_expression):

        # check for initial time
        if self.t != 0:
            raise ValueError("can't set initial condition when time is not zero")

        # interpolate expression
        self.q_.interpolate(ufl_expression)

    def setup_solver(self):

        # setup psi solver
        self.Apsi = ((self.F * self.psi * self.phi +
                      inner(grad(self.psi), grad(self.phi)) -
                      self.beta * self.phi * self.psi.dx(1)) *
                     dx)
        self.Lpsi = -self.q_forced * self.phi * dx

        # boundary conditions
        self.bc = [DirichletBC(self.Vcg, 0., 1),
                   DirichletBC(self.Vcg, 0., 2),
                   DirichletBC(self.Vcg, 0., 3),
                   DirichletBC(self.Vcg, 0., 4)]

        self.psi_problem = LinearVariationalProblem(self.Apsi, self.Lpsi, self.psi_, bcs=self.bc)

        self.psi_solver = LinearVariationalSolver(self.psi_problem,
                                                  solver_parameters={'ksp_type': 'cg',
                                                                     'pc_type': 'sor'})

        self.n = FacetNormal(self.mesh)
        self.un = 0.5 * (dot(self.gradperp(self.psi_), self.n) +
                         abs(dot(self.gradperp(self.psi_), self.n)))

        # setup advection solver
        self.a_mass = self.p * self.q * dx

        self.a_int = (dot(grad(self.p), -self.gradperp(self.psi_) * self.q)) * dx

        self.a_flux = (dot(jump(self.p), self.un('+') * self.q('+') -
                           self.un('-') * self.q('-'))) * dS

        self.arhs = self.a_mass - self.const_dt * (self.a_int + self.a_flux)

        self.q_problem = LinearVariationalProblem(self.a_mass, action(self.arhs,
                                                                      self.q_old), self.dq)

        self.q_solver = LinearVariationalSolver(self.q_problem,
                                                solver_parameters={'ksp_type': 'cg',
                                                                   'pc_type': 'sor'})

    def __update_forcing(self):

        if self.variance == 0:
            self.forcing.assign(0)
        else:
            self.forcing.dat.data[:] = (self.variance *
                                        np.random.normal(0, 1, np.shape(self.forcing.dat.data)))

    def __update_q_forced(self):

        self.__update_forcing()
        self.q_forced.assign(self.forcing + self.q_old)

    def timestepper(self, T):

        if self.t > T:
            raise ValueError("can't timestep to t when t is less than current time")

        # if the actual time, just return nothing
        if self.t == T:
            return

        while self.t < T:

            # re-adjust timestep if over the jump need to reach t
            if self.t + self.dt > T:
                self.const_dt.assign(T - self.t)
            else:
                self.const_dt.assign(self.dt)

            # update forcing
            self.__update_q_forced()

            # set old time-step to be current one
            self.q_old.assign(self.q_)

            # 1st RK step
            self.psi_solver.solve()
            self.q_solver.solve()
            self.q_old.assign(self.dq)

            # 2nd RK step
            self.psi_solver.solve()
            self.q_solver.solve()
            self.q_old.assign(((3.0 / 4.0) * self.q_) + ((1.0 / 4.0) * self.dq))

            # 3rd RK step
            self.psi_solver.solve()
            self.q_solver.solve()
            self.q_.assign(((1.0 / 3.0) * self.q_) + ((2.0 / 3.0) * self.dq))

            # update time
            self.t += self.const_dt.dat.data[0]
