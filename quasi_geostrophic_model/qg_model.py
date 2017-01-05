""" qg model class """

from __future__ import division, absolute_import

from firedrake import *

from firedrake.mg.utils import get_level

from mpi4py import MPI

from quasi_geostrophic_model.min_dx import MinDx

import numpy as np


__all__ = ["quasi_geostrophic", "two_level_quasi_geostrophic"]


class base_class(object):

    def __init__(self, dg_fs, cg_fs, variance):

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
        self.dt = round(self.mdx.comm.allreduce(self.mdx.dat.data_ro.min(),
                                                MPI.MIN), 8)
        self.const_dt = Constant(self.dt)

        # set-up noise variance
        self.variance = variance

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

        # helmholtz solver functions and projections
        self.u = TrialFunction(self.Vcg)
        self.u_ = Function(self.Vcg)
        self.v = TestFunction(self.Vcg)
        self.dw = Function(self.Vcg)
        self.forcingProjector = Projector(self.u_, self.forcing)

        # set-up constants
        self.F = Constant(1.0)
        self.beta = Constant(0.1)

        # set-up ou forcing parameters
        self.theta = 2.0

        # set-up ou process
        self.sigma = 0.0

        # define perpendicular grad
        self.gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

        # set-up solvers
        self.setup_solver()

        super(base_class, self).__init__()

    def initial_condition(self, ufl_expression):

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

        # setup helmholtz solver for forcing
        self.hlhs = self.dw * self.v * dx
        self.hrhs = (dot(grad(self.v), grad(self.u)) + self.v * self.u) * dx

    def __update_sigma(self, dW):

        # time-step ou process
        self.sigma += -(self.theta * (self.const_dt.dat.data[0] * self.sigma)) + dW

    def __update_u(self):

        self.dw.dat.data[:] = (np.sqrt(self.const_dt.dat.data[0]) *
                               np.random.normal(0, 1.0,
                                                np.shape(self.dw.dat.data)))
        self.u_.assign(0)

        solve(self.hrhs == self.hlhs, self.u_,
              solver_parameters={'ksp_type': 'cg'})

    def __update_forcing(self):

        # scale with ou process
        self.u_.dat.data[:] = self.u_.dat.data[:] * self.sigma

        self.forcingProjector.project()

    def __update_q_forced(self):

        self.__update_forcing()
        self.q_forced.assign(self.forcing + self.q_old)

    def timestep(self):

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


class quasi_geostrophic(object):

    def __init__(self, dg_fs, cg_fs, variance):

        """ class specifying a randomly forced quasi geostrophic model

            Arguments:

            :arg dg_fs: The discontinuous :class:`FunctionSpace` used for Potential Vorticity
            :type dg_fs: :class:`FunctionSpace`

            :arg cg_fs: The continuous :class:`FunctionSpace` used for the Streamfunction
            :type cg_fs: :class:`FunctionSpace`

            :arg variance: Variance of ou process controlling magnitude of random forcing
            :type variance: int, if zero, forcing is off

        """

        self.qg_class = base_class(dg_fs, cg_fs, variance)

        self.variance = self.qg_class.variance
        self.mesh = self.qg_class.mesh

        self.dt = self.qg_class.dt

        self.q_ = self.qg_class.q_
        self.psi_ = self.qg_class.psi_

        self.t = 0

        super(quasi_geostrophic, self).__init__()

    def initial_condition(self, ufl_expression):

        # check for initial time
        if self.t != 0:
            raise ValueError("can't set initial condition when time is not zero")

        self.qg_class.initial_condition(ufl_expression)

    def timestepper(self, T):

        if self.t > T:
            raise ValueError("can't timestep to t when t is less than current time")

        # if the actual time, just return nothing
        if self.t == T:
            return

        while self.t < T:

            # re-adjust timestep if over the jump need to reach t
            if self.t + self.dt > T:
                self.qg_class.const_dt.assign(T - self.t)
            else:
                self.qg_class.const_dt.assign(self.dt)

            if self.variance > 0:
                # update ou process
                dW = (np.sqrt(self.qg_class.const_dt.dat.data[0]) *
                      np.random.normal(0, np.sqrt(self.variance), 1)[0])
                self.qg_class._base_class__update_sigma(dW)

            # update forcing and carry out time-step
            self.qg_class._base_class__update_u()  # if one wants to specify u, replace line
            self.qg_class._base_class__update_q_forced()
            self.qg_class.timestep()

            # update time
            self.t += self.qg_class.const_dt.dat.data[0]


class two_level_quasi_geostrophic(object):

    def __init__(self, dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, variance):

        """ class specifying a correlated randomly forced quasi geostrophic model
        on two levels

            Arguments:

            :arg dg_fs_c: The discontinuous :class:`FunctionSpace` used for Potential Vorticity
                          on the coarse level
            :type dg_fs_c: :class:`FunctionSpace`

            :arg cg_fs_c: The continuous :class:`FunctionSpace` used for the Streamfunction
                          on the coarse level
            :type cg_fs_c: :class:`FunctionSpace`

            :arg dg_fs_f: The discontinuous :class:`FunctionSpace` used for Potential Vorticity
                          on the fine level
            :type dg_fs_f: :class:`FunctionSpace`

            :arg cg_fs_f: The continuous :class:`FunctionSpace` used for the Streamfunction
                          on the fine level
            :type cg_fs_f: :class:`FunctionSpace`

            :arg variance: Variance of ou process controlling magnitude of random forcing
            :type variance: int, if zero, forcing is off

        """

        self.qg_class_c = base_class(dg_fs_c, cg_fs_c, variance)
        self.qg_class_f = base_class(dg_fs_f, cg_fs_f, variance)

        self.variance = variance
        self.mesh_c = self.qg_class_c.mesh
        self.mesh_f = self.qg_class_f.mesh

        # check for hierarchy existance
        hierarchy_c, self.lvl_c = get_level(self.mesh_c)
        hierarchy_f, self.lvl_f = get_level(self.mesh_f)
        if hierarchy_c is not hierarchy_f:
            raise ValueError('Coarse and fine meshes need to be on same hierarchy')
        else:
            self.mesh_hierarchy = hierarchy_c

        # check levels
        if self.lvl_c is not self.lvl_f - 1:
            raise ValueError('Coarse level is not one below fine level')

        self.dt_c = self.qg_class_c.dt
        self.dt_f = self.qg_class_f.dt

        # build a Function for aggregate u forcing from fine
        self.aggregate_u = Function(self.qg_class_f.Vcg)

        # check for refinement in time-steps
        if self.mesh_hierarchy.refinements_per_level is not 1:
            raise ValueError('Currently only 1 refinement per level can be used')

        if np.abs(self.dt_c - (2 * self.dt_f)) > 1e-5:
            raise ValueError('Time-steps are not of correct refinement')

        self.q_ = tuple([self.qg_class_c.q_, self.qg_class_f.q_])
        self.psi_ = tuple([self.qg_class_c.psi_, self.qg_class_f.psi_])

        self.t = 0

        # for checking
        self.__t_c = 0
        self.__t_f = 0

        super(two_level_quasi_geostrophic, self).__init__()

    def initial_condition(self, ufl_expression_c, ufl_expression_f):

        # check for initial time
        if self.t != 0:
            raise ValueError("can't set initial condition when time is not zero")

        self.qg_class_c.initial_condition(ufl_expression_c)
        self.qg_class_f.initial_condition(ufl_expression_f)

    def timestepper(self, T):

        if self.t > T:
            raise ValueError("can't timestep to t when t is less than current time")

        # if the actual time, just return nothing
        if self.t == T:
            return

        while self.t < T:

            if self.t + self.dt_f > T:

                # reassign both coarse and fine time-steps
                self.qg_class_c.const_dt.assign(T - self.t)
                self.qg_class_f.const_dt.assign(T - self.t)

                # do one solve for each and update time

                if self.variance > 0:
                    # compute random increment for ou process
                    dW = (np.sqrt(self.qg_class_f.const_dt.dat.data[0]) *
                          np.random.normal(0, np.sqrt(self.variance), 1)[0])

                    # update ou processes
                    self.qg_class_c._base_class__update_sigma(dW)
                    self.qg_class_f._base_class__update_sigma(dW)

                    # update fine u
                    self.qg_class_f._base_class__update_u()

                    # inject onto coarse u
                    inject(self.qg_class_f.u_, self.qg_class_c.u_)

                    # update both forcing
                    self.qg_class_c._base_class__update_q_forced()
                    self.qg_class_f._base_class__update_q_forced()

                # timestep
                self.qg_class_c.timestep()
                self.qg_class_f.timestep()

                # update time
                self.t += self.qg_class_c.const_dt.dat.data[0]

                self.__t_c += self.qg_class_c.const_dt.dat.data[0]
                self.__t_f += self.qg_class_f.const_dt.dat.data[0]

            else:

                self.qg_class_f.const_dt.assign(self.dt_f)

                # solve one fine then one fine and coarse

                if self.variance > 0:
                    # update fine ou process and aggregate increments
                    dWc = 0
                    dW = (np.sqrt(self.qg_class_f.const_dt.dat.data[0]) *
                          np.random.normal(0, np.sqrt(self.variance), 1)[0])
                    dWc += dW
                    self.qg_class_f._base_class__update_sigma(dW)

                    # update fine u
                    self.qg_class_f._base_class__update_u()
                    self.aggregate_u.assign(0)
                    self.aggregate_u.assign(self.aggregate_u + self.qg_class_f.u_)

                    # update forcing
                    self.qg_class_f._base_class__update_q_forced()

                # time-step
                self.qg_class_f.timestep()

                self.__t_f += self.qg_class_f.const_dt.dat.data[0]

                # now check if another whole fine timestep can be done
                if self.t + self.dt_f + self.dt_f > T:

                    # reassign both coarse and fine time-steps
                    self.qg_class_c.const_dt.assign(T - self.t)
                    self.qg_class_f.const_dt.assign(T - self.t - self.dt_f)

                else:

                    self.qg_class_c.const_dt.assign(self.dt_c)
                    self.qg_class_f.const_dt.assign(self.dt_f)

                if self.variance > 0:
                    # update fine and coarse ou process and aggregate increments
                    dW = (np.sqrt(self.qg_class_f.const_dt.dat.data[0]) *
                          np.random.normal(0, np.sqrt(self.variance), 1)[0])
                    dWc += dW
                    self.qg_class_c._base_class__update_sigma(dWc)
                    self.qg_class_f._base_class__update_sigma(dW)

                    # update fine u
                    self.qg_class_f._base_class__update_u()
                    self.aggregate_u.assign(self.aggregate_u + self.qg_class_f.u_)

                    # inject onto coarse u
                    inject(self.aggregate_u, self.qg_class_c.u_)

                    # update both forcing
                    self.qg_class_c._base_class__update_q_forced()
                    self.qg_class_f._base_class__update_q_forced()

                # timestep
                self.qg_class_c.timestep()
                self.qg_class_f.timestep()

                # update time
                self.t += self.qg_class_c.const_dt.dat.data[0]

                self.__t_c += self.qg_class_c.const_dt.dat.data[0]
                self.__t_f += self.qg_class_f.const_dt.dat.data[0]
