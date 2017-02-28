""" qg model class with optional adaptive time-stepping """

from __future__ import division, absolute_import

from firedrake import *

from firedrake.mg.utils import get_level

from mpi4py import MPI

from quasi_geostrophic_model.min_dx import MinDx

import numpy as np


__all__ = ["quasi_geostrophic", "two_level_quasi_geostrophic"]


class base_class(object):

    def __init__(self, dg_fs, cg_fs, variance, adaptive_timestep):

        # define function spaces
        self.Vdg = dg_fs
        self.Vcg = cg_fs

        self.mesh = dg_fs.mesh()

        # define base noise fields
        self.Xis = []
        self.nxi = 3  # number of terms in KL expansion
        self.W = []
        for i in range(self.nxi):
            self.Xis.append(Function(self.Vcg))
            self.W.append(Constant(0))

        x = SpatialCoordinate(self.mesh)
        self.Xis[0].interpolate(sin(pi * x[0]) * sin(pi * x[1]))
        self.Xis[1].interpolate(sin(2 * pi * x[0]) * sin(pi * x[1]))
        self.Xis[2].interpolate(sin(pi * x[0]) * sin(pi * 2 * x[1]))

        # check that both function spaces have same mesh
        if self.mesh is not cg_fs.mesh():
            raise ValueError("both function spaces need to be on same mesh")

        # only allow 2D meshes
        if self.mesh.geometric_dimension() != 2:
            raise ValueError("model is only compatible with 2-dimensional meshes")

        # set-up time-step
        self.mdx = MinDx(self.mesh)
        if adaptive_timestep is True:
            # scale (dt / dx) by p <= 0.3 as we want Courant <= 0.3. Assume |u| = 1
            self.dt = 0.3 * round(self.mdx.comm.allreduce(self.mdx.dat.data_ro.min(),
                                                          MPI.MIN), 8)
        else:
            self.dt = 0.3 * 0.05  # allows up to dx = 0.05

        self.const_dt = Constant(self.dt)

        # set-up noise variance
        self.variance = variance

        # define functions
        self.q_ = Function(self.Vdg)  # actual q
        self.psi_ = Function(self.Vcg)  # actual streamfunction
        self.q_old = Function(self.Vdg)  # last time-step q
        self.dq = Function(self.Vdg)  # intermediate q for inter - RK steps
        self.forcing = Function(self.Vcg)

        # solver functions
        self.psi = TrialFunction(self.Vcg)
        self.phi = TestFunction(self.Vcg)
        self.q = TrialFunction(self.Vdg)
        self.p = TestFunction(self.Vdg)

        # set-up constants
        self.F = Constant(1.0)
        self.beta = Constant(0.1)

        # set-up ou forcing parameters
        self.theta = 2.0

        # initializes optional ou process
        self.sigma = 1.0

        # spatial scaling constant (do we need?)
        v = TestFunction(self.Vcg)
        ones = Function(self.Vcg).assign(1)
        self.ml = assemble(v * ones * dx)

        # define perpendicular grad
        self.gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

        # set-up solvers
        self.setup_solver()

        super(base_class, self).__init__()

    def initial_condition(self, ufl_expression):

        """ interpolates initial condition to q field """

        # interpolate structured expression for q and solve for q
        self.q_.interpolate(ufl_expression)

    def setup_solver(self):

        """ set-ups the psi and q solvers and specifies boundary conditions """

        # setup psi solver
        self.Apsi = ((self.F * self.psi * self.phi +
                      inner(grad(self.psi), grad(self.phi)) -
                      self.beta * self.phi * self.psi.dx(1)) *
                     dx)
        self.Lpsi = -self.q_old * self.phi * dx

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

    def __update_sigma(self, dW):

        """ generates optional ou process controlling variance of random field """

        # time-step ou process
        self.sigma += -(self.theta * (self.const_dt.dat.data[0] * self.sigma)) + dW

    def __update_forcing(self):

        """ updates the random increments for the kl expansion of the random field """

        # generate random field
        for i in range(self.nxi):
            self.W[i].assign(np.random.normal(0, 1.0) *
                             0.01 * self.variance)

    def __update_kl_expansion(self):

        """ generates the random field with a kahunen loeve expansion of Xi's """

        self.forcing.assign(0)
        for i in range(self.nxi):
            self.forcing.assign(self.forcing + ((self.W[i] * self.Xis[i])))
        self.forcing.assign(self.forcing)

    def __update_psi_forced(self):

        """ adds the random field to psi at each RK step """

        if self.variance == 0:
            self.psi_.assign(self.psi_)
        else:
            self.psi_.assign(self.psi_ + self.forcing)

    def timestep(self):

        """ completes one RK3 time-step """

        # set old time-step to be current one
        self.q_old.assign(self.q_)

        # 1st RK step
        self.psi_solver.solve()
        self.__update_psi_forced()
        self.q_solver.solve()
        self.q_old.assign(self.dq)

        # 2nd RK step
        self.psi_solver.solve()
        self.__update_psi_forced()
        self.q_solver.solve()
        self.q_old.assign(((3.0 / 4.0) * self.q_) + ((1.0 / 4.0) * self.dq))

        # 3rd RK step
        self.psi_solver.solve()
        self.__update_psi_forced()
        self.q_solver.solve()
        self.q_.assign(((1.0 / 3.0) * self.q_) + ((2.0 / 3.0) * self.dq))


class quasi_geostrophic(object):

    def __init__(self, dg_fs, cg_fs, variance, adaptive_timestep=False):

        """ class specifying a randomly forced quasi geostrophic model

            Arguments:

            :arg dg_fs: The discontinuous :class:`FunctionSpace` used for Potential Vorticity
            :type dg_fs: :class:`FunctionSpace`

            :arg cg_fs: The continuous :class:`FunctionSpace` used for the Streamfunction
            :type cg_fs: :class:`FunctionSpace`

            :arg variance: Variance of ou process controlling magnitude of random forcing
            :type variance: int, if zero, forcing is off

        """

        self.qg_class = base_class(dg_fs, cg_fs, variance, adaptive_timestep)

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
                # update optional ou process
                dW = (np.sqrt(self.qg_class.const_dt.dat.data[0]) *
                      np.random.normal(0, np.sqrt(self.variance), 1)[0])
                self.qg_class._base_class__update_sigma(dW)

                # update forcing and carry out time-step
                self.qg_class._base_class__update_forcing()
                self.qg_class._base_class__update_kl_expansion()

            self.qg_class.timestep()

            # update time
            self.t += self.qg_class.const_dt.dat.data[0]

            # find velocity
            self.u = norm(self.qg_class.gradperp(self.psi_))
            if (self.u * self.dt) / round(self.qg_class.mdx.comm.allreduce(self.qg_class.mdx.dat.data_ro.min(),
                                          MPI.MIN), 8) > 0.3:
                print 'Courant number exceeded 0.3'


class two_level_quasi_geostrophic(object):

    def __init__(self, dg_fs_c, cg_fs_c, dg_fs_f, cg_fs_f, variance, adaptive_timestep=False):

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

        self.qg_class_c = base_class(dg_fs_c, cg_fs_c, variance, adaptive_timestep)
        self.qg_class_f = base_class(dg_fs_f, cg_fs_f, variance, adaptive_timestep)

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
        self.cdtf = 0

        if self.mesh_hierarchy.refinements_per_level is not 1:
            raise ValueError('Currently only 1 refinement per level can be used')

        # check for refinement in time-steps if adaptive-timestepping
        if adaptive_timestep is True:

            if np.abs(self.dt_c - (2 * self.dt_f)) > 1e-5:
                raise ValueError('Time-steps are not of correct refinement')

        else:

            if np.abs(self.dt_c - self.dt_f) > 1e-5:
                raise ValueError('Time-steps are not the same without adaptivity')

        self.q_ = tuple([self.qg_class_c.q_, self.qg_class_f.q_])
        self.psi_ = tuple([self.qg_class_c.psi_, self.qg_class_f.psi_])

        self.t = 0

        # for checking
        self.__t_c = 0
        self.__t_f = 0

        self.adaptive_timestep = adaptive_timestep

        super(two_level_quasi_geostrophic, self).__init__()

    def __coupling_coarse_forcing(self):

        """ adds increments of fine forcing in inbetween time-steps on to coarse forcing """
        # timestep number normalizing
        self.cdtf += 1

        for i in range(self.qg_class_f.nxi):
            self.qg_class_c.W[i].assign(self.qg_class_c.W[i] + self.qg_class_f.W[i])

    def __normalize_coupled_coarse_forcing(self):

        """ normalizes coarse forcing with scaled number of fine timesteps """
        for i in range(self.qg_class_f.nxi):
            self.qg_class_c.W[i].assign(self.qg_class_c.W[i] /
                                        np.sqrt(self.cdtf))

    def __reset_coarse_forcing(self):

        """ resets increments of coarse forcing """
        self.cdtf = 0
        for i in range(self.qg_class_f.nxi):
            self.qg_class_c.W[i].assign(0)

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

            if self.adaptive_timestep is True:

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

                        # update fine forcing, couple coarse forcing
                        self.__reset_coarse_forcing()
                        self.qg_class_f._base_class__update_forcing()
                        self.__coupling_coarse_forcing()
                        self.__normalize_coupled_coarse_forcing()

                        # generate kl expansions
                        self.qg_class_f._base_class__update_kl_expansion()
                        self.qg_class_c._base_class__update_kl_expansion()

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

                        # update fine forcing, couple coarse forcing
                        self.__reset_coarse_forcing()
                        self.qg_class_f._base_class__update_forcing()
                        self.__coupling_coarse_forcing()

                        # generate fine kl expansion
                        self.qg_class_f._base_class__update_kl_expansion()

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

                        # update fine forcing, couple coarse forcing
                        self.qg_class_f._base_class__update_forcing()
                        self.__coupling_coarse_forcing()
                        self.__normalize_coupled_coarse_forcing()

                        # generate kl expansions
                        self.qg_class_f._base_class__update_kl_expansion()
                        self.qg_class_c._base_class__update_kl_expansion()

                    # timestep
                    self.qg_class_c.timestep()
                    self.qg_class_f.timestep()

            else:

                # re-adjust timestep if over the jump need to reach t
                if self.t + self.dt_c > T:
                    self.qg_class_c.const_dt.assign(T - self.t)
                    self.qg_class_f.const_dt.assign(T - self.t)
                else:
                    self.qg_class_c.const_dt.assign(self.dt_c)
                    self.qg_class_f.const_dt.assign(self.dt_f)

                if self.variance > 0:
                    # update optional ou process
                    dW = (np.sqrt(self.qg_class_c.const_dt.dat.data[0]) *
                          np.random.normal(0, np.sqrt(self.variance), 1)[0])
                    self.qg_class_c._base_class__update_sigma(dW)
                    self.qg_class_f._base_class__update_sigma(dW)

                    # update forcing and carry out time-step
                    self.__reset_coarse_forcing()
                    self.qg_class_f._base_class__update_forcing()
                    self.__coupling_coarse_forcing()
                    self.__normalize_coupled_coarse_forcing()
                    self.qg_class_c._base_class__update_kl_expansion()
                    self.qg_class_f._base_class__update_kl_expansion()

                self.qg_class_c.timestep()
                self.qg_class_f.timestep()

                # update time
                self.t += self.qg_class_c.const_dt.dat.data[0]

                self.__t_c += self.qg_class_c.const_dt.dat.data[0]
                self.__t_f += self.qg_class_f.const_dt.dat.data[0]

                # find velocity
                self.u = [norm(self.qg_class_c.gradperp(self.psi_[0])),
                          norm(self.qg_class_f.gradperp(self.psi_[1]))]
                if ((self.u[0] * self.dt_c) /
                    round(self.qg_class_c.mdx.comm.allreduce(self.qg_class_c.mdx.dat.data_ro.min(),
                                                             MPI.MIN), 8)) > 0.3:
                    print 'Courant number of coarse approx exceeded 0.3'
                if ((self.u[1] * self.dt_f) /
                    round(self.qg_class_f.mdx.comm.allreduce(self.qg_class_f.mdx.dat.data_ro.min(),
                                                             MPI.MIN), 8)) > 0.3:
                    print 'Courant number of coarse approx exceeded 0.3'
