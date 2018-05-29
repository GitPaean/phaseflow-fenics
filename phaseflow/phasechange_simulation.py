""" **phasechange_simulation.py** implements the convection-coupled melting of phase-change materials. """
import fenics
import phaseflow
import matplotlib


class State(phaseflow.state.State):
    
    def write_solution(self, file):
        """ Write the solution to a file.

        Parameters
        ----------
        file : phaseflow.helpers.SolutionFile

            This method should have been called from within the context of the open `file`.
            
        solved_state_index: int
        
            Index into self.solved_states. Zero by default, since usually we want to write the latest solution.
        """
        phaseflow.helpers.print_once("Writing solution to " + str(file.path))

        p, u, T = self.solution.leaf_node().split()

        p.rename("p", "pressure")

        u.rename("u", "velocity")

        T.rename("T", "temperature")
        
        for var in [p, u, T]:

            file.write(var, self.time)
            
    def plot(self):
    
        p, u, T = fenics.split(self.solution.leaf_node())
        
        for var, label in zip((u, T), ("$u$", "$T$", "$\Omega_h$")):
        
            fenics.plot(var)
            
            fenics.plot(self.solution.function_space().mesh().leaf_node())

            matplotlib.pyplot.title(label + ", $t = " + str(self.time) + "$")

            matplotlib.pyplot.xlabel("$x$")

            matplotlib.pyplot.ylabel("$y$")
        
            matplotlib.pyplot.show()
            

class PhaseChangeSimulation(phaseflow.simulation.Simulation):

    def __init__(self):
        
        super().__init__(State=State)
        
        self.rayleigh_number = fenics.Constant(1.)
        
        self.prandtl_number = fenics.Constant(1.)
        
        self.stefan_number = fenics.Constant(1.)
        
        self.gravity_direction = fenics.Constant((0., -1.))
        
        self.liquid_viscosity = fenics.Constant(1.)
        
        self.solid_viscosity = fenics.Constant(1.e8)
        
        self.pressure_penalty_factor = fenics.Constant(1.e-7)
        
        self.regularization_central_temperature = fenics.Constant(0.)
        
        self.regularization_smoothing_parameter = fenics.Constant(0.01)
        
    @property
    def element(self):
        
        P1 = fenics.FiniteElement('P', self.mesh.ufl_cell(), 1)
        
        P2 = fenics.VectorElement('P', self.mesh.ufl_cell(), 2)
        
        return fenics.MixedElement([P1, P2, P1])
    
    @property
    def buoyancy(self):
        """ Idealized linear Boussinesq buoyancy """
        p, u, T = fenics.split(self.solving_state.solution)
        
        Pr, Ra, g = self.prandtl_number, self.rayleigh_number, self.gravity_direction
        
        return T*Ra*g/Pr
        
    def make_solid_volume_fraction_function(self):
        """ Regularized solid volume fraction """
        T_r, r = self.regularization_central_temperature, self.regularization_smoothing_parameter
        
        tanh = fenics.tanh
        
        def phi(T):
        
            return 0.5*(1. + tanh((T_r - T)/r))
        
        return phi
        
    def make_time_discrete_terms(self):
        
        Delta_t = fenics.Constant(self.solving_state.time - self.solved_states[0].time)
        
        p, u, T = fenics.split(self.solving_state.solution)
        
        p_n, u_n, T_n = fenics.split(self.solved_states[0].solution)
    
        u_t = (u - u_n)/Delta_t
        
        T_t = (T - T_n)/Delta_t
        
        phi = self.make_solid_volume_fraction_function()
        
        phi_t = (phi(T) - phi(T_n))/Delta_t
        
        return u_t, T_t, phi_t
        
    @property
    def governing_form(self):
        """ Implement the variational form per @cite{zimmerman2018monolithic}. """
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        mu_L = self.liquid_viscosity
        
        mu_S = self.solid_viscosity
        
        phi = self.make_solid_volume_fraction_function()
        
        f_B = self.buoyancy
        
        p, u, T = fenics.split(self.solving_state.solution)
        
        mu = mu_L + (mu_S - mu_L)*phi(T)
        
        gamma = self.pressure_penalty_factor
        
        u_t, T_t, phi_t = self.make_time_discrete_terms()
        
        psi_p, psi_u, psi_T = fenics.TestFunctions(self.function_space)
        
        dx = self.integration_measure
        
        inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
        mass = -psi_p*div(u)
        
        momentum = dot(psi_u, u_t + f_B + dot(grad(u), u)) \
            - div(psi_u)*p \
            + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))
            
        enthalpy = psi_T*(T_t - 1./Ste*phi_t) \
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
            
        stabilization = -gamma*psi_p*p
        
        F = (mass + momentum + enthalpy + stabilization)*dx
            
        return F
    
    def plot_phi(self, state):
        
        p, u, T = fenics.split(state.solution.leaf_node())
        
        phi = self.make_solid_volume_fraction_function()
        
        fenics.plot(phi(T))

        matplotlib.pyplot.title(label + ", $t = " + str(state.time) + "$")

        matplotlib.pyplot.xlabel("$x$")

        matplotlib.pyplot.ylabel("$y$")
    
        matplotlib.pyplot.show()
        