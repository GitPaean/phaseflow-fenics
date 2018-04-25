""" **phasechange_simulation.py** implements the convection-coupled melting of phase-change materials. """
import fenics
import phaseflow


class BinaryAlloySolidificationSimulation(phaseflow.simulation.Simulation):

    def __init__(self):
        """ This extends the `__init__` method with attributes for the convection-coupled phase-change model. """
        phaseflow.simulation.Simulation.__init__(self)
        
        self.timestep_size = 1.
        
        self.temperature_rayleigh_number = 1.
        
        self.concentration_rayleigh_number = 1.
        
        self.stefan_number = 1.
        
        self.prandtl_number = 1.
        
        self.lewis_number = 1.
        
        self.slope_of_ideal_liquidus = -1.
        
        self.gravity = (0., -1.)
        
        self.liquid_viscosity = 1.
        
        self.solid_viscosity = 1.e8
        
        self.solute_diffusivity = 1.
        
        self.penalty_parameter = 1.e-7
        
        self.temperature_element_degree = 1
        
        
    def setup_element(self):
        """ Implement the mixed element per @cite{danaila2014newton}. """
        pressure_element = fenics.FiniteElement("P", 
            self.mesh.ufl_cell(), self.temperature_element_degree)
        
        velocity_element = fenics.VectorElement("P", 
            self.mesh.ufl_cell(), self.temperature_element_degree + 1)

        temperature_element = fenics.FiniteElement(
            "P", self.mesh.ufl_cell(), self.temperature_element_degree)
        
        phase_element = fenics.FiniteElement(
            "P", self.mesh.ufl_cell(), self.temperature_element_degree)
        
        self.element = fenics.MixedElement([pressure_element, velocity_element, temperature_element,
            phase_element])
        
        
    def make_buoyancy_function(self):

        Pr = fenics.Constant(self.prandtl_number)
        
        Le = fenics.Constant(self.lewis_number)
        
        Ra_T = fenics.Constant(self.temperature_rayleigh_number)
        
        Ra_C = fenics.Constant(self.concentration_rayleigh_number)
        
        g = fenics.Constant(self.gravity)
        
        m = fenics.Constant(self.slope_of_ideal_liquidus)
        
        def f_B(T):
            """ Idealized linear Boussinesq Buoyancy with $Re = 1$ """
            C = T/m
            
            return (T*Ra_T + C*Ra_C/Le)/Pr*g
            
            
        return f_B
        
    
    def apply_time_discretization(self, Delta_t, u):
    
        u_t = phaseflow.backward_difference_formulas.apply_backward_euler(Delta_t, u)
        
        return u_t
    
    
    def make_time_discrete_terms(self):
    
        p_np1, u_np1, T_np1, phi_np1 = fenics.split(self.state.solution)
        
        p_n, u_n, T_n, phi_n = fenics.split(self.old_state.solution)
        
        u = [u_np1, u_n]
        
        T = [T_np1, T_n]
        
        phi = [phi_np1, phi_n]
        
        if self.second_order_time_discretization:
            
            p_nm1, u_nm1, T_nm1, phi_nm1 = fenics.split(self.old_old_state.solution)
            
            u.append(u_nm1)
            
            T.append(T_nm1)
            
            phi.append(phi_nm1)
        
        if self.second_order_time_discretization:
        
            Delta_t = [self.fenics_timestep_size, self.old_fenics_timestep_size]
            
        else:
        
            Delta_t = self.fenics_timestep_size
        
        u_t = self.apply_time_discretization(Delta_t, u)
        
        T_t = self.apply_time_discretization(Delta_t, T)
        
        phi_t = self.apply_time_discretization(Delta_t, phi)
        
        return u_t, T_t, phi_t
    
    
    def setup_governing_form(self):
        """ Implement the variational form per @cite{zimmerman2018monolithic}. """
        Pr = fenics.Constant(self.prandtl_number)
        
        Ste = fenics.Constant(self.stefan_number)
        
        f_B = self.make_buoyancy_function()
        
        mu_L, mu_S = fenics.Constant(self.liquid_viscosity), fenics.Constant(self.solid_viscosity)
        
        gamma = fenics.Constant(self.penalty_parameter)
        
        Le = fenics.Constant(self.lewis_number)
        
        D =  fenics.Constant(self.solute_diffusivity)
        
        p, u, T, phi = fenics.split(self.state.solution)
        
        u_t, T_t, phi_t = self.make_time_discrete_terms()
        
        psi_p, psi_u, psi_T, psi_phi = fenics.TestFunctions(self.function_space)
        
        dx = self.integration_metric
        
        inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
        """ Mass """
        self.governing_form = (
            -psi_p*(div(u) + gamma*p)
            )*dx
        
        """ Momentum """        
        self.governing_form += (
            dot(psi_u, u_t + dot(grad(u), u) + f_B(T))
            - div(psi_u)*p
            + 2.*inner(sym(grad(psi_u)), (mu_S + phi*(mu_L - mu_S))*sym(grad(u)))
            )*dx
            
        """ Enthalpy """
        self.governing_form += (
            psi_T*(T_t - 1./Ste*phi_t)
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
            )*dx
            
        """ Solute concentration """
        self.governing_form += (
            psi_phi*((1. - phi)*T_t - 1./Ste*phi_t)
            + dot(grad(psi_phi), 1./(Pr*Le)*(1. - phi)*D*grad(T) - T*u)
            )*dx
        
        