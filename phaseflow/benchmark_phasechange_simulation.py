""" **benchmark_phasechange_simulation.py** applies the phase-change model to a variety of benchmark problems. """
import fenics
import phaseflow


class ConvectionCoupledMeltingBenchmarkSimulation(phaseflow.phasechange_simulation.PhaseChangeSimulation):    
    
    def __init__(self):
    
        self.timestep_size = 10.
        
        self.hot_wall_temperature = 1.
        
        self.cold_wall_temperature = -0.01
        
        self.initial_hot_wall_refinement_cycles = 6
        
        class HotWall(fenics.SubDomain):
    
            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[0], 0.)
                
        self.hot_wall = HotWall()
        
        class ColdWall(fenics.SubDomain):
    
            def inside(self, x, on_boundary):

                return on_boundary and fenics.near(x[0], 1.)
                
        self.cold_wall = ColdWall()
        
        class Walls(fenics.SubDomain):
        
            def inside(self, x, on_boundary):

                return on_boundary
                
        self.walls = Walls()
        
        super().__init__()
        
        self.prandtl_number = 56.2
        
        self.rayleigh_number = 3.27e5
        
        self.stefan_number = 0.045
        
        self.regularization_central_temperature = 0.01
        
        self.regularization_smoothing_parameter = 0.025
        
        self.adaptive_goal_tolerance = 4.e-5
        
        self.quadrature_degree = 8
        
    @property
    def coarse_mesh(self):
    
        return fenics.UnitSquareMesh(1, 1)
        
    @property
    def initial_mesh(self):
        
        mesh = phaseflow.refine.refine_mesh_near_subdomain(
            self.coarse_mesh, 
            self.hot_wall, 
            refinement_cycles = self.initial_hot_wall_refinement_cycles)
            
        return mesh
        
    @property
    def initial_states(self):
        
        initial_melt_thickness = 1./2.**(self.initial_hot_wall_refinement_cycles - 1)

        function = fenics.interpolate(
            fenics.Expression(("0.", "0.", "0.", "(T_h - T_c)*(x[0] < x_m0) + T_c"),
                T_h = self.hot_wall_temperature, 
                T_c = self.cold_wall_temperature,
                x_m0 = initial_melt_thickness,
                element = self.element),
            self.function_space)
            
        return self.State(solution = function, time = 0.)
        
    @property
    def boundary_conditions(self):
        
        W = self.function_space
        
        W_u = W.sub(1)

        W_T = W.sub(2)

        return [
            fenics.DirichletBC(W_u, (0., 0.), self.walls),
            fenics.DirichletBC(W_T, self.hot_wall_temperature, self.hot_wall),
            fenics.DirichletBC(W_T, self.cold_wall_temperature, self.cold_wall)]
        
    @property
    def adaptive_goal_form(self):
        """ Redefine this to return a `fenics.NonlinearVariationalForm`. """
        u_t, T_t, phi_t = self.make_time_discrete_terms()
        
        dx = self.integration_measure
        
        return phi_t*dx
    
    def run(self, timesteps = 1):
        
        for it in range(timesteps):
        
            self.step_to_time(self.solved_states[0].time + (it + 1)*self.timestep_size)
            