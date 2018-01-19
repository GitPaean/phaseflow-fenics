import phaseflow
import fenics

 
class Benchmark:
 
    def __init__(self):
    
        self.model = None
        
        self.adaptive_goal_integrand = None
        
        self.adaptive_solver_tolerance = 1.e-4
    
        self.output_dir = None
        
        self.end_time = None
        
        self.stop_when_steady = False
        
        self.steady_relative_tolerance = 1.e-4
        
        
    def verify(self):
    
        assert(False)
        
        
    def run(self):
    
        assert(self.model is not None)
        
        solver = phaseflow.core.Solver(
            model = self.model, 
            adaptive_goal_integrand = self.adaptive_goal_integrand, 
            adaptive_solver_tolerance = self.adaptive_solver_tolerance)

        time_stepper = phaseflow.core.TimeStepper(
            solver = solver,
            output_dir = self.output_dir,
            stop_when_steady = self.stop_when_steady,
            steady_relative_tolerance = self.steady_relative_tolerance)
        
        time_stepper.run_until(self.end_time)
            
        self.verify()
    
 
class Cavity(Benchmark):

    def __init__(self, grid_size = 20):
    
        Benchmark.__init__(self)
        
        self.mesh = fenics.UnitSquareMesh(fenics.mpi_comm_world(), grid_size, grid_size)
    
        self.left_wall = "near(x[0],  0.)"
        
        self.right_wall = "near(x[0],  1.)"
        
        self.bottom_wall = "near(x[1],  0.)"
        
        self.top_wall = "near(x[1],  1.)"
        
  
    def verify_horizontal_velocity_at_centerline(self, y, ux, tolerance):
        
        assert(len(y) == len(ux))
        
        x = 0.5
        
        bbt = self.model.mesh.bounding_box_tree()
        
        for i, true_ux in enumerate(ux):
        
            p = fenics.Point(x, y[i])
            
            if bbt.collides_entity(p):
            
                values = self.model.state.solution(p)
                
                ux = values[1]
                
                assert(abs(ux - true_ux) < tolerance)
                
  
class LidDrivenCavity(Cavity):

    def __init__(self, grid_size = 20):
    
        Cavity.__init__(self, grid_size)
        
        fixed_walls = self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        time_step_size = 1.e12
        
        self.end_time = time_step_size
        
        self.model = phaseflow.pure_isotropic.Model(self.mesh,
            initial_values = ("0.", self.top_wall, "0.", "1."),
            boundary_conditions = [
                {"subspace": 1, "location": self.top_wall, "value": (1., 0.)},
                {"subspace": 1, "location": fixed_walls, "value": (0., 0.)}],
            time_step_size = time_step_size,
            liquid_viscosity = 0.01)
            
        p, u, T = fenics.split(self.model.state.solution)
        
        self.output_dir = "output/benchmarks/lid_driven_cavity"
    
    
    def verify(self):
        """ Verify against ghia1982. """
        self.verify_horizontal_velocity_at_centerline(
            y = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000],
            ux = [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, 
                -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000],
            tolerance = 2.e-2)
    
    
class HeatDrivenCavity(Cavity):

    def __init__(self, grid_size = 20):
    
        Cavity.__init__(self, grid_size)
        
        T_hot = 0.5
    
        T_cold = -T_hot
    
        initial_values = ("0.", "0.", "0.",
            "T_hot + x[0]*(T_cold - T_hot)".replace("T_hot", str(T_hot)).replace("T_cold", str(T_cold)))
        
        walls = self.top_wall + " | " + self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall
        
        self.model = phaseflow.pure_isotropic.Model(self.mesh,
            initial_values = initial_values,
            boundary_conditions = [
                {"subspace": 1, "location": walls, "value": (0., 0.)},
                {"subspace": 2, "location": self.left_wall, "value": T_hot},
                {"subspace": 2, "location": self.right_wall, "value": T_cold}],
            buoyancy = phaseflow.pure.IdealizedLinearBoussinesqBuoyancy(
                rayleigh_numer = 1.e6, 
                prandtl_number = 0.71),
            time_step_size = 1.e-3,
            liquid_viscosity = 0.01)
            
        self.output_dir = "output/benchmarks/heat_driven_cavity"
        
        self.end_time = 10.

        self.stop_when_steady = True
        
        self.steady_relative_tolerance = 1.e-4
        
    
    def verify(self):
        """ Verify against the result published in \cite{wang2010comprehensive}. """
        data = {"Ra": 1.e6, "Pr": 0.71, "x": 0.5, 
            "y": [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 
            "ux": [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
        
        bbt = w.function_space().mesh().bounding_box_tree()
        
        for i, true_ux in enumerate(data['ux']):
        
            p = fenics.Point(data['x'], data['y'][i])
            
            if bbt.collides_entity(p):
            
                wval = w(p)
            
                ux = wval[1]*data['Pr']/data['Ra']**0.5
            
                assert(abs(ux - true_ux) < 2.e-2)
    
    
if __name__=='__main__':

    LidDrivenCavity().run()
    
    HeatDrivenCavity().run()
    