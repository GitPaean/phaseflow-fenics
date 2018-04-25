""" This module runs the benchmark test suite. """
from .context import phaseflow
import fenics


BaseSimulation = phaseflow.binary_alloy_solidification_simulation.BinaryAlloySolidificationSimulation

class FreezingSaltWaterFromAboveSimulation(BaseSimulation):
    
    def __init__(self):
        
        BaseSimulation.__init__(self)
        
        self.output_dir += "freezing_salt_water_cavity_from_above/"
        
        self.initial_liquid_temperature = 1.
        
        self.cold_wall_temperature = -0.01
        
        self.stefan_number = 0.125  # Water freezing benchmark
        
        self.prandtl_number = 7.  # Water freezing benchmark
        
        self.lewis_number = 80.  # Typical for sea ice (Worster)
        
        self.solute_diffusivity = 1.  # @todo Double check scaling .
        
        self.slope_of_ideal_liquidus = -1.  # @todo Write down the scaling and compute correct slope.
        
        self.temperature_rayleigh_number = 2.e6  # Water freezing benchmark
        
        self.concentration_rayleigh_number = 1.e6
        
        self.gravity = (0., -1.)
        
        self.solid_viscosity = 1.e8
        
        self.liquid_viscosity = 1.
        
        self.end_time = 6.
        
        self.timestep_size = 2.
        
        self.initial_cold_wall_refinement_cycles = 6
        
        self.initial_mesh_size = (1, 1)
        
        self.initial_pci_position = None
        
        self.adaptive_goal_tolerance = 1.e-4
        
        self.xmin = 0.
        
        self.ymin = 0.
        
        self.xmax = 1.
        
        self.ymax = 1.
        
        
    def validate_attributes(self):
        
        if type(self.initial_mesh_size) is type(20):
        
            self.initial_mesh_size = (self.initial_mesh_size, self.initial_mesh_size)
        
        
    def setup_derived_attributes(self):
        
        BaseSimulation.setup_derived_attributes(self)
        
        self.left_wall = "near(x[0],  xmin)".replace("xmin", str(self.xmin))
        
        self.right_wall = "near(x[0],  xmax)".replace("xmax", str(self.xmax))
        
        self.bottom_wall = "near(x[1],  ymin)".replace("ymin",  str(self.ymin))
        
        self.top_wall = "near(x[1],  ymax)".replace("ymax", str(self.ymax))
        
        self.walls = \
                self.top_wall + " | " + self.bottom_wall + " | " + self.left_wall + " | " + self.right_wall

        T_initial_liquid = fenics.Constant(self.initial_liquid_temperature)
        
        T_cold = fenics.Constant(self.cold_wall_temperature)
        
        self.boundary_conditions = [
                {"subspace": 1, "location": self.walls, "value": (0., 0.)},
                {"subspace": 2, "location": self.top_wall, "value": T_cold}]
                
        if self.initial_pci_position == None:
            """ Set the initial PCI position such that the melted area is covered by one layer of cells. """
            initial_pci_position = \
                1. - 1./float(self.initial_mesh_size[1])/2.**(self.initial_cold_wall_refinement_cycles - 1)
        
        else:
        
            initial_pci_position = 0. + self.initial_pci_position
        
        initial_temperature = "(T_cold - T_liquid_initial)*(x[1] > initial_pci_position) + T_liquid_initial"
        
        initial_temperature = initial_temperature.replace("initial_pci_position", str(initial_pci_position))
        
        initial_temperature = initial_temperature.replace(
            "T_liquid_initial", str(self.initial_liquid_temperature))
        
        initial_temperature = initial_temperature.replace("T_cold", str(self.cold_wall_temperature))
        
        self.initial_temperature = initial_temperature
        
        self.initial_phase = "(x[1] > initial_pci_position)".replace(
            "initial_pci_position", str(initial_pci_position))
        
        
    def setup_coarse_mesh(self):
        """ This creates the rectangular mesh """    
        self.mesh = fenics.RectangleMesh(fenics.mpi_comm_world(), 
            fenics.Point(self.xmin, self.ymin), fenics.Point(self.xmax, self.ymax),
            self.initial_mesh_size[0], self.initial_mesh_size[1], "crossed")

    
    def refine_initial_mesh(self):
        """ Refine near the hot wall. """
        ymax = self.ymax
        
        class ColdWall(fenics.SubDomain):
    
            def inside(self, x, on_boundary):
            
                return on_boundary and fenics.near(x[1], ymax)

    
        cold_wall = ColdWall()
        
        for i in range(self.initial_cold_wall_refinement_cycles):
            
            edge_markers = fenics.MeshFunction("bool", self.mesh, 1, False)
            
            cold_wall.mark(edge_markers, True)

            fenics.adapt(self.mesh, edge_markers)
        
            self.mesh = self.mesh.child()
            
            
    def setup_initial_values(self):
        """ Set the initial values. """
        self.old_state.interpolate(("0.", "0.", "0.", self.initial_temperature, self.initial_phase))
        
        
    def setup_adaptive_goal_form(self):
        
        p, u, T, phi  = fenics.split(self.state.solution)
        
        self.adaptive_goal_form = phi*self.integration_metric
        
       

def test_freezing_salt_water_from_above():

    FreezingSaltWaterFromAboveSimulation().run()
    
    