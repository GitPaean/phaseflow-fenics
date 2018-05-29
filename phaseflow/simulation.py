""" **simulation.py** contains the Simulation class. """
import fenics
import h5py
import collections
import abc

class Simulation(metaclass = abc.ABCMeta):
    """ This is an abstract class for time-dependent simulations using FEniCS 
    with mixed finite elements and goal-oriented adaptive mesh refinement. 
    """
    def __init__(self, State = None, number_of_solved_states_to_store = 2):
    
        if State is None:
        
            State = phaseflow.state.State
            
        self.State = State    
        
        """ The degree of the quadrature rule used for numerical integration. 
        
        If `self.quadrature_degree = None`, then the exact quadrature rule will be used.
        """
        self._quadrature_degree = None
        
        """ A copy of the coarse mesh which will not be affected by adaptive mesh refinement """
        self._coarse_mesh = self.coarse_mesh
        
        self._mesh = self.initial_mesh
        
        self._function_space = fenics.FunctionSpace(self.mesh, self.element)
        
        self._solving_state = self.State(fenics.Function(self.function_space))
        
        self._solved_states = (self.State(fenics.Function(self.function_space)),)*number_of_solved_states_to_store
        
        initial_states = self.initial_states
        
        """ Ensure that initial_states is iterable. """
        try:
        
            iterator = iter(initial_states)
        
        except TypeError as type_error:
            
            initial_states = (initial_states,)
    
        for solved_state, initial_state in zip(self.solved_states, initial_states):
        
            solved_state.set_from_other_instance(initial_state)
        
        self.solving_state.set_from_other_instance(self.solved_states[0])
        
        self.adaptive_goal_tolerance = 1.e12
        
        self.nonlinear_solver_max_iterations = 50
        
        self.nonlinear_solver_absolute_tolerance = 1.e-10
        
        self.nonlinear_solver_relative_tolerance = 1.e-9
        
        self.nonlinear_solver_relaxation = 1.
        
    @property
    @abc.abstractmethod
    def coarse_mesh(self):
        """ Redefine this to return a `fenics.Mesh`. """
        pass
        
    @property
    @abc.abstractmethod
    def element(self):
        """ Redefine this to return a `fenics.MixedElement`. """
        pass
    
    @property
    @abc.abstractmethod
    def governing_form(self):
        """ Redefine this to return a `fenics.NonlinearVariationalForm`. """
        pass
        
    @property
    @abc.abstractmethod
    def initial_states(self):
        """ Redefine this to return a list of states required for the discrete initial value problem. """
        pass
        
    @property
    @abc.abstractmethod
    def boundary_conditions(self):
        """ Redefine this to return a list of `fenics.DirichletBC`. """
        pass
        
    @property
    @abc.abstractmethod
    def adaptive_goal_form(self):
        """ Redefine this to return a `fenics.NonlinearVariationalForm`. """
        pass
    
    @property
    def initial_mesh(self):
        """ Redefine this to return a manually refined mesh before adaptive mesh refinement. """
        return self.coarse_mesh
    
    @property
    def mesh(self):
    
        return self._mesh
        
    @property
    def function_space(self):
    
        return self._function_space
         
    @property
    def solving_state(self):
    
        return self._solving_state
    
    @property
    def solved_states(self):
    
        return self._solved_states
    
    @property
    def quadrature_degree(self):
    
        return self._quadrature_degree
    
    @mesh.setter
    def mesh(self, value):
    
        self._mesh = value
        
    @function_space.setter
    def function_space(self, value):
        
        self._function_space = value
        
    @solving_state.setter
    def solving_state(self, value):
    
        assert(type(value) is type(self.solving_state))
        
        self._solving_state.set_from_other_instance(value)
    
    @quadrature_degree.setter
    def quadrature_degree(self, value):
    
        self._quadrature_degree = value
        
        if value is None:
        
            self.integration_measure = fenics.dx
        
        else:
        
            self.integration_measure = fenics.dx(metadata={'quadrature_degree': value})
    
    def solve(self):
        """ Set up the problem and solver, and solve the problem. """
        JF = fenics.derivative(
            self.governing_form,
            self.solving_state.solution, 
            fenics.TrialFunction(self.function_space))
        
        problem = fenics.NonlinearVariationalProblem( 
            self.governing_form,
            self.solving_state.solution, 
            self.boundary_conditions, 
            JF)
        
        solver = fenics.AdaptiveNonlinearVariationalSolver(
            problem = problem,
            goal = self.adaptive_goal_form)
            
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["maximum_iterations"] = self.nonlinear_solver_max_iterations
    
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["absolute_tolerance"] = self.nonlinear_solver_absolute_tolerance
        
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["relative_tolerance"] = self.nonlinear_solver_relative_tolerance
        
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["relaxation_parameter"] = self.nonlinear_solver_relaxation
            
        solver.solve(self.adaptive_goal_tolerance)
    
    def store_solved_state(self):
        """ Here we're manually managing something like a FIFO queue. """
        for i in range(1, len(self.solved_states)):
        
            self.solved_states[i].set_from_other_instance(self.solved_states[i - 1])
            
        self.solved_states[0].set_from_other_instance(self.solving_state)
    
    def step_to_time(self, time):
    
        assert(time > self.solved_states[0].time)
        
        self.solving_state.time = time
        
        self.solve()
        
        self.store_solved_state()
        
    def write_checkpoint(self, checkpoint_dirpath):
        """Write states to a checkpoint file. """
        checkpoint_filepath = checkpoint_dirpath + "checkpoint_t" + str(self.solved_states[0].time) + ".h5"
        
        self.latest_checkpoint_filepath = checkpoint_filepath
        
        phaseflow.helpers.print_once("Writing checkpoint file to " + checkpoint_filepath)
        
        with fenics.HDF5File(fenics.mpi_comm_world(), checkpoint_filepath, "w") as h5:
            
            h5.write(self.solved_states[0].solution.function_space().mesh().leaf_node(), "mesh")
        
            for i in range(len(self.solved_states)):
            
                h5.write(self.solved_states[i].solution.leaf_node(), "solution" + str(i))
            
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            with h5py.File(checkpoint_filepath, "r+") as h5:
            
                for i in range(len(self.solved_states)):
                
                    h5.create_dataset("time" + str(i), data = self.solved_states[i].time)
    
    def read_checkpoint(self, checkpoint_filepath):
        """Read states from a checkpoint file. """
        phaseflow.helpers.print_once("Reading checkpoint file from " + checkpoint_filepath)
        
        self.mesh = fenics.Mesh()
            
        with fenics.HDF5File(self.mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
        
            h5.read(self.mesh, "mesh", True)
        
        with fenics.HDF5File(self.mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
        
            for i in range(len(self.solved_states)):
            
                h5.read(self.solved_states[i].solution, "solution" + str(i))
            
        with h5py.File(checkpoint_filepath, "r") as h5:
                
            for i in range(len(self.solved_states)):
            
                self.solved_states[i].time = h5["time" + str(i)].value
        
        self.restarted = True
        
    def coarsen(self,
            absolute_tolerance = 1.e-3,
            maximum_refinement_cycles = 6,
            scalar_solution_component_index = 3):
        """ Attempt to create a new coarser mesh which meets the error tolerance.
        
        If the tolerance is met, then `self.mesh` will be set with the coarsened mesh.
        
        Returns the coarsened mesh.
        """
        new_mesh = fenics.Mesh(self.coarse_mesh)
        
        for cycle in range(maximum_refinement_cycles - 1):
        
            new_function_space = fenics.FunctionSpace(new_mesh, self.element)
        
            new_states = [fenics.project(state.solution.leaf_node(), new_function_space)
                for state in self.solved_states]
            
            exceeds_tolerance = fenics.MeshFunction("bool", new_mesh.leaf_node(), self.mesh.topology().dim(), False)

            exceeds_tolerance.set_all(False)
        
            for cell in fenics.cells(new_mesh.leaf_node()):
                
                for i in range(len(self.solved_states)):
                    
                    coarsened_value = new_states[i].solution.leaf_node()(cell.midpoint())\
                        [self.coarsening_scalar_solution_component_index]
                    
                    fine_value = self.solved_states[i].solution.leaf_node()(cell.midpoint())\
                        [self.coarsening_scalar_solution_component_index]
                
                    if (abs(coarsened_value - fine_value) > self.coarsening_absolute_tolerance):
                
                        exceeds_tolerance[cell] = True
                        
                        break
                
            if any(exceeds_tolerance):
                
                new_mesh = fenics.refine(new_mesh, exceeds_tolerance)
                
            else:
                
                break
        
        if not any(exceeds_tolerance):
        
            self.mesh = new_mesh
            
            self.function_space = new_function_space
            
            self.solved_states = new_states
            
            self.solving_state = self.State(fenics.Function(new_function_space), new_states[0].time)
        
        return new_mesh
        