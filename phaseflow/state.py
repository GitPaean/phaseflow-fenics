""" **state.py** contains the State class. """
import fenics
import phaseflow.helpers


class State:
    """ Contain a time-dependent `solution` which is a `fenics.Function` and an associated time,
    along with associated methods, e.g. for interpolating mathematical expressions onto the solution,
    and writing the solution to a file.
    
    References to the function space and element are saved as attributes, since they are needed for the
    `self.interpolate` method.
    
    Parameters
    ----------
    solution : fenics.Function
    
    time : float
    """
    def __init__(self, solution, time = 0.):
        
        self._solution = solution
        
        self.time = time
        
    @property
    def solution(self):
    
        return self._solution
        
    @solution.setter
    def solution(self, value):
    
        self._solution = value.copy(deepcopy = True)
        
    def copy(self, deepcopy = False):
        
        return type(self)(self.solution.copy(deepcopy = True), 0. + self.time)
        
    def write_solution(self, file):
        """ Write the solution to a file.

        Parameters
        ----------
        file : phaseflow.helpers.SolutionFile

            This method should have been called from within the context of the open `file`.
        """
        phaseflow.helpers.print_once("Writing solution to " + str(file.path))

        for var in self.solution.leaf_node().split():

            file.write(var, self.time)
        