""" This module runs the benchmark test suite. """
from .context import phaseflow
import fenics


def test_convection_coupled_phasechange_benchmark():

    sim = phaseflow.benchmark_phasechange_simulation.ConvectionCoupledMeltingBenchmarkSimulation()
    
    sim.run(timesteps = 3)
    