""" **refine.py** defines functions for mesh refinement. """
import fenics


def refine_mesh_near_subdomain(mesh, subdomain, refinement_cycles = 1):
    
    refined_mesh = fenics.Mesh(mesh)
    
    for cycle in range(refinement_cycles):

        edge_markers = fenics.MeshFunction("bool", refined_mesh, 1, False)

        subdomain.mark(edge_markers, True)

        fenics.adapt(refined_mesh, edge_markers)

        refined_mesh = refined_mesh.child()

    return refined_mesh
    