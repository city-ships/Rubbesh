import dolfinx
import dolfinx.fem.petsc
import dolfinx.mesh
import dolfinx.plot
import ufl
import numpy as np
import pyvista
from mpi4py import MPI
from petsc4py import PETSc # For PETSc.ScalarType

# MPI communicator
comm = MPI.COMM_WORLD

# 0. Simulation Parameters
# Mesh parameters
cube_min_coord = [0.0, 0.0, 0.0]
cube_max_coord = [1.0, 1.0, 1.0]
num_elements_per_dim = [10, 10, 10] # Number of elements in x, y, z

# Material parameters
E = 1.0e5  # Young's modulus
nu = 0.45   # Poisson's ratio (0.4 is less prone to locking with P1 than 0.49)
# For nu ~ 0.5 (e.g., 0.49 or 0.499), consider P2 elements.
# nu = 0.49

# Lamé parameters
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# Body force
bz_val = 1.0e5  # Magnitude of body force in z-direction
f_body_vec = np.array([bz_val, 0.,0.], dtype=PETSc.ScalarType)

# Finite element degree
element_degree = 1 # 1 for P1 (linear), 2 for P2 (quadratic)

# Visualization parameters
displacement_magnification_factor = 1
# 1. Mesh Generation
domain = dolfinx.mesh.create_box(
    comm,
    [np.array(cube_min_coord, dtype=PETSc.ScalarType),
     np.array(cube_max_coord, dtype=PETSc.ScalarType)],
    num_elements_per_dim,
    cell_type=dolfinx.mesh.CellType.tetrahedron,
    ghost_mode=dolfinx.mesh.GhostMode.none # Or .shared_facet if processing in parallel and needing shared facet info
)
tdim = domain.topology.dim # Topological dimension (3 for 3D)
fdim = tdim - 1         # Facet dimension

# 2. Define Function Space
# VectorFunctionSpace for displacement (3 components)
V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", element_degree))

# 3. Define Boundary Conditions
# Fix the face at x = 0
def fixed_boundary_x0(x):
    return np.isclose(x[0], cube_min_coord[0])

fixed_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, fixed_boundary_x0)
fixed_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, fixed_facets)

# Define the zero displacement value for the BC
u_D_val = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
u_D = dolfinx.fem.Constant(domain, u_D_val) # Use fem.Constant for BC values
bc = dolfinx.fem.dirichletbc(u_D, fixed_dofs, V)

# 4. Define Trial and Test Functions
u = ufl.TrialFunction(V)  # Trial function (for solution)
v = ufl.TestFunction(V)   # Test function (for variational form)

# 5. Define Body Force and Material Tensors
f_body = dolfinx.fem.Constant(domain, f_body_vec) # Volumetric body force

# Strain tensor (epsilon)
def epsilon(u_func):
    return 0.5 * (ufl.grad(u_func) + ufl.grad(u_func).T)

# Stress tensor (sigma) for isotropic material
def sigma(u_func):
    return lambda_ * ufl.tr(epsilon(u_func)) * ufl.Identity(tdim) + 2 * mu * epsilon(u_func)

# 6. Define Variational Problem (Weak Form)
# Bilinear form a(u, v) = ∫ (sigma(u) : epsilon(v)) dV
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
# Linear form L(v) = ∫ (f_body ⋅ v) dV
L = ufl.dot(f_body, v) * ufl.dx

# 7. Solve the Linear Variational Problem
# Create a Function to store the solution
uh = dolfinx.fem.Function(V)
uh.name = "Displacement"

# Set up the linear problem
problem = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=[bc], u=uh,
    petsc_options={
        "ksp_type": "preonly",  # Direct solver (LU) often via "preonly" and "pc_type": "lu"
        "pc_type": "lu",        # Good for smaller problems
        "pc_factor_mat_solver_type": "mumps" # MUMPS if available, otherwise PETSc default
    }
)
# For larger problems or iterative solvers:
# problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], u=uh,
# petsc_options={"ksp_type": "cg", "pc_type": "gamg", "pc_gamg_type": "agg",
# "pc_gamg_coarse_eq_limit": 1000, "mg_levels_ksp_type": "chebyshev",
# "mg_levels_pc_type": "jacobi", "ksp_rtol": 1e-8})

print("Solving the linear system...")
problem.solve()
print("Solved.")

# 8. Visualization using PyVista
# pyvista.start_xvfb() # For headless environments

# Create a PyVista plotter
plotter = pyvista.Plotter(window_size=[1600, 900])
# For some backends or issues, explicitly setting off_screen=False might be needed
# plotter = pyvista.Plotter(window_size=[800, 600], off_screen=False)


# --- Corrected way to get mesh for PyVista ---
mesh = V.mesh
try:
    from dolfinx.plot import vtk_mesh
    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
except (ImportError, AttributeError):
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    cells_flat = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(num_cells, -1)
    num_vertices_per_cell = cells_flat.shape[1]
    cells_pv = np.hstack((np.full((num_cells, 1), num_vertices_per_cell), cells_flat)).ravel()
    cell_types_pv = np.full(num_cells, pyvista.CellType.TETRA, dtype=np.uint8)
    topology = cells_pv
    cell_types = cell_types_pv
    geometry = mesh.geometry.x

original_mesh_pv = pyvista.UnstructuredGrid(topology, cell_types, geometry)

if V.ufl_element().degree == 1:
    num_dofs_per_node_V = V.dofmap.index_map_bs
    num_mesh_nodes = mesh.geometry.x.shape[0]
    displacements_at_vertices = uh.x.array.reshape(num_mesh_nodes, num_dofs_per_node_V)
else:
    V_plot = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    u_plot = dolfinx.fem.Function(V_plot)
    u_plot.interpolate(uh)
    num_dofs_per_node_V_plot = V_plot.dofmap.index_map_bs
    num_mesh_nodes = mesh.geometry.x.shape[0]
    displacements_at_vertices = u_plot.x.array.reshape(num_mesh_nodes, num_dofs_per_node_V_plot)

# --- DEBUG: Check displacement values ---
print(f"Displacements at vertices (first 5): \n{displacements_at_vertices[:5, :]}")
print(f"Norm of displacements (first 5): \n{np.linalg.norm(displacements_at_vertices[:5, :], axis=1)}")
max_disp_vec = displacements_at_vertices[np.argmax(np.linalg.norm(displacements_at_vertices, axis=1)),:]
print(f"Max displacement vector: {max_disp_vec}, Magnitude: {np.linalg.norm(max_disp_vec)}")
max_z_disp_val = np.max(displacements_at_vertices[:, 2])
min_z_disp_val = np.min(displacements_at_vertices[:, 2])
print(f"Raw Max displacement in z (at vertices): {max_z_disp_val:.3e}")
print(f"Raw Min displacement in z (at vertices): {min_z_disp_val:.3e}")
# --- END DEBUG ---

original_mesh_pv.point_data["u_original_vertices"] = displacements_at_vertices

warped_mesh_pv = original_mesh_pv.copy()
# Ensure displacements are actually being added:
displacement_vectors_scaled = displacements_at_vertices * displacement_magnification_factor
warped_mesh_pv.points = original_mesh_pv.points + displacement_vectors_scaled

# --- DEBUG: Check warped coordinates ---
print(f"Original points (first 2): \n{original_mesh_pv.points[:2, :]}")
print(f"Displacement vectors scaled (first 2): \n{displacement_vectors_scaled[:2, :]}")
print(f"Warped points (first 2): \n{warped_mesh_pv.points[:2, :]}")
# --- END DEBUG ---

# Store the unscaled displacements on the warped mesh for consistent coloring
warped_mesh_pv.point_data["u_vertex_displacements"] = displacements_at_vertices


# --- Plotting ---
# Add original mesh (wireframe or translucent)
plotter.add_mesh(original_mesh_pv, style="wireframe", color="gray", opacity=0.3, label="Original")

# Add deformed mesh
# Ensure 'u_vertex_displacements' exists and has the correct shape for scalars
plotter.add_mesh(warped_mesh_pv,
                 scalars=warped_mesh_pv.point_data["u_vertex_displacements"][:, 2], # Color by Z-component of actual displacement
                 show_edges=True,
                 label="Deformed",
                 cmap="viridis",
                 scalar_bar_args={'title': "Z-Displacement (actual)"})

plotter.add_text(f"Deformed Cube (Displacement x{displacement_magnification_factor})", position="upper_edge", font_size=10)

      
# # --- Explicit Camera Control ---
# # Calculate a sensible focal point (e.g., center of the original undeformed mesh)
# center_original = np.array(original_mesh_pv.center) # Ensure center is a numpy array
# plotter.camera.focal_point = center_original.tolist() # PyVista camera often prefers list/tuple

# # Position camera at a distance along a vector from the focal point
# # The distance can be estimated from the bounds of the original mesh
# bounds = np.array(original_mesh_pv.bounds) # Convert bounds tuple to numpy array
# min_coords = bounds[0::2]  # (xmin, ymin, zmin)
# max_coords = bounds[1::2]  # (xmax, ymax, zmax)
# diagonal_vector = max_coords - min_coords
# distance = np.linalg.norm(diagonal_vector) * 1.5 # 1.5x diagonal length

# # Set camera position (example: along a vector from center)
# # Using a simple offset for isometric-like view
# camera_pos_offset = distance / np.sqrt(3) # Equal offset in x, y, z for a common isometric angle
# plotter.camera.position = (center_original[0] + camera_pos_offset,
                           # center_original[1] + camera_pos_offset,
                           # center_original[2] + camera_pos_offset)

# # Alternative: a common viewing direction vector normalized
# # view_vector = np.array([1, 1, 1]) / np.linalg.norm(np.array([1,1,1]))
# # plotter.camera.position = tuple(center_original + view_vector * distance)

# plotter.reset_camera_clipping_range() # Important after setting position/focal point

    

# plotter.view_isometric() # Overrides explicit camera settings above, use one or the other
#plotter.enable_zoom_style()
plotter.add_axes()

print(f"Displaying original and deformed mesh. Magnification: {displacement_magnification_factor}x.")
print(f"Magnified Max displacement in z (at vertices): {max_z_disp_val * displacement_magnification_factor:.2e}")
print(f"Magnified Min displacement in z (at vertices): {min_z_disp_val * displacement_magnification_factor:.2e}")

plotter.show()
