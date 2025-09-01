import dolfinx
import dolfinx.fem.petsc
import dolfinx.mesh
# dolfinx.plot is not explicitly used in the latest provided script for mesh conversion
import ufl
import numpy as np
import pyvista
from mpi4py import MPI
from petsc4py import PETSc # For PETSc.ScalarType
import gmsh # Added for Gmsh integration

# MPI communicator
comm = MPI.COMM_WORLD
rank = comm.rank

# 0. Simulation Parameters
# Material parameters
E = 1.0e5  # Young's modulus
nu = 0.499  # Poisson's ratio

# Lamé parameters
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# Body force
bz_val = 1.0e4  # Magnitude of body force
# Current script has force in +X direction
f_body_vec = np.array([bz_val, 0., 0.], dtype=PETSc.ScalarType)

# Finite element degree
element_degree = 1 # 1 for P1 (linear), 2 for P2 (quadratic)

# Visualization parameters
displacement_magnification_factor = 1.0 # Adjust if displacements are too small/large to see

# --- GMSH Model Generation for L-shape ---
gmsh.initialize()
if rank == 0:
    gmsh.option.setNumber("General.Terminal", 1) # Print Gmsh output on rank 0
else:
    gmsh.option.setNumber("General.Terminal", 0)

# Define L-shape using 3 unit cubes
# C1: [0,1]x[0,1]x[0,1] (base)
# C2: [0,1]x[0,1]x[1,2] (stacked on C1, using OCC addBox(x,y,z, dx,dy,dz))
# C3: [1,2]x[0,1]x[1,2] (side of C2, forming the L arm)
c1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
c2 = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1) # dx,dy,dz are lengths from (x,y,z)
c3 = gmsh.model.occ.addBox(1, 1, 0, 1, 1, 1)

# Fuse the boxes to form a single volume
# fragment computes intersections and prepares for a conforming mesh.
# The output 'ov' contains pairs of (dim, tag) for all created entities.
ov, _ = gmsh.model.occ.fragment([(3, c1), (3, c2), (3, c3)], [])
gmsh.model.occ.synchronize()

# Create a physical group for the entire L-shape volume.
# This helps gmshio identify what to mesh if multiple volumes exist.
volumes = gmsh.model.occ.getEntities(dim=3)
if not volumes:
    if rank == 0:
        print("No 3D volumes found in Gmsh model after fragment!")
    gmsh.finalize()
    exit()

L_shape_physical_tag = 101 # Arbitrary tag for the physical volume
gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], L_shape_physical_tag)
gmsh.model.setPhysicalName(3, L_shape_physical_tag, "L_Shape_Volume")

# Set mesh size (adjust mesh_size_factor for finer/coarser mesh)
mesh_size_factor = 0.07 # Smaller means finer. E.g., 0.3 means elements roughly 0.3 units long.
gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_factor * 0.8)
gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_factor * 1.2)
# gmsh.option.setNumber("Mesh.Algorithm", 6)  # Example: Frontal-Delaunay

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Convert Gmsh model to dolfinx mesh
try:
    # model_to_mesh will mesh entities in physical groups if defined.
    # It partitions the mesh across MPI ranks.
    domain, _, _ = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm, rank, gdim=3,
        partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none)
    )
    # We don't strictly need cell_tags or facet_tags if using geometric BC identification
except Exception as e:
    if rank == 0:
        print(f"ERROR: Gmsh to FEniCSx conversion failed: {e}")
        gmsh.write("L_shape_mesh_error.msh") # Save mesh for inspection
    gmsh.finalize()
    exit()

gmsh.finalize()
# --- End GMSH Model Generation ---


tdim = domain.topology.dim
fdim = tdim - 1

# 2. Define Function Space
V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", element_degree))

# 3. Define Boundary Conditions
# Fix the face at x = 0. This is the combined x=0 faces of C1 and C2.
def fixed_boundary_x0(x):
    # x is a [3, num_points] array of coordinates
    return np.isclose(x[0], 0.0)

fixed_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, fixed_boundary_x0)
fixed_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, fixed_facets)

u_D_val = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
u_D = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0)) # For vector space, need to provide it to component
# For vector spaces, dirichletbc can take a Constant with the correct dimension, or apply component-wise
# For fixing all components to 0, a 3D constant [0,0,0] is direct.
bc_u_D = dolfinx.fem.Constant(domain, u_D_val)
bc = dolfinx.fem.dirichletbc(bc_u_D, fixed_dofs, V)


# 4. Define Trial and Test Functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# 5. Define Body Force and Material Tensors
f_body = dolfinx.fem.Constant(domain, f_body_vec)

def epsilon(u_func):
    return 0.5 * (ufl.grad(u_func) + ufl.grad(u_func).T)

def sigma(u_func):
    return lambda_ * ufl.tr(epsilon(u_func)) * ufl.Identity(tdim) + 2 * mu * epsilon(u_func)

# 6. Define Variational Problem
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.dot(f_body, v) * ufl.dx # Renamed from L to avoid clash with lambda_

# 7. Solve the Linear Variational Problem
uh = dolfinx.fem.Function(V)
uh.name = "Displacement"

problem = dolfinx.fem.petsc.LinearProblem(
    a, L_form, bcs=[bc], u=uh,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
)
if rank == 0:
    print("Solving the linear system for L-shape...")
problem.solve()
if rank == 0:
    print("Solved.")

# 8. Visualization using PyVista
plotter = pyvista.Plotter(window_size=[1600, 900])

# Get mesh data for PyVista
# This part can be simplified if dolfinx.plot.vtk_mesh is available and works for your version
try:
    from dolfinx.plot import vtk_mesh as dolfinx_vtk_mesh
    topology, cell_types, geometry = dolfinx_vtk_mesh(domain, domain.topology.dim)
except (ImportError, AttributeError):
    if rank == 0:
        print("dolfinx.plot.vtk_mesh not found or failed, using manual PyVista grid construction.")
    # Manual construction (fallback)
    domain.topology.create_connectivity(domain.topology.dim, 0) # cells to vertices
    num_cells = domain.topology.index_map(domain.topology.dim).size_global
    cell_entities = np.arange(num_cells, dtype=np.int32) # Assuming contiguous cells
    cells_flat = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(num_cells, -1) # For tets, num_vertices_per_cell is 4
    
    num_vertices_per_cell = cells_flat.shape[1]
    # PyVista cell format: [num_points_cell0, v0_0, v0_1, ..., num_points_cell1, v1_0, ...]
    topology = np.hstack((np.full((num_cells, 1), num_vertices_per_cell, dtype=np.int32), cells_flat)).ravel()
    
    # Determine VTK cell type (e.g., TETRA for CellType.tetrahedron)
    if domain.topology.cell_name() == "tetrahedron":
        vtk_cell_type_pv = pyvista.CellType.TETRA
    elif domain.topology.cell_name() == "hexahedron":
        vtk_cell_type_pv = pyvista.CellType.HEXAHEDRON
    else: # Add more types or default
        vtk_cell_type_pv = pyvista.CellType.TETRA # Fallback, might be wrong
        if rank == 0: print(f"Warning: Unknown cell type {domain.topology.cell_name()} for PyVista, defaulting to TETRA.")

    cell_types = np.full(num_cells, vtk_cell_type_pv, dtype=np.uint8)
    geometry = domain.geometry.x


original_mesh_pv = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Interpolate solution to P1 if using higher-order for correct vertex displacement
if V.ufl_element().degree == 1:
    displacements_at_vertices = uh.x.array.reshape(geometry.shape[0], V.dofmap.index_map_bs)
else:
    V_plot = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    u_plot = dolfinx.fem.Function(V_plot)
    u_plot.interpolate(uh)
    displacements_at_vertices = u_plot.x.array.reshape(geometry.shape[0], V_plot.dofmap.index_map_bs)

# --- Displacement Statistics (Rank 0) ---
if rank == 0:
    print(f"Displacements at vertices (first 3 norms): {np.linalg.norm(displacements_at_vertices[:3,:], axis=1)}")
    max_disp_magnitude = np.max(np.linalg.norm(displacements_at_vertices, axis=1))
    print(f"Max displacement vector magnitude: {max_disp_magnitude:.3e}")
    # For coloring, often one component (e.g., X or Z) is useful
    max_x_disp_val = np.max(displacements_at_vertices[:, 0]) # Max X-disp for X-body-force
    min_x_disp_val = np.min(displacements_at_vertices[:, 0]) # Min X-disp
    print(f"Raw Max displacement in X (at vertices): {max_x_disp_val:.3e}")
    print(f"Raw Min displacement in X (at vertices): {min_x_disp_val:.3e}")
# --- END DEBUG ---

warped_mesh_pv = original_mesh_pv.copy()
warped_mesh_pv.points += displacements_at_vertices * displacement_magnification_factor
warped_mesh_pv.point_data["Displacement (X)"] = displacements_at_vertices[:, 0] # Store X-displacement for coloring

# --- Plotting ---
plotter.add_mesh(original_mesh_pv, style="wireframe", color="gray", opacity=0.3, label="Original")
plotter.add_mesh(warped_mesh_pv,
                 scalars="Displacement (X)", # Color by X-displacement
                 show_edges=True,
                 label="Deformed",
                 cmap="viridis",
                 scalar_bar_args={'title': "X-Displacement (Actual)"})

plotter.add_text(f"Deformed L-Shape (Disp. x{displacement_magnification_factor})",
                 position="upper_edge", font_size=12)
plotter.add_axes()
# plotter.view_isometric() # Uncomment for a standard isometric view initially

if rank == 0:
    print(f"Displaying L-shape. Magnification: {displacement_magnification_factor}x.")
    print(f"Magnified Max X-displacement: {max_x_disp_val * displacement_magnification_factor:.2e}")


# --- Export Deformed Mesh for Slicing (Rank 0 only) ---
if rank == 0: # Ensure only one process writes the file
    print("\nAttempting to export deformed mesh...")
    try:
        # 1. Extract the outer surface of the volumetric deformed mesh
        # This returns a pyvista.PolyData object (surface mesh)
        surface_deformed = warped_mesh_pv.extract_surface()

        # 2. Optional: Clean the surface mesh.
        # This can remove duplicate points, degenerate cells, etc.,
        # and can be helpful for getting a cleaner STL.
        # It also typically ensures the output is triangulated.
        surface_deformed_cleaned = surface_deformed.clean().triangulate()
        # If .clean() causes issues or removes too much, you can try just:
        # surface_deformed_cleaned = surface_deformed.triangulate()

        # 3. Save as STL (binary is usually smaller and faster)
        stl_filename = "deformed_l_shape.stl"
        surface_deformed_cleaned.save(stl_filename, binary=True)
        print(f"Deformed shape saved as STL: {stl_filename}")

        # 4. Optionally, save as OBJ
        obj_filename = "deformed_l_shape.obj"
        surface_deformed_cleaned.save(obj_filename)
        print(f"Deformed shape saved as OBJ: {obj_filename}")

    except Exception as e:
        print(f"Error during mesh export: {e}")
# --- End Export ---

plotter.show()
