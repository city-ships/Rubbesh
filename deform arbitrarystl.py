import dolfinx
import dolfinx.fem.petsc
import dolfinx.mesh
import ufl
import numpy as np
import pyvista
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import os # For file operations

# --- Configuration ---
STL_FILE_PATH = "toothpastesqueezers.stl"  # low res stl
OUTPUT_DIR = "fem_output"         # Directory to save results

# Material parameters
E = 1.0e5
nu = 0.45 # Or 0.499 if you prefer, but mind large deformations

# Body force (e.g., in +Z direction)
body_force_magnitude = 1.0e3
f_body_vec_applied = np.array([0.0, 0.0, body_force_magnitude], dtype=PETSc.ScalarType)

# Finite element degree
element_degree = 1

# Visualization
displacement_magnification_factor = 1 # Adjust based on deformation magnitude

# Gmsh mesh size factor (relative to characteristic length or absolute if desired)
# A smaller value means a finer mesh. # has no effect now, Gmsh ignores this
gmsh_mesh_size_factor = 0.1 # Example: if characteristic length is 10, mesh size ~1

# Tolerance for identifying nodes for boundary conditions
GEOMETRICAL_BC_TOLERANCE = 1e-5
# --- End Configuration ---

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.rank

def main():
	if rank == 0:
		if not os.path.exists(STL_FILE_PATH):
			print(f"ERROR: STL file not found at '{STL_FILE_PATH}'")
			return
		if not os.path.exists(OUTPUT_DIR):
			os.makedirs(OUTPUT_DIR)
		print(f"--- Starting FEM analysis for {STL_FILE_PATH} ---")

	# 1. Load STL and Find Largest Triangle (PyVista - on rank 0, then broadcast)
	largest_triangle_vertex_coords = None
	if rank == 0:
		print("Step 1: Loading STL and finding largest triangle...")
		try:
			surface_mesh = pyvista.read(STL_FILE_PATH)
			if not isinstance(surface_mesh, pyvista.PolyData):
				surface_mesh = surface_mesh.extract_geometry() # Ensure PolyData

			if surface_mesh.n_points == 0 or surface_mesh.n_cells == 0:
				raise ValueError("STL file is empty or could not be read properly.")

			# Ensure mesh is triangulated for simple area calculation
			surface_mesh = surface_mesh.triangulate()

			max_area = -1.0
			best_triangle_vertices = None

			# Iterate through faces to calculate area
			# PyVista faces are like [3, v0, v1, v2, 3, v3, v4, v5, ...]
			faces_as_array = surface_mesh.faces.reshape(-1, 4) # Each row: [3, v0, v1, v2]

			for face_info in faces_as_array:
				if face_info[0] != 3: # Should always be 3 after triangulate
					continue
				v_indices = face_info[1:]
				p0 = surface_mesh.points[v_indices[0]]
				p1 = surface_mesh.points[v_indices[1]]
				p2 = surface_mesh.points[v_indices[2]]

				# Area = 0.5 * || (P1-P0) x (P2-P0) ||
				area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

				if area > max_area:
					max_area = area
					best_triangle_vertices = np.array([p0, p1, p2])

			if best_triangle_vertices is None:
				raise ValueError("Could not find any triangles in the STL.")

			largest_triangle_vertex_coords = best_triangle_vertices
			print(f"Largest triangle area: {max_area:.4f}")
			print(f"Vertices of largest triangle:\n{largest_triangle_vertex_coords}")

		except Exception as e:
			print(f"ERROR in Step 1 (PyVista): {e}")
			largest_triangle_vertex_coords = None # Signal error

	# Broadcast largest_triangle_vertex_coords to all processes
	largest_triangle_vertex_coords = comm.bcast(largest_triangle_vertex_coords, root=0)
	if largest_triangle_vertex_coords is None:
		if rank == 0: print("Aborting due to error in finding largest triangle.")
		return

	# 2. Tetrahedral Meshing with Gmsh
	if rank == 0: print("Step 2: Generating tetrahedral mesh with Gmsh...")
	gmsh.initialize()
	if rank == 0:
		gmsh.option.setNumber("General.Terminal", 1)
	else:
		gmsh.option.setNumber("General.Terminal", 0) # Suppress output on other ranks

	try:
		# Merge STL into Gmsh
		gmsh.merge(STL_FILE_PATH)
		gmsh.model.occ.synchronize() # Important after STL import for OCC-based kernels

		# Create a surface loop and volume (required for 3D meshing from surface)
		# Get all discrete surfaces from the STL import
		surfaces = gmsh.model.getEntities(dim=2)
		if not surfaces:
			raise ValueError("No surfaces found after merging STL. Is the STL valid?")

		surface_tags = [s[1] for s in surfaces]
		sloop = gmsh.model.geo.addSurfaceLoop(surface_tags) # Use geo for discrete entities
		vol = gmsh.model.geo.addVolume([sloop])
		gmsh.model.geo.synchronize() # Synchronize after geo operations

		# Create Physical Group for the volume
		gmsh.model.addPhysicalGroup(3, [vol], 1)
		gmsh.model.setPhysicalName(3, 1, "Volume")

		# Set mesh size
		# characteristic_length = gmsh.model.getBoundingBox(-1,-1,-1)[3] # dx of bounding box
		# mesh_size = characteristic_length * gmsh_mesh_size_factor
		gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
		gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
		gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
		# A more direct way to set a general mesh size:
		gmsh.option.setNumber("Mesh.MeshSizeMin", gmsh_mesh_size_factor * 0.8) # Example
		gmsh.option.setNumber("Mesh.MeshSizeMax", gmsh_mesh_size_factor * 1.2) # Example
		# gmsh.option.setGlobalMeshSizeFactor(gmsh_mesh_size_factor) # Alternative for global scaling

		gmsh.model.mesh.generate(3) # Generate 3D mesh

		# Convert Gmsh model to dolfinx mesh
		domain, _, _ = dolfinx.io.gmshio.model_to_mesh(
			gmsh.model, comm, rank=0, gdim=3, # rank=0 for reading model on one proc
			partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none)
		)
	except Exception as e:
		if rank == 0: print(f"ERROR in Step 2 (Gmsh): {e}")
		gmsh.finalize()
		return
	finally:
		gmsh.finalize() # Always finalize Gmsh

	tdim = domain.topology.dim
	fdim = tdim - 1
	if rank == 0: print("Gmsh meshing complete.")

	# 3. Define Function Space and Boundary Conditions
	if rank == 0: print("Step 3: Defining FE problem and BCs...")
	V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", element_degree))

	# Boundary condition: Fix nodes corresponding to the largest triangle's vertices
	fixed_node_coords_np = np.array(largest_triangle_vertex_coords)

	def fixed_nodes_predicate(x_coords_dolfinx):
		# x_coords_dolfinx is a [3, num_points] array from dolfinx
		is_close = np.full(x_coords_dolfinx.shape[1], False, dtype=bool)
		for i in range(fixed_node_coords_np.shape[0]): # Iterate over 3 vertices
			node_coord = fixed_node_coords_np[i, :]
			# Calculate squared distance from dolfinx points to this fixed node coord
			dist_sq = np.sum((x_coords_dolfinx.T - node_coord)**2, axis=1)
			is_close = np.logical_or(is_close, dist_sq < GEOMETRICAL_BC_TOLERANCE**2)
		return is_close

	# Locate DOFs corresponding to these geometric points
	fixed_dofs = dolfinx.fem.locate_dofs_geometrical(V, fixed_nodes_predicate)

	u_D_val = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
	bc_u_D = dolfinx.fem.Constant(domain, u_D_val)
	bc = dolfinx.fem.dirichletbc(bc_u_D, fixed_dofs, V)
	if rank == 0:
		num_fixed_dofs = len(fixed_dofs) if hasattr(fixed_dofs, '__len__') else 0
		print(f"Number of DOFs fixed for largest triangle: {num_fixed_dofs}")
		if num_fixed_dofs == 0:
			print("WARNING: No DOFs were fixed for the boundary condition. Check the predicate and tolerance.")


	# 4. Define and Solve Elastic Problem
	if rank == 0: print("Step 4: Solving linear elastic problem...")
	u = ufl.TrialFunction(V)
	v = ufl.TestFunction(V)

	lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
	mu_val = E / (2 * (1 + nu)) # Renamed from mu to mu_val to avoid clash if mu is a function

	f_body = dolfinx.fem.Constant(domain, f_body_vec_applied)

	def epsilon(u_func):
		return 0.5 * (ufl.grad(u_func) + ufl.grad(u_func).T)

	def sigma(u_func):
		return lambda_ * ufl.tr(epsilon(u_func)) * ufl.Identity(tdim) + 2 * mu_val * epsilon(u_func)

	a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
	L_form = ufl.dot(f_body, v) * ufl.dx

	uh = dolfinx.fem.Function(V)
	uh.name = "Displacement"

	problem = dolfinx.fem.petsc.LinearProblem(
		a, L_form, bcs=[bc], u=uh,
		petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
	)
	problem.solve()
	if rank == 0: print("Solved.")

	# 5. Visualization (PyVista)
	if rank == 0: print("Step 5: Preparing visualization...")
	plotter = pyvista.Plotter(window_size=[1200, 800])

	# Get mesh data for PyVista from dolfinx domain
	try:
		from dolfinx.plot import vtk_mesh as dolfinx_vtk_mesh # alias to avoid conflict
		topology, cell_types, geometry = dolfinx_vtk_mesh(domain, domain.topology.dim)
	except (ImportError, AttributeError):
		if rank == 0: print("dolfinx.plot.vtk_mesh not found, using manual PyVista grid construction.")
		domain.topology.create_connectivity(domain.topology.dim, 0)
		num_cells = domain.topology.index_map(domain.topology.dim).size_global
		cells_flat = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(num_cells, -1)
		num_vertices_per_cell = cells_flat.shape[1]
		topology = np.hstack((np.full((num_cells, 1), num_vertices_per_cell, dtype=np.int32), cells_flat)).ravel()
		if domain.topology.cell_name() == "tetrahedron": vtk_cell_type_pv = pyvista.CellType.TETRA
		else: vtk_cell_type_pv = pyvista.CellType.VOXEL # Basic fallback
		cell_types = np.full(num_cells, vtk_cell_type_pv, dtype=np.uint8)
		geometry = domain.geometry.x

	original_mesh_pv = pyvista.UnstructuredGrid(topology, cell_types, geometry)

	if V.ufl_element().degree == 1:
		displacements_at_vertices = uh.x.array.reshape(geometry.shape[0], V.dofmap.index_map_bs)
	else:
		V_plot = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 1))
		u_plot = dolfinx.fem.Function(V_plot)
		u_plot.interpolate(uh)
		displacements_at_vertices = u_plot.x.array.reshape(geometry.shape[0], V_plot.dofmap.index_map_bs)
	
	if rank == 0:
		max_disp_val_component = np.max(displacements_at_vertices[:, 2]) # Z-component for Z-force
		print(f"Raw Max Z-displacement: {max_disp_val_component:.3e}")

	warped_mesh_pv = original_mesh_pv.copy()
	warped_mesh_pv.points += displacements_at_vertices * displacement_magnification_factor
	# Store displacement component used for coloring
	disp_component_for_color = displacements_at_vertices[:, np.argmax(np.abs(f_body_vec_applied))] # Component along force
	warped_mesh_pv.point_data["Displacement_Magnitude_ForceDir"] = disp_component_for_color


	# Export deformed shape
	if rank == 0:
		try:
			surface_deformed = warped_mesh_pv.extract_surface().clean().triangulate()
			stl_out_path = os.path.join(OUTPUT_DIR, os.path.basename(STL_FILE_PATH).replace(".stl", "_deformed.stl"))
			surface_deformed.save(stl_out_path, binary=True)
			print(f"Deformed shape saved as STL: {stl_out_path}")
		except Exception as e:
			print(f"Error exporting deformed mesh: {e}")


	plotter.add_mesh(original_mesh_pv, style="wireframe", color="gray", opacity=0.3, label="Original")
	plotter.add_mesh(warped_mesh_pv,
					 scalars="Displacement_Magnitude_ForceDir",
					 show_edges=True, label="Deformed", cmap="viridis",
					 scalar_bar_args={'title': "Displacement (Force Dir.)"})
	plotter.add_text(f"Deformed {os.path.basename(STL_FILE_PATH)} (Disp. x{displacement_magnification_factor})",
					 position="upper_edge", font_size=12)
	plotter.add_axes()
	# plotter.view_isometric()

	if rank == 0: print("Displaying visualization. Close window to exit.")
	plotter.show()
	if rank == 0: print("--- Analysis complete ---")

if __name__ == "__main__":
	main()
