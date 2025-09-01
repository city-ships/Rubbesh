# Rubbesh
A proof-of-concept using FEM to physically deform STL files for overhang-free 3D printing, as a possible alternative to the S4 slicer approach. This quick demo simulates rubber-like material deformation. For demonstration only.


# **Physics-Based Deformation using FEM**

## **Overview**

This repository contains Python scripts that demonstrate a physics-based approach to deforming 3D models using the Finite Element Method (FEM). This method serves as a physically-grounded alternative to purely geometric or procedural deformation methods, such as those in the [S4 Slicer project](https://github.com/jyjblrd/S4_Slicer).  
By simulating how an object would deform under specified forces and boundary conditions, one can generate complex, physically plausible shapes. The final deformed mesh can be exported as an STL file for direct use in 3D printing slicers.

## **Key Dependencies**

An environment with the following packages is required:

* **dolfinx**: The core FEM solver library.  
* **gmsh**: A 3D finite element mesh generator.  
* **pyvista**: For 3D plotting and mesh analysis.  
* **numpy**: For numerical operations.  
* **mpi4py**: For parallel execution.

## **Scripts Overview**

The repository contains three primary examples:

* **Cube\_deformation.py**: A foundational example demonstrating the complete FEM workflow on a simple cube primitive.  
* **L-shape-deformation.py**: An intermediate example that uses gmsh to mesh a more complex L-shaped geometry, deforms it, and exports the result as an STL file.  
* **deform\_arbitrarystl.py**: The most generalized script. It loads an arbitrary STL file, generates a solid tetrahedral mesh from it, simulates deformation, and saves the resulting deformed surface as a new STL file. It uses a simple heuristic (fixing the largest triangle) to anchor the object.

## **Usage**

Each script is self-contained. Parameters like material properties and forces can be modified within the files. The scripts are run using mpiexec.  
\# To run a script on N processor cores (e.g., N=4)  
mpiexec \-n 4 python3 deform\_arbitrarystl.py

## **Project Status and Future Work**

The current state is a functional **proof-of-concept**. The implementation is direct and procedural ("vibvecoded"), which is effective for validation but offers opportunities for generalization.  
Potential improvements include:

* **Refactoring**: Abstracting the core FEM logic into reusable functions or classes.  
* **User Interface**: Developing a command-line interface (CLI) to allow users to specify parameters without modifying the source code.  
* **Boundary Conditions**: Implementing more robust methods for users to define fixed regions.  
* **Non-Linear Analysis**: Incorporating hyperelastic models to accurately simulate large deformations.
