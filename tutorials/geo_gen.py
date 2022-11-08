import trimesh as tm
import numpy as np
from scipy.spatial.transform import Rotation as rot

mesh = tm.creation.box([5e3, 1e3, 1e3]) # create the box
mesh.rezero()                           # translate all vertices so that all coordinates are positive
mesh.show()                             # visualise geometry

mesh_new = tm.creation.box([3e3, 2e3, 1e3]) # create the box
mesh_new.rezero()                           # rezero
mesh_new.vertices[:, [0,1]] += 1e3          # move all vertices 1e3 in the x and y coordinates

mesh = tm.boolean.union([mesh, mesh_new], engine = 'scad') # unite both meshes

# mesh.show()

# move the hole to the desired place
mesh_hole = tm.creation.box([1e3, 3e3, 1e3])
mesh_hole.rezero()
# mesh_hole.vertices[:, 2] -= mesh_hole.vertices[:, 2].min() # ensure that the base of the hole is in the z = 0 plane.
mesh_hole.vertices[:, 0] += 2e3                          # move it 2.5e3 in x
mesh_hole.vertices[:, 1] += 1e3                          # move it 1.5e3 in y direction

mesh = tm.boolean.difference([mesh, mesh_hole], engine = 'scad') # make the hole (new_mesh = mesh - mesh_hole)

mesh.rezero() # rezero everything
# mesh.show()
_ = tm.exchange.export.export_mesh(mesh, 'my_mesh2', 'stl')

# %% [markdown]
# We can also cut a mesh by defining the plane normal and plane origin, where the normal points to the side that will be kept:

# %%
# mesh = tm.intersections.slice_mesh_plane(mesh, plane_normal = [-1, -1, 0], plane_origin = [3e3, 3e3, 0], cap = True) # cut the upper right corner at 45°
# mesh = tm.intersections.slice_mesh_plane(mesh, plane_normal = [ 1, -1, 0], plane_origin = [2e3, 3e3, 0], cap = True) # cut the upper left  corner at 45°

# mesh.show()

# # %% [markdown]
# # Finally, we can process the geometry to check for duplicated vertices, triangles, etc, that could lead to potential errors, check whether the mesh is watertight and save it as `my_mesh.stl`:

# # %%
# mesh.process(validate = True)
# print(mesh.is_watertight)

# extension = 'stl'
# filename = 'my_mesh.' + extension

# _ = tm.exchange.export.export_mesh(mesh, filename, extension)


# %% [markdown]
# To use the geometry in nano-$\kappa$, we can write the following parameters on a `.txt` file:
# 
#     --mat_folder       <path_to_folder>
#     --hdf_file         <filename>.hdf5
#     --poscar_file      <filename>
#     --mat_names        <material_name>
#     --geometry         <path>\my_mesh.stl
#     --subvolumes       voronoi 9
#     --bound_pos        relative -0.01 0.15 0.5 1.01 0.15 0.5 0.5 1.01 0.5
#     --bound_cond       T T T R
#     --reservoir_gen    fixed_rate
#     --bound_values     298 209 302 0
#     --reference_temp   300
#     --energy_normal    mean
#     --temp_dist        cold
#     --particles        total 1e5
#     --part_dist        random_subvol
#     --timestep         1
#     --iterations       1500
#     --results_folder   my_mesh_test
#     --results_location <path_to_folder>
#     --colormap         jet
#     --fig_plot         
#     --rt_plot          energy
# 
# This uses relative positions to apply $T_{cold} = 298$ K on left and right sides (relative coordinates [-0.01, 0.15, 0.5] and [1.01, 0.15, 0.5]) and $T_{hot} = 302$ K at the facet between the diagonal cuts at the top (relative coordinate [0.5, 1.01, 0.5]). The other walls have a roughness of 0 $\text{\AA}$. The subvolumes are 9, of the Voronoi type. The simulation will run for 1500 iterations of 1 ps timestep. The following animation was generated showing the energy of the particles in the domain, as demanded at the `--rt_plot` parameter:
# 
# <!-- ![](readme_fig\last_anim_frame.png) -->
# ![](readme_fig\simulation.gif)
# 


