# Meshing in Femwell

Meshing can be done in femwell and actually it provides some routines for setting up the typical mesh for a waveguide cross-section. These routines are contained in the package `femwell.mesh`.

Femwell meshing routines are based on external meshing and geometry libraries among which we find

- [`gmsh`](https://gmsh.info/), a well-known open source library for generation of finite element mesh generation;
- [`pygmsh`](https://github.com/nschloe/pygmsh), python library providing wrapping many features of gmsh and easing its use in Python
- [`shapely`](https://github.com/shapely/shapely), python library providing routines for geometric objects analysis and manipulation.

The entry point for meshing in femwell is the routine `mesh_from_OrderedDict` available in `femwell.mesh`. This one accepts a `OrderedDict` object specifying the geometrical entities  the cross-section is composed and performs the following actions

- run gmsh
- write the gmsh mesh to a temporary file
- load the mesh with scikit-fem for use in the finite element assembly

A practical example defining a buried waveguide is the following

```python
wg_width = 0.5
wg_thickness = 0.3
box_width = 8
box_heigth = 6
core = shapely.geometry.box(-wg_width / 2, 0, +wg_width / 2, wg_thickness)
cladding = shapely.geometry.box(-box_width/2, -box_heigth/2, box_width/2, box_heigth/2)
env = cladding
# env = shapely.affinity.scale(core.buffer(5, resolution=8), xfact=0.5)

polygons = OrderedDict(
    core=core,
    box=clip_by_rect(env, -np.inf, -np.inf, np.inf, 0),
    clad=clip_by_rect(env, -np.inf, 0, np.inf, np.inf),
)

resolutions = dict(core={"resolution": 0.05, "distance": 0.05})

mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=10))
mesh.draw().show()
```

![alt text](image.png)

In case of overlapping polygons, femwell operate in a similar way to [Lumerical](https://optics.ansys.com/hc/en-us/articles/360034915233-Understanding-mesh-order-for-overlapping-objects) and gives priority to the shapes defined first in the dictionary passed to `mesh_from_OrderedDict`. For example in the code snippet above, the core and the cladding overlap, but core shape have the priority because it comes first in the `polygons` dictionary.

### Controlling the mesh size


### Material region

## Custom Meshes

Since femwell its modular, it accepts every mesh loaded into scikit-fem. You just need to replace mesh_from_OrderedDict and load the mesh you generated yourself using a code like this

```python
mesh = from_meshio(meshio.read("custom_mesh.msh"))
```