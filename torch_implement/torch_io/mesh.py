"""Loads mesh from disk"""


import os
from typing import Sequence
from trimesh.exchange.load import load_mesh

from torch_io import file
from physics import torch_config_pb2


def load(name: str, path: str,
         resource_paths: Sequence[str]) -> torch_config_pb2.MeshGeometry:
  """Returns MeshGeometry with faces/vertices loaded from path.

  Args:
    name: Name of the mesh.
    path: Path to mesh file name.
    resource_paths: Sequence of paths that may contain the mesh geometry file.

  Returns:
    mesh_geom: MeshGeometry object.

  Raises:
    AssertionError: if the mesh_geom.path is not found in resource_paths.
  """
  mesh_geom = torch_config_pb2.MeshGeometry()
  mesh_geom.name = name
  mesh_geom.path = path

  # Load the first existing mesh file into mesh_geom.
  for resource_path in resource_paths:
    path = os.path.join(resource_path, path)
    if not file.Exists(path):
      continue
    with file.File(path, 'rb') as f:
      mesh = load_mesh(f, file_type=str(mesh_geom.path))
      for v in mesh.vertices:
        mesh_geom.vertices.add(x=v[0], y=v[1], z=v[2])
      mesh_geom.faces.extend(mesh.faces.flatten())
      for v in mesh.vertex_normals:
        mesh_geom.vertex_normals.add(x=v[0], y=v[1], z=v[2])
      for v in mesh.face_normals:
        mesh_geom.face_normals.add(x=v[0], y=v[1], z=v[2])
    return mesh_geom

  raise AssertionError(f'{mesh_geom.path} was not found.')