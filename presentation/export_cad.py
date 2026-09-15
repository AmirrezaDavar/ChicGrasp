#!/usr/bin/env python3
"""Export visible CAD triangles with composed USD transforms for presentation.

Requires pxr/OpenUSD. No source geometry or simulation state is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('usd',type=Path)
ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs/assets/data')
args=ap.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
stage=Usd.Stage.Open(str(args.usd))
cache=UsdGeom.XformCache()
triangles=[]
groups=[]
sources=[]
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    mesh=UsdGeom.Mesh(prim)
    if mesh.ComputeVisibility()=='invisible':
        continue
    pts=np.array(mesh.GetPointsAttr().Get())
    matrix=np.array(cache.GetLocalToWorldTransform(prim))
    pts=(np.c_[pts,np.ones(len(pts))]@matrix)[:,:3]
    indices=np.array(mesh.GetFaceVertexIndicesAttr().Get())
    counts=np.array(mesh.GetFaceVertexCountsAttr().Get())
    # CAD tessellator already supplies triangles; convex polygon fans also work.
    offset=0
    group=1 if '/left_finger' in str(prim.GetPath()) else 2 if '/right_finger' in str(prim.GetPath()) else 0
    for n in counts:
        face=indices[offset:offset+n]
        for j in range(1,n-1):
            triangles.append(pts[face[[0,j,j+1]]])
            groups.append(group)
        offset+=n
    sources.append(str(prim.GetPath()))
triangles=np.asarray(triangles)
np.savez_compressed(args.output/'gripper_mesh.npz',triangles=triangles,groups=np.array(groups))
(args.output/'cad_provenance.json').write_text(json.dumps(dict(source=args.usd.name,
    source_sha256=hashlib.sha256(args.usd.read_bytes()).hexdigest(),mesh_prims=sources,
    triangles=len(triangles),stage_meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
    description='Local gripper CAD assembly, composed transforms preserved. Color accents added only in the renderer; no inferred jaw motion.',
    dependencies=[dict(file=Path(l.realPath).name,sha256=hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest())
                  for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()]),indent=2))
print('Exported',len(triangles),'triangles')
