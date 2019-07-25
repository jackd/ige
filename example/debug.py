from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shape_tfds.shape.shapenet.core import base

synset = 'rifle'
model_id = '4a32519f44dc84aabafe26e2eb69ebf4'
ids, names = base.load_synset_ids()
synset_id = ids['rifle']
print(synset_id)
exit()

with base.mesh_loader_context(synset_id) as loader:
    mesh = loader[model_id]
    mesh.show()
    print(mesh)
    mesh = base.as_mesh(mesh)
    print(mesh)
