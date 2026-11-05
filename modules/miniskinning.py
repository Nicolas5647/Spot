import Sofa
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d


def addSkinningFrames(robot):
    nFrames = len(robot.Joints.jointsDof.position.value)
    skin = robot.Joints.addChild('SkinningFrames')
    skin.addObject('MechanicalObject', name='frames', template='Rigid3',
                   position=[[0, 0, 0, 0, 0, 0, 1]] * nFrames)
    skin.addObject('IdentityMapping', input='@../jointsDof', output='@frames',
                   rigidIndexPerPoint=list(range(nFrames)), globalToLocalCoords=True)
    return skin


class SplatAssignment(Sofa.Core.Controller):
    def __init__(self, robot, cloud, **kwargs):
        super().__init__(**kwargs)
        self.addData('idx', type='vector<int>', value=[0]*len(cloud.geometry.container.positions.value))
        self.name = "SplatAssignment"
        self.robot, self.cloud = robot, cloud
        self.done = False


    def onEvent(self, event):
        pass

    def assignment(self):
        if self.done:
            return
        self.done = True

        scene = o3d.t.geometry.RaycastingScene()
        mesh_index = {}

        for node in self.robot.Joints.children:
            if not hasattr(node, 'jointMapping') or not hasattr(node, 'Visual'):
                continue
            
            index = node.jointMapping.index.value
            
            for vis in node.Visual.children:
                if not vis.name.value.endswith("hip_0"):
                    v = np.array(vis.visualModel.position.value, dtype=np.float32)
                    f = np.array(vis.visualModel.triangles.value, dtype=np.uint32)

                    mesh = scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
                    mesh_index[mesh] = index

        pts = np.array(self.cloud.geometry.container.positions.value, dtype=np.float32)
        res = scene.compute_closest_points(o3d.core.Tensor(pts)) # Calcule le mesh le plus proche pour chaque point
        pts_ids = res['geometry_ids'].numpy() # Liste des meshs les plus proches pour chaque point

        lookup = np.zeros(max(mesh_index) + 1, dtype=int)
        for mesh, idx in mesh_index.items():
            lookup[mesh] = idx

        with self.idx.writeableArray() as w:
            w[:] = lookup[pts_ids]

        