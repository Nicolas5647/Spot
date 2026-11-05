import Sofa

class Cloud(Sofa.Prefab):
    def __init__(self, name, filename="splats/spot.ply", frames=None):
        Sofa.Prefab.__init__(self)
        self.name = name
        self.filename = filename
        self.frames = frames
        self.__addSpot()

    def __addSpot(self):
        self.addObject('RequiredPlugin', name="settings", pluginName='Sofa.PointCloud')

        self.addChild('geometry')
        self.geometry.addObject("PointCloudContainer", name="container", filename=self.filename)
        self.geometry.container.init()
        n = len(self.geometry.container.indices.value)

        self.addChild('visual')
        self.visual.addObject("PointCloudVisualModel", name="renderer",
                              indices=self.geometry.container.indices.linkpath,
                              geometry=self.geometry.container.linkpath)