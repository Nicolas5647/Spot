import Sofa
import numpy as np

class SpotRobot(Sofa.Prefab):

    def __init__(self, urdf="data/model.urdf", freeFlyer=False):
        Sofa.Prefab.__init__(self)
        self.name = 'SpotRobot'
        self.freeFlyer = freeFlyer
        self.urdf = urdf

        # Add the robot model to the scene graph
        self.__addRobot()

    def __addRobot(self):

        # Robot node
        settings = self.addChild('Settings')
        settings.addObject("RequiredPlugin", name="Sofa.RigidBodyDynamics") # Needed to use components [URDFModelLoader]
        settings.addObject('RequiredPlugin', name='Sofa.Component.Mapping.NonLinear') # Needed to use components [RigidMapping]  
        settings.addObject('RequiredPlugin', name='Sofa.Component.Mass') # Needed to use components [UniformMass]
        settings.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Constant') # Needed to use components [MeshTopology]  
        settings.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D') # Needed to use components [OglModel]  

        self.addObject('URDFModelLoader', 
                        filename=self.urdf, 
                        modelDirectory="data/", 
                        useFreeFlyerRootJoint=self.freeFlyer, 
                        printLog=False, 
                        addCollision=False)
        robot = self.getChild("Robot")
        mechanical = robot.Model.getMechanicalState()
        mechanical.showObject = False
        mechanical.showObjectScale = 0.01
        mechanical.drawMode = 0
    

# Test/example scene
def createScene(rootnode):

    from modules.header import addHeader, addSolvers
    import Sofa.ImGui as MyGui
    from math import pi
    from modules.robotconfigurations import spot_ctrl_joint_infos as spotInitConfiguration

    settings, modelling, simulation = addHeader(rootnode, inverse=False, withCollision=False, friction=0)

    addSolvers(simulation, rayleighStiffness=0.001)
    rootnode.VisualStyle.displayFlags = ["showVisual"]

    # Units are in m, kg, s
    rootnode.dt = 0.01
    rootnode.gravity = [0., -9.81, 0.]

    # Robot
    simulation.addChild(SpotRobot())
    robot = simulation.SpotRobot.Robot

    # Direct problem
    names = robot.Joints.children
    positions = np.copy(robot.getMechanicalState().position.value)
    for i in range(len(positions)):
        jointName = names[i+1].name.value
        value = 0 if jointName not in spotInitConfiguration else spotInitConfiguration[jointName].pos_desired
        positions[i] = value
        joint = robot.addObject('JointConstraint', template='Vec1', name='joint' + str(i), index=i, 
                                valueType="angle", 
                                value=value
                                )
        MyGui.MyRobotWindow.addSetting(jointName, joint.value, -pi, pi)

    return
