from modules.header import addHeader, addSolvers
from spot import setupRobot
from cube import setupCube
from splib3 import animation
from scipy.spatial.transform import Rotation as R, Slerp



def createScene(rootnode):
    settings, modelling, simulation = addHeader(rootnode, inverse=True, withCollision=True)
    rootnode.ConstraintSolver.epsilon = 1e-3
    rootnode.addObject("InteractiveCamera", name="camera")
    rootnode.addObject("PointCloudRenderer", name="renderer", camera=rootnode.camera.linkpath)
    manager = animation.AnimationManager(rootnode)
    rootnode.addObject(manager)

    addSolvers(simulation, rayleighStiffness=0)
    #simulation.solver.printLog = True
    #simulation.EulerImplicitIntegrationScheme.printLog = True
    #simulation.GenericConstraintCorrection.printLog = True
    #rootnode.ConstraintSolver.printLog = True
    rootnode.VisualStyle.displayFlags = ["showVisual"]

    setupRobot(rootnode, simulation, manager)
    #setupCube(rootnode, simulation)

    return