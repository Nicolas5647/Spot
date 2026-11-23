from modules.header import addHeader, addSolvers
from splib3 import animation
from spot import addSliddingActuator, addTarget, addRootJointEffector
from math import pi
import numpy as np


def targetAnimation(target, factor):
    with target.position.writeableArray() as w:
        w[0][3:7] = [0, 0, np.sin((factor - 5) * pi), np.cos((factor - 5) * pi)]
        w[0][:3] = [np.cos(factor) - 0.5, 0, 0]

def addCube(simulation):
    cube = simulation.addChild("SpotRobot")
    cube.addObject('GenericConstraintCorrection', linearSolver=simulation.solver.linkpath, ODESolver=simulation.EulerImplicitIntegrationScheme.linkpath)
    cube.addObject('MechanicalObject', template='Rigid3', position=[-1, 0, 0, 0, 0, 0, 1])

    visual = cube.addChild('Visu')
    visual.addObject('MeshOBJLoader', filename='mesh/cube.obj', scale=0.1)
    visual.addObject('OglModel', src='@MeshOBJLoader')
    visual.addObject('RigidMapping')

def setupCube(rootnode, simulation):
    addCube(simulation)
    robot = simulation.SpotRobot
    addSliddingActuator(robot)

    target = addTarget(rootnode)
    addRootJointEffector(robot, target)
    animation.animate(targetAnimation, {"target" : target.MechanicalObject}, mode="pingpong", duration=10)