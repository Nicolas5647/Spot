from modules.robot import SpotRobot
from math import pi
from modules.robotconfigurations import spot_ctrl_joint_infos as spotInitConfiguration
from urchin import URDF
from modules.cloud import Cloud
from modules.miniskinning import SplatAssignment
from splib3 import animation
import Sofa.ImGui as MyGui
import numpy as np
import json


with open("data/walk.json", "r", encoding="UTF-8") as file:
    data = json.load(file)
dt = []
for value in data["frames"]:
    dt.append(value["t"]/data["duration"])


def LinearRamp(beginValue=0.0, endValue=1.0, scale=0.5):
    begin = np.asarray(beginValue, dtype=float)
    end = np.asarray(endValue, dtype=float)
    return (end - begin) * scale + begin


def targetAnimation(target, factor):
    with target.position.writeableArray() as w:
        w[0][3:7] = [0, np.sin((factor - 0.5) * pi/16), 0, np.cos((factor - 0.5) * pi/16)]
        #w[0][:3] = [np.cos(factor) - 0.5, 0, 0]

def walkAnimation(target, factor):
    i = 0
    while i < len(dt) - 1 and dt[i + 1] <= factor:
        i += 1
    j = i + 1 if i < len(dt) - 1 else 0
    span = (dt[j] if j > i else 1.0) - dt[i]
    local = (factor - dt[i]) / span if span > 0 else 0.0
    for node in target.children:
        begin = data["frames"][i][node.name.value]
        end = data["frames"][j][node.name.value]
        with node.dofs.position.writeableArray() as w:
            w[0][:3] = LinearRamp(begin, end, local)
    

def setupRobot(rootnode, simulation):
    simulation.addChild(SpotRobot("data/model.urdf", freeFlyer=True))
    robot = simulation.SpotRobot.Robot
    robot.Joints.universe.activated = False

    jointNodes = [n for n in robot.Joints.children
                  if n.name.value not in ("universe", "root_joint")]
    positions = np.copy(robot.getMechanicalState().position.value)
    constraints = addJointConstraint(robot, jointNodes, positions)
    
    tmp = np.copy(robot.dofs.position.value)
    robot.dofs.position.value = positions.tolist()
    addJointActuator(robot, jointNodes)
    addSliddingActuator(simulation.SpotRobot.RootJoint)

    goals = addGoals(rootnode)
    target = addTarget(rootnode)

    addEffector(robot, goals)
    addRootJointEffector(robot.Joints.root_joint, target)
    removeConstraint(robot, constraints)

    animation.animate(targetAnimation, {"target" : target.MechanicalObject}, mode="pingpong", duration=2)
    animation.animate(walkAnimation, {"target" : goals}, duration=1.4, mode="loop")

    addCloud(rootnode, robot)
    robot.dofs.position.value = tmp

    disableVisual(robot)

def disableVisual(robot):
    # Masquer les meshes du robot
    for link in robot.Joints.children:
        if hasattr(link, 'Visual'):
            for vis in link.Visual.children:
                vis.visualModel.enable.value = False

def addCloud(rootnode, robot, filename="splats/spot_export.ply"):
    cloud = rootnode.addChild(Cloud("cloud", filename))
    rootnode.init()
    rootnode.addObject(SplatAssignment(robot, cloud))

    rootnode.cloud.visual.renderer.frames.setParent(robot.Joints.jointsDof.position)
    rootnode.cloud.visual.renderer.initFrames.value = robot.Joints.jointsDof.position.value
    rootnode.cloud.visual.renderer.frameIndices.setParent(rootnode.SplatAssignment.idx)


def addJointActuator(robot, jointNodes):
    # Actuator pour les jambes
    urdf = URDF.load('data/model.urdf')
    for i in range(len(robot.dofs.position)):
        jointName = jointNodes[i].name.value
        min = urdf.joint_map[jointName].limit.lower 
        max = urdf.joint_map[jointName].limit.upper
        robot.addObject('JointActuator', template='Vec1', name='joint' + str(i), index=i,
                            maxAngle=max, minAngle=min, initAngle=(min+max)/2, 
                            maxAngleVariation=0.025)

def addSliddingActuator(node):
    # Sliding Actuator
    node.addObject('UniformMass', template='Rigid3', totalMass=16)
    node.addObject('PartialFixedProjectiveConstraint', template='Rigid3',
                        fixedDirections=[0, 1, 1, 1, 0, 1])
    node.addObject('SlidingActuator', name="Slide", template='Rigid3', indices=0,
                        direction=[1, 0, 0, 0, 0, 0], maxDispVariation=0.01,
                        maxForce=100, minForce=-100)
    node.addObject('SlidingActuator', name="Rotate", template='Rigid3', indices=0,
                        direction=[0, 0, 0, 0, 1, 0], maxDispVariation=0.01,
                        maxForce=100, minForce=-100)

def addEffector(robot, goals):
    # Effector: Jambes
    i = 0
    for node in robot.Joints.children:
        if node.name.value.endswith("kn"):
            effector = node.addChild("Effector")
            effector.addObject("MechanicalObject", name="MechanicalObject", position=[0, 0, -0.35])            
            effector.addObject('PositionEffector', indices=0, effectorGoal=goals.children[i].dofs.findData('position').getLinkPath(),
                            weight=1, useDirections=[1, 1, 1])
            effector.addObject("RigidMapping")
            i += 1

def addRootJointEffector(robot, target):
    # Effector: root_joint
    effector = robot.addChild('Effector')
    effector.addObject('MechanicalObject', template="Rigid3", position=[0, 0, 0, 0, 0 ,0, 1],
                       showObject=True, drawMode=2, showObjectScale=0.1)
    effector.addObject('PositionEffector', template="Rigid3", indices=0, effectorGoal=[0, 0, 0, 0, 0, 0, 1],
                       weight=2, useDirections=[1, 0, 0, 0, 1, 0])
    effector.addObject('RigidMapping')
    effector.PositionEffector.effectorGoal.setParent(target.MechanicalObject.position)

def addGoals(rootnode):
    # Goals pour les jambes
    # avant gauche, avant droit, arrière gauche, arrière droit
    positions = [[0.5974371584582934, 0.165945, -0.3800290937445745], 
                 [0.34243834638029763, -0.16594499999326034, -0.37108745269629617], 
                 [-0.2532616536197024, 0.16594499999326034, -0.37108745269629617], 
                 [0.0017371584582933086, -0.165945, -0.3800290937445745]]
    names = ["fl", "fr", "hl", "hr"]
    goals = rootnode.addChild("Goals")
    for i in range(4):
        goal = goals.addChild(names[i])
        goal.addObject('EulerImplicitIntegrationScheme', firstOrder=True)
        goal.addObject('CGLinearSolver', iterations=25, tolerance=1e-5, threshold=1e-5)
        goal.addObject('MechanicalObject', name='dofs', position=positions[i],
                        showObject=True, showObjectScale=0.05, drawMode=2)
        goal.addObject('UncoupledConstraintCorrection')
    return goals

def addTarget(rootnode):
    # Target pour le corps
    target = rootnode.addChild('Target')
    target.addObject('EulerImplicitIntegrationScheme', firstOrder=True)
    target.addObject('CGLinearSolver', tolerance=1e-5, iterations=20, threshold=1e-5)
    target.addObject('MechanicalObject', template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1], showObject=True, showObjectScale=0.15)
    target.addObject('PartialFixedProjectiveConstraint', template='Rigid3',
                 fixedDirections=[0, 1, 1, 1, 0, 1])
    target.addObject('UniformMass', template='Rigid3', totalMass=0.1)
    target.addObject('UncoupledConstraintCorrection')
    return target

def addJointConstraint(robot, jointNodes, positions):
    constraints = []
    for i in range(len(positions)):
        jointName = jointNodes[i].name.value
        value = 0 if jointName not in spotInitConfiguration else spotInitConfiguration[jointName].pos_desired
        positions[i] = value
        joint = robot.addObject('JointConstraint', template='Vec1', name=jointName, index=i,
                                valueType="displacement",
                                value=value)
        constraints.append(joint)
        MyGui.MyRobotWindow.addSetting(jointName, joint.value, -pi, pi)
    return constraints

def removeConstraint(robot, constraints):
    for c in constraints:
        robot.removeObject(c)

def printTree(node, indent=0):
    print(" " * indent + "[" + node.name.value + "]")
    for obj in node.objects:
        print(" " * (indent + 2) + "- " + obj.getClassName() + " : " + obj.name.value)
    for child in node.children:
        printTree(child, indent + 4)
