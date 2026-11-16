from modules.header import addHeader, addSolvers
from spot import SpotRobot
from splib3 import animation
from math import pi
import Sofa.ImGui as MyGui
import numpy as np
import json


with open("data/meshWalk.json", "r", encoding="UTF-8") as file:
    data = json.load(file)
dt = []
for value in data["frames"]:
    dt.append(value["t"]/data["duration"])

def createScene(rootnode):
    settings, modelling, simulation = addHeader(rootnode, inverse=False, withCollision=True)
    rootnode.ConstraintSolver.epsilon = 1e-3
    rootnode.addObject("InteractiveCamera", name="camera", computeZClip=False, zFar=10000)
    rootnode.addObject("PointCloudRenderer", name="renderer", camera=rootnode.camera.linkpath)
    rootnode.addObject(animation.AnimationManager(rootnode))

    addSolvers(simulation, rayleighStiffness=0)
    #simulation.solver.printLog = True
    #simulation.EulerImplicitIntegrationScheme.printLog = True
    #simulation.GenericConstraintCorrection.printLog = True
    #rootnode.ConstraintSolver.printLog = True
    rootnode.VisualStyle.displayFlags = ["showVisual"]

    setupRobot(rootnode, simulation)

    return

def LinearRamp(beginValue=0.0, endValue=1.0, scale=0.5):
    """Linear interpolation between two values

    Examples:
        LinearRamp(10, 20, 0.5)    

    """
    return (endValue-beginValue) * scale + beginValue

def walk(target, effectors, factor):
    i = 0
    while i < len(dt) - 1 and dt[i + 1] <= factor:
        i += 1
    j = i + 1 if i < len(dt) - 1 else 0
    span = (dt[j] if j > i else 1.0) - dt[i]
    local = (factor - dt[i]) / span if span > 0 else 0.0
    for constraint in target:
        begin = data["frames"][i]["joints"][constraint.name.value]
        end = data["frames"][j]["joints"][constraint.name.value]
        constraint.value = LinearRamp(begin, end, local)

    save(effectors, factor)


def walkDone(target, effectors, factor):
    walk(target, effectors, factor)
    frames = []
    tmp = framePositions[1]["t"]
    for f in framePositions:
        f["t"] -= tmp
        frames.append(f)


    js = {"duration" : data["duration"], "frames" : frames[1:]}
    with open("effectors.json", "w", encoding="UTF-8") as f:
        json.dump(js, f, indent=2)
    print(f"{len(framePositions)} frames sauvegardées")

framePositions = []
nextKey = 0
def save(target, factor):
    global nextKey
    while nextKey < len(dt) and factor >= dt[nextKey]:
        entry = {"t": data["frames"][nextKey]["t"]}
        for name, mo in target.items():
            entry[name[:2]] = np.array(mo.position.value[0]).tolist()   # [x, y, z]
        framePositions.append(entry)
        nextKey += 1


def setupRobot(rootnode, simulation):
    simulation.addChild(SpotRobot("data/model.urdf", freeFlyer=False))
    robot = simulation.SpotRobot.Robot

    jointNodes = [n for n in robot.Joints.children
                  if n.name.value not in ("universe", "root_joint")]
    positions = np.copy(robot.getMechanicalState().position.value)
    constraints = addJointConstraint(robot, jointNodes, positions)

    mechanical = addEffector(robot)

    animation.animate(walk, {"target": constraints, "effectors": mechanical},
                  duration=1.4, onDone=walkDone)
    #animation.animate(walk, {"target" : constraints}, duration=1.4, mode="loop")


def addEffector(robot):
    # Effector: Jambes
    i = 0
    mechanicals = {}
    for node in robot.Joints.children:
        if node.name.value.endswith("kn"):
            effector = node.addChild("Effector")
            mecha = effector.addObject("MechanicalObject", name="MechanicalObject", position=[0, 0, -0.35], showObject=True, showObjectScale=0.1)
            effector.addObject("RigidMapping")
            i += 1
            mechanicals[node.name.value] = mecha
    return mechanicals


def addJointConstraint(robot, jointNodes, positions):
    constraints = []
    for i in range(len(positions)):
        jointName = jointNodes[i].name.value
        value = 0 if jointName not in  data["frames"][0]["joints"] else data["frames"][0]["joints"][jointName] 
        positions[i] = value
        joint = robot.addObject('JointConstraint', template='Vec1', name=jointName, index=i,
                                valueType="displacement",
                                value=value)
        constraints.append(joint)
        MyGui.MyRobotWindow.addSetting(jointName, joint.value, -pi, pi)
    return constraints