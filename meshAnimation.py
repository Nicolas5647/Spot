from modules.header import addHeader, addSolvers
from spot import SpotRobot
from splib3 import animation
from math import pi
import Sofa
import Sofa.Core
import Sofa.ImGui as MyGui
import numpy as np
import json


ANIMATION = "data/animation/joints/meshSit.json"
OUTPUT = "Spot/data/animation/reverse/test.json"

POSE_MIN_TIME = 0.2
POSE_MAX_TIME = 5.0
POSE_STABLE_STEPS = 10


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

def anim(target, data, dt, effectors, factor):
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

    save(effectors, data, dt, factor)


def animDone(target, data, dt, effectors, factor):
    anim(target, data, dt, effectors, factor)
    frames = []
    tmp = framePositions[1]["t"]
    for f in framePositions:
        f["t"] -= tmp
        frames.append(f)

    writeOutput(frames[1:], data)


def writeOutput(frames, data):
    js = {"duration" : data["duration"], "frames" : frames}
    with open(OUTPUT, "w", encoding="UTF-8") as f:
        json.dump(js, f, indent=2)
    print(f"{len(frames)} frames sauvegardées")


def effectorsEntry(target, t):
    entry = {"t": t}
    for name, mo in target.items():
        entry[name[:2]] = np.array(mo.position.value[0]).tolist()   # [x, y, z]
    return entry

framePositions = []
nextKey = 0
def save(target, data, dt, factor):
    global nextKey
    while nextKey < len(dt) and factor >= dt[nextKey]:
        framePositions.append(effectorsEntry(target, data["frames"][nextKey]["t"]))
        nextKey += 1


class PoseRecorder(Sofa.Core.Controller):
    def __init__(self, rootnode, data, effectors, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.rootnode = rootnode
        self.effectors = effectors
        self.previous = None
        self.stableCount = 0
        self.startTime = None
        self.data = data
        self.done = False

    def currentPositions(self):
        return np.array([np.array(mo.position.value[0]) for mo in self.effectors.values()])

    def onAnimateEndEvent(self, event):
        if self.done:
            return

        time = self.rootnode.time.value
        if self.startTime is None:
            self.startTime = time
        elapsed = time - self.startTime

        current = self.currentPositions()
        if elapsed >= POSE_MIN_TIME and self.previous is not None \
                and np.max(np.abs(current - self.previous)) < 1e-6:
            self.stableCount += 1
        else:
            self.stableCount = 0
        self.previous = current

        stable = self.stableCount >= POSE_STABLE_STEPS
        if stable or elapsed >= POSE_MAX_TIME:
            self.done = True
            if not stable:
                print(f"Pose non stabilisée après {POSE_MAX_TIME}s, enregistrement quand même")
            writeOutput([effectorsEntry(self.effectors, self.data["frames"][0]["t"])], self.data)

def read():
    with open(ANIMATION, "r", encoding="UTF-8") as file:
        data = json.load(file)

    dt = []
    for value in data["frames"]:
        if data["duration"] != 0:
            dt.append(value["t"]/data["duration"])
        else:
            dt.append(value["t"])

    return data, dt

def setupRobot(rootnode, simulation):
    simulation.addChild(SpotRobot("data/model.urdf", freeFlyer=False))
    robot = simulation.SpotRobot.Robot

    jointNodes = [n for n in robot.Joints.children
                  if n.name.value not in ("universe", "root_joint")]
    positions = np.copy(robot.getMechanicalState().position.value)
    data, dt = read()
    constraints = addJointConstraint(robot, data, jointNodes, positions)

    mechanical = addEffector(robot)
    if data["duration"] == 0:
        rootnode.addObject(PoseRecorder(rootnode, data, mechanical, name="PoseRecorder"))
    else:
        animation.animate(anim, {"target": constraints, "data": data, "dt": dt, "effectors": mechanical},
                          duration=data["duration"], onDone=animDone)


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


def addJointConstraint(robot, data, jointNodes, positions):
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