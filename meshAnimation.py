from modules.header import addHeader, addSolvers
from spot import SpotRobot
from splib3 import animation
from math import pi
import Sofa
import Sofa.Core
import Sofa.ImGui as MyGui
import numpy as np
import json


ANIMATION = "data/animation/turn.json"
OUTPUT = "Spot/data/animation/reverse/turn.json"

POSE_MIN_TIME = 0.2
POSE_MAX_TIME = 5.0
POSE_STABLE_STEPS = 10


def createScene(rootnode):
    settings, modelling, simulation = addHeader(rootnode, inverse=False, withCollision=True)
    rootnode.ConstraintSolver.epsilon = 1e-3
    rootnode.addObject("InteractiveCamera", name="camera", computeZClip=False, zFar=10000)
    rootnode.addObject("PointCloudRenderer", name="renderer", camera=rootnode.camera.linkpath)
    rootnode.addObject(animation.AnimationManager(rootnode))
    rootnode.gravity = [0, 0, 0]

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
    begin = np.asarray(beginValue, dtype=float)
    end = np.asarray(endValue, dtype=float)
    return (end - begin) * scale + begin

def anim(constraint, target, root, data, dt, effectors, factor):
    i = 0
    while i < len(dt) - 1 and dt[i + 1] <= factor:
        i += 1
    j = i + 1 if i < len(dt) - 1 else 0
    span = (dt[j] if j > i else 1.0) - dt[i]
    local = (factor - dt[i]) / span if span > 0 else 0.0

    rootBegin = data["frames"][i]["root"]
    rootEnd = data["frames"][j]["root"]
    with root.position.writeableArray() as w:
        if "root" in data["frames"][i]:
            w[0] = LinearRamp(rootBegin, rootEnd, local)
        else:
            w[0] = [0, 0, 0, 0, 0, 0, 1]

    for constraint in constraint:
        begin = data["frames"][i]["joints"][constraint.name.value]
        end = data["frames"][j]["joints"][constraint.name.value]
        constraint.value = LinearRamp(begin, end, local)

    save(effectors, target, data, dt, factor)


def animDone(constraint, target, root, data, dt, effectors, factor):
    anim(constraint, target, root, data, dt, effectors, factor)
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


def effectorsEntry(effector, target, t):
    entry = {"t": t}
    for name, mo in effector.items():
        entry[name[:2]] = np.array(mo.position.value[0]).tolist()   # [x, y, z]
    entry["root"] = target.position.value[0].tolist()
    return entry

framePositions = []
nextKey = 0
def save(effector, target, data, dt, factor):
    global nextKey
    while nextKey < len(dt) and factor >= dt[nextKey]:
        framePositions.append(effectorsEntry(effector, target, data["frames"][nextKey]["t"]))
        nextKey += 1


class PoseRecorder(Sofa.Core.Controller):
    def __init__(self, rootnode, data, goalsEffectors, targetEffector, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.rootnode = rootnode
        self.goalsEffectors = goalsEffectors
        self.targetEffector = targetEffector
        self.previous = None
        self.stableCount = 0
        self.startTime = None
        self.data = data
        self.done = False

    def currentPositions(self):
        return np.array([np.array(mo.position.value[0]) for mo in self.goalsEffectors.values()])

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
            writeOutput([effectorsEntry(self.goalsEffectors, self.data["frames"][0]["t"])], self.data)


def quat_mul(a, b):
    """Produit de quaternions au format (x, y, z, w)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1)

def read():
    with open(ANIMATION, "r", encoding="UTF-8") as file:
        js = json.load(file)
    data = {}
    dt = []
    if js.get("format") != "urdfbase.sequence":
        frameDuration = js["FrameDuration"]
        frames = np.array(js["Frames"])
        root_pos = frames[:, :7]
        root_pos[:, :3] -= root_pos[0, :3] 

        a = np.radians(-90)
        q_R = np.array([np.sin(a/2), 0.0, 0.0, np.cos(a/2)])
        q_R = np.broadcast_to(q_R, root_pos[:, 3:].shape)
        root_pos[:, 3:] = quat_mul(root_pos[:, 3:], q_R)

        joints = frames[:, 7:]
        nameJoints = ["fr.hx", "fr.hy", "fr.kn", "fl.hx", "fl.hy", "fl.kn", "hr.hx", "hr.hy", "hr.kn", "hl.hx", "hl.hy", "hl.kn"]

        data["duration"] = len(frames) * frameDuration
        data["frames"] = []
        for i in range(len(frames)):
            frame = {"t" : i * frameDuration}
            frame["root"] = root_pos[i, :].tolist()
            frame["joints"] = {nameJoints[j]: joints[i, j] for j in range(len(nameJoints))}
            data["frames"].append(frame)
    else:
        data = js

    for value in data["frames"]:
        if data["duration"] != 0:
            dt.append(value["t"]/data["duration"])
        else:
            dt.append(value["t"])
    return data, dt

def setupRobot(rootnode, simulation):
    simulation.addChild(SpotRobot("data/model.urdf", freeFlyer=True))
    robot = simulation.SpotRobot.Robot
    robot.Joints.universe.activated = False


    jointNodes = [n for n in robot.Joints.children
                  if n.name.value not in ("universe", "root_joint")]
    positions = np.copy(robot.getMechanicalState().position.value)
    data, dt = read()
    constraints = addJointConstraint(robot, data, jointNodes, positions)

    goalsEffectors = addEffector(robot)
    targetEffector = addRootJointEffector(robot.Joints.root_joint)
    simulation.SpotRobot.RootJoint.addObject('PartialFixedProjectiveConstraint', template='Rigid3',
                        fixedDirections=[0, 0, 1, 0, 0, 0])
    if data["duration"] == 0:
        rootnode.addObject(PoseRecorder(rootnode, data, goalsEffectors, targetEffector, name="PoseRecorder"))
    else:
        animation.animate(anim, {"constraint": constraints, "target": targetEffector.MechanicalObject , "root": simulation.SpotRobot.RootJoint.getObject("Free-Flyer"), "data": data, "dt": dt, "effectors": goalsEffectors},
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


def addRootJointEffector(robot):
    # Effector: root_joint
    effector = robot.addChild('Effector')
    effector.addObject('MechanicalObject', template="Rigid3", position=[0, 0, 0, 0, 0 ,0, 1],
                       showObject=True, drawMode=2, showObjectScale=0.1)
    effector.addObject('RigidMapping')
    return effector


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