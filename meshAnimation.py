from modules.header import addHeader, addSolvers
from spot import SpotRobot
from splib3 import animation
from math import pi
import Sofa
import Sofa.Core
try:
    import Sofa.ImGui as MyGui
except ImportError:
    MyGui = None
import numpy as np
import json


ANIMATION = "data/animation/frames/meshTurn.json"
OUTPUT = "Spot/data/animation/reverse/turn.json"

def quat_mul(a, b):
    """Produit de quaternions au format (x, y, z, w)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1)


def quat_conj(q):
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q, v):
    """Tourne le vecteur v par le quaternion q."""
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_slerp(a, b, s):
    """Interpolation sphérique entre deux quaternions (plus court chemin)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.dot(a, b)
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + s * (b - a)
    else:
        th = np.arccos(d)
        q = (np.sin((1.0 - s) * th) * a + np.sin(s * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


def LinearRamp(beginValue=0.0, endValue=1.0, scale=0.5):
    """Linear interpolation between two values

    Examples:
        LinearRamp(10, 20, 0.5)

    """
    begin = np.asarray(beginValue, dtype=float)
    end = np.asarray(endValue, dtype=float)
    return (end - begin) * scale + begin


def createScene(rootnode):
    settings, modelling, simulation = addHeader(rootnode, inverse=False, withCollision=True)
    rootnode.ConstraintSolver.epsilon = 1e-3
    rootnode.addObject("InteractiveCamera", name="camera", computeZClip=False, zFar=10000)
    rootnode.addObject("PointCloudRenderer", name="renderer", camera=rootnode.camera.linkpath)
    rootnode.addObject(animation.AnimationManager(rootnode))
    rootnode.gravity = [0, 0, 0]

    addSolvers(simulation, rayleighStiffness=0)
    rootnode.VisualStyle.displayFlags = ["showVisual"]

    setupRobot(rootnode, simulation)

    return

recording = {"factor": None}

def anim(constraint, target, root, data, dt, effectors, factor):
    i = 0
    while i < len(dt) - 1 and dt[i + 1] <= factor:
        i += 1
    j = i + 1 if i < len(dt) - 1 else 0
    span = (dt[j] if j > i else 1.0) - dt[i]
    local = (factor - dt[i]) / span if span > 0 else 0.0

    frameBegin = data["frames"][i]
    frameEnd = data["frames"][j]

    with root.position.writeableArray() as w:
        if "root" in frameBegin and "root" in frameEnd:
            rootBegin = frameBegin["root"]
            rootEnd = frameEnd["root"]
            w[0][:3] = LinearRamp(rootBegin[:3], rootEnd[:3], local)
            w[0][2] = 0
            w[0][3:] = quat_slerp(rootBegin[3:], rootEnd[3:], local)
        else:
            w[0] = [0, 0, 0, 0, 0, 0, 1]

    for c in constraint:
        name = c.name.value
        begin = frameBegin["joints"].get(name, 0.0)
        end = frameEnd["joints"].get(name, 0.0)
        c.value = LinearRamp(begin, end, local)

    recording["factor"] = factor


def animDone(constraint, target, root, data, dt, effectors, factor):
    anim(constraint, target, root, data, dt, effectors, factor)


def writeOutput(frames, data):
    js = {"duration": data["duration"], "frames": frames}
    with open(OUTPUT, "w", encoding="UTF-8") as f:
        json.dump(js, f, indent=2)
    print(f"{len(frames)} frames sauvegardées")


def hasRootFrames(data):
    return len(data["frames"]) > 0 and all("root" in f for f in data["frames"])


def effectorsEntry(effectors, target, t, hasRoot=True):
    entry = {"t": t}
    if hasRoot:
        rootPose = np.array(target.position.value[0], dtype=float)
        entry["root"] = rootPose.tolist()
    else:
        rootPose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)
    p_root, q_root = rootPose[:3], rootPose[3:]

    for name, mo in effectors.items():
        p = np.array(mo.position.value[0], dtype=float)
        p = quat_rotate(quat_conj(q_root), p - p_root)
        entry[name[:2]] = p.tolist()
    return entry


class FrameRecorder(Sofa.Core.Controller):
    """Enregistre les positions en FIN de pas, une fois le solveur passé."""

    def __init__(self, data, dt, effectors, target, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.data = data
        self.dt = dt
        self.effectors = effectors
        self.target = target
        self.hasRoot = hasRootFrames(data)
        self.frames = []
        self.nextKey = 0
        self.done = False

    def onAnimateEndEvent(self, event):
        factor = recording["factor"]
        if self.done or factor is None:
            return

        while self.nextKey < len(self.dt) and factor >= self.dt[self.nextKey] - 1e-9:
            t = self.data["frames"][self.nextKey]["t"]
            self.frames.append(effectorsEntry(self.effectors, self.target, t, self.hasRoot))
            self.nextKey += 1

        if self.nextKey >= len(self.dt):
            self.done = True
            writeOutput(self.frames, self.data)


class PoseRecorder(Sofa.Core.Controller):
    """Animation d'une seule frame : applique la pose, laisse 1 pas puis enregistre."""

    def __init__(self, data, effectors, target, freeFlyer, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.data = data
        self.effectors = effectors
        self.target = target
        self.freeFlyer = freeFlyer
        self.hasRoot = hasRootFrames(data)
        self.steps = 0
        self.started = False
        self.done = False

    def onAnimateBeginEvent(self, event):
        if self.started:
            return
        self.started = True
        with self.freeFlyer.position.writeableArray() as w:
            if self.hasRoot:
                w[0] = self.data["frames"][0]["root"]
                w[0][2] = 0
            else:
                w[0] = [0, 0, 0, 0, 0, 0, 1]

    def onAnimateEndEvent(self, event):
        if self.done:
            return
        self.steps += 1
        if self.steps <= 1:
            return
        self.done = True
        entry = effectorsEntry(self.effectors, self.target,
                               self.data["frames"][0]["t"], self.hasRoot)
        writeOutput([entry], self.data)


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
        nameJoints = ["fr.hx", "fr.hy", "fr.kn", "fl.hx", "fl.hy", "fl.kn",
                      "hr.hx", "hr.hy", "hr.kn", "hl.hx", "hl.hy", "hl.kn"]

        data["duration"] = len(frames) * frameDuration
        data["frames"] = []
        for i in range(len(frames)):
            frame = {"t": i * frameDuration}
            frame["root"] = root_pos[i, :].tolist()
            frame["joints"] = {nameJoints[j]: joints[i, j] for j in range(len(nameJoints))}
            data["frames"].append(frame)
    else:
        data = js

    for value in data["frames"]:
        if data["duration"] != 0:
            dt.append(value["t"] / data["duration"])
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
    freeFlyer = simulation.SpotRobot.RootJoint.getObject("Free-Flyer")
    if len(data["frames"]) == 1 or data["duration"] == 0:
        rootnode.addObject(PoseRecorder(data, goalsEffectors, targetEffector.MechanicalObject,
                                        freeFlyer, name="PoseRecorder"))
    else:
        rootnode.addObject(FrameRecorder(data, dt, goalsEffectors, targetEffector.MechanicalObject,
                                         name="FrameRecorder"))
        animation.animate(anim, {"constraint": constraints,
                                 "target": targetEffector.MechanicalObject,
                                 "root": freeFlyer,
                                 "data": data, "dt": dt, "effectors": goalsEffectors},
                          duration=data["duration"], onDone=animDone)


def addEffector(robot):
    # Effector: Jambes (un point Vec3 au bout de chaque tibia)
    mechanicals = {}
    for node in robot.Joints.children:
        if node.name.value.endswith("kn"):
            effector = node.addChild("Effector")
            mecha = effector.addObject("MechanicalObject", name="MechanicalObject",
                                       position=[0, 0, -0.35], showObject=True, showObjectScale=0.1)
            effector.addObject("RigidMapping")
            mechanicals[node.name.value] = mecha
    return mechanicals


def addRootJointEffector(robot):
    # Effector: root_joint
    effector = robot.addChild('Effector')
    effector.addObject('MechanicalObject', template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1],
                       showObject=True, drawMode=2, showObjectScale=0.1)
    effector.addObject('RigidMapping')
    return effector


def addJointConstraint(robot, data, jointNodes, positions):
    constraints = []
    for i in range(len(positions)):
        jointName = jointNodes[i].name.value
        value = data["frames"][0]["joints"].get(jointName, 0)
        positions[i] = value
        joint = robot.addObject('JointConstraint', template='Vec1', name=jointName, index=i,
                                valueType="displacement",
                                value=value)
        constraints.append(joint)
        if MyGui is not None:
            MyGui.MyRobotWindow.addSetting(jointName, joint.value, -pi, pi)
    return constraints