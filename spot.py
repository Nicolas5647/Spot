from modules.robot import SpotRobot
from math import pi
from modules.robotconfigurations import spot_ctrl_joint_infos as spotInitConfiguration
from urchin import URDF
from modules.cloud import Cloud
from modules.miniskinning import SplatAssignment
from splib3 import animation
import Sofa
import Sofa.ImGui as MyGui
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import json
import os

FOLDER = "data/animation/reverse"


def LinearRamp(beginValue=0.0, endValue=1.0, scale=0.5):
    begin = np.asarray(beginValue, dtype=float)
    end = np.asarray(endValue, dtype=float)
    return (end - begin) * scale + begin

def findKeyframes(dt, factor):
    i = 0
    while i < len(dt) - 1 and dt[i + 1] <= factor:
        i += 1
    j = i + 1 if i < len(dt) - 1 else 0
    span = (dt[j] if j > i else 1.0) - dt[i]
    local = (factor - dt[i]) / span if span > 0 else 0.0
    return i, j, min(max(local, 0.0), 1.0)


def spotAnimation(target, frames, dt, factor):
    # Les pieds sont enregistrés dans le repère du root : ce sont directement
    # les coordonnées locales du RigidMapping Target -> Goals
    i, j, local = findKeyframes(dt, factor)

    with target.mapping.initialPoints.writeableArray() as w:
        for k, node in enumerate(target.children):
            name = node.name.value
            w[k] = LinearRamp(frames["frames"][i][name],
                              frames["frames"][j][name], local)


def composePose(begin, relPosition, relRotation):
    # begin ∘ rel : la trajectoire relative est appliquée à partir de la pose de départ
    rotationBegin = R.from_quat(begin[3:])
    position = begin[:3] + rotationBegin.apply(relPosition)
    rotation = (rotationBegin * relRotation).as_quat()
    return position, rotation


def targetAnimation(target, frames, dt, state, factor):
    mo = target.MechanicalObject

    # Nouvelle boucle (ou premier appel) : on repart de la pose actuelle de la cible
    if state["begin"] is None or (state["previous"] is not None and factor < state["previous"]):
        state["begin"] = np.array(mo.position.value[0], dtype=float, copy=True)
    state["previous"] = factor
    begin = state["begin"]

    if len(frames) == 1:
        # Déplacement unique réparti sur toute la durée de l'animation
        deplacement = np.asarray(frames[0], dtype=float)
        relPosition = deplacement[:3] * factor
        relRotation = Slerp([0, 1], R.from_quat([[0, 0, 0, 1], deplacement[3:]]))(factor)
    else:
        i, j, local = findKeyframes(dt, factor)
        if j < i:
            # Dernier segment : on garde la dernière pose, la boucle suivante repartira d'ici
            j, local = i, 0.0
        a = np.asarray(frames[i], dtype=float)
        b = np.asarray(frames[j], dtype=float)
        relPosition = LinearRamp(a[:3], b[:3], local)
        relRotation = Slerp([0, 1], R.from_quat([a[3:], b[3:]]))(local)

    position, rotation = composePose(begin, relPosition, relRotation)
    with mo.position.writeableArray() as w:
        w[0][:3] = position
        w[0][3:] = rotation

def hasRootFrames(anim):
    return len(anim["frames"]) > 0 and all("root" in f for f in anim["frames"])


def relativeRootFrames(anim):
    # Pose du root de chaque frame exprimée par rapport à la première frame
    roots = np.array([f["root"] for f in anim["frames"]], dtype=float)
    rotation0 = R.from_quat(roots[0, 3:])
    relPositions = rotation0.inv().apply(roots[:, :3] - roots[0, :3])
    relRotations = (rotation0.inv() * R.from_quat(roots[:, 3:])).as_quat()
    return np.hstack([relPositions, relRotations]).tolist()


class AnimationController(Sofa.Core.Controller):
    def __init__(self, target, goals, manager, robotRoot, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "AnimationController"
        self.done = False
        self.target = target
        self.robotRoot = robotRoot
        self.goals = goals
        self.manager = manager
        self.index = 0
        self.lastKey = ""

        self.currentSpotAnim = None
        self.currentTargetAnim = None
        self.setup()

    def setup(self):
        self.animations = {}
        for name in os.listdir(FOLDER):
            path = os.path.join(FOLDER, name)
            with open(path, "r", encoding="UTF-8") as file:
                self.animations[name[:-5]] = json.load(file)

        self.animationsTarget = {key : [[0, 0, 0, 0, 0, 0, 1]] for key in self.animations.keys()}
        for key, anim in self.animations.items():
            if len(anim["frames"]) > 1:
                if hasRootFrames(anim):
                    self.animationsTarget[key] = relativeRootFrames(anim)
                else:
                    arrfl = np.asarray([anim["frames"][i]["fl"] for i in range(len(anim["frames"]))])
                    x = np.mean(abs(np.min(arrfl[:, 0]) - np.max(arrfl[:, 0])))
                    position = [x, 0, 0]
                    rotation = [0, 0, 0, 1]
                    self.animationsTarget[key] = [position + rotation]

    def onKeypressedEvent(self, event):
        key, value = list(self.animations.items())[self.index]
        frames = self.animations[key]
        dt = []
        for value in frames["frames"]:
            if frames["duration"] != 0:
                dt.append(value["t"]/frames["duration"])
            else:
                dt.append(value["t"])
        print(f"{key} animation")


        if self.currentSpotAnim:
            self.manager.removeAnimation(self.currentSpotAnim)

        animation.animate(spotAnimation, {"target" : self.goals, "frames" : frames, "dt" : dt}, duration=frames["duration"], mode="loop")
        self.currentSpotAnim = self.manager.animations[-1]

        if key in self.animationsTarget and self.lastKey != key:
            self.lastKey = key
            if self.currentTargetAnim:
                self.manager.removeAnimation(self.currentTargetAnim)

            state = {"begin": None, "previous": None}
            animation.animate(targetAnimation, {"target" : self.target, "frames" : self.animationsTarget[key], "dt": dt, "state": state},
                              duration=frames["duration"], mode="loop")
            self.currentTargetAnim = self.manager.animations[-1]


        if self.index < len(self.animations.values()) - 1:
            self.index += 1
        else:
            self.index = 0

    def onAnimateBeginEvent(self, event):
        if self.done:
            return
        self.done = True
        self.onKeypressedEvent("")


def tmpAnimation(target, factor):
    with target.position.writeableArray() as w:
        w[0][3:7] = [0, np.sin((factor - 0.5) * pi/16), 0, np.cos((factor - 0.5) * pi/16)]
        #w[0][:3] = [np.cos(factor) - 0.5, 0, 0]

def setupRobot(rootnode, simulation, manager):
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

    target = addTarget(rootnode)
    goals = addGoals(target)

    addEffector(robot, goals)
    addRootJointEffector(robot.Joints.root_joint, target)
    removeConstraint(robot, constraints)

    robotRoot = simulation.SpotRobot.RootJoint.getMechanicalState()
    rootnode.addObject(AnimationController(target, goals, manager, robotRoot))
    #animation.animate(tmpAnimation, {"target" : target.MechanicalObject}, mode="pingpong", duration=2)

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
        min = -pi #urdf.joint_map[jointName].limit.lower
        max = pi #urdf.joint_map[jointName].limit.upper
        robot.addObject('JointActuator', template='Vec1', name='joint' + str(i), index=i,
                            maxAngle=max, minAngle=min, initAngle=(min+max)/2,
                            maxAngleVariation=0.05)

def addSliddingActuator(node):
    node.addObject('UniformMass', template='Rigid3', totalMass=16)
    node.addObject('SlidingActuator', name="SlideX", template='Rigid3', indices=0,
                        direction=[1, 0, 0, 0, 0, 0], maxDispVariation=0.01,
                        maxForce=10000, minForce=-10000)
    node.addObject('SlidingActuator', name="SlideY", template='Rigid3', indices=0,
                        direction=[0, 1, 0, 0, 0, 0], maxDispVariation=0.01,
                        maxForce=10000, minForce=-10000)
    node.addObject('SlidingActuator', name="SlideZ", template='Rigid3', indices=0,
                        direction=[0, 0, 1, 0, 0, 0], maxDispVariation=0.01,
                        maxForce=10000, minForce=-10000)

    node.addObject('SlidingActuator', name="RotateX", template='Rigid3', indices=0,
                        direction=[0, 0, 0, 1, 0, 0], maxDispVariation=0.01,
                        maxForce=1000, minForce=-1000)
    node.addObject('SlidingActuator', name="RotateY", template='Rigid3', indices=0,
                        direction=[0, 0, 0, 0, 1, 0], maxDispVariation=0.01,
                        maxForce=1000, minForce=-1000)
    node.addObject('SlidingActuator', name="RotateZ", template='Rigid3', indices=0,
                        direction=[0, 0, 0, 0, 0, 1], maxDispVariation=0.01,
                        maxForce=1000, minForce=-1000)

def addEffector(robot, goals):
    # Effector: Jambes
    for node in robot.Joints.children:
        if node.name.value.endswith("kn"):
            leg = node.name.value[:2]
            effector = node.addChild("Effector")
            effector.addObject("MechanicalObject", name="MechanicalObject", position=[0, 0, -0.35])
            effector.addObject('PositionEffector', indices=0, effectorGoal=goals.getChild(leg).dofs.position.getLinkPath(),
                            weight=1, useDirections=[1, 1, 1])
            effector.addObject("RigidMapping")

def addRootJointEffector(robot, target):
    # Effector: root_joint
    effector = robot.addChild('Effector')
    effector.addObject('MechanicalObject', template="Rigid3", position=[0, 0, 0, 0, 0 ,0, 1],
                       showObject=True, drawMode=2, showObjectScale=0.1)
    effector.addObject('PositionEffector', template="Rigid3", indices=0, effectorGoal=[0, 0, 0, 0, 0, 0, 1],
                       weight=5, useDirections=[1, 1, 1, 1, 1, 1])
    effector.addObject('RigidMapping')
    effector.PositionEffector.effectorGoal.setParent(target.MechanicalObject.position)

def addGoals(rootnode):
    # Goals pour les jambes
    # avant gauche, avant droit, arrière gauche, arrière droit
    positions = [[0.594152228928174, 0.1659450060830276, -0.5408313014780995],
                 [0.5941522289614842, -0.1659449980128254, -0.540831302760199],
                 [-0.0015037042913236265, 0.1659449964009789, -0.5335857022040742],
                 [-0.0015037043397926886, -0.1659450018970651, -0.5335857006252905]]
    names = ["fl", "fr", "hl", "hr"]
    goals = rootnode.addChild("Goals")
    goals.addObject('EulerImplicitIntegrationScheme', firstOrder=True)
    goals.addObject('CGLinearSolver', iterations=25, tolerance=1e-5, threshold=1e-5)
    goals.addObject('MechanicalObject', name='dofs', template='Vec3',
                    position=positions,
                    showObject=True, showObjectScale=0.05, drawMode=2)
    goals.addObject('UncoupledConstraintCorrection')
    goals.addObject('RigidMapping', name="mapping")

    for i, name in enumerate(names):
        g = goals.addChild(name)
        g.addObject('MechanicalObject', name='dofs', template='Vec3')
        g.addObject('SubsetMapping', input=goals.dofs.getLinkPath(),
                    output=g.dofs.getLinkPath(), indices=[i], handleTopologyChange=False)
    return goals

def addTarget(rootnode):
    # Target pour le corps
    target = rootnode.addChild('Target')
    target.addObject('EulerImplicitIntegrationScheme', firstOrder=True)
    target.addObject('CGLinearSolver', tolerance=1e-5, iterations=20, threshold=1e-5)
    target.addObject('MechanicalObject', template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1], showObject=True, showObjectScale=0.15)
    target.addObject('PartialFixedProjectiveConstraint', template='Rigid3',
                 fixedDirections=[0, 0, 1, 0, 0, 0])
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