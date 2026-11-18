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
import json
import os

FOLDER = "data/animation/reverse"

def spotAnimation(target, frames, dt, factor):
    i = 0
    while i < len(dt) - 1 and dt[i + 1] <= factor:
        i += 1
    j = i + 1 if i < len(dt) - 1 else 0
    span = (dt[j] if j > i else 1.0) - dt[i]
    local = (factor - dt[i]) / span if span > 0 else 0.0

    with target.mapping.initialPoints.writeableArray() as w:
        for k, node in enumerate(target.children):
            name = node.name.value
            w[k] = LinearRamp(frames["frames"][i][name],
                              frames["frames"][j][name], local)

begin = [0, 0, 0, 0, 0, 0, 1]
def targetAnimation(target, frames, factor):
    global begin
    if factor == 0:
        begin = np.array(target.MechanicalObject.position.value[0], copy=True)

    with target.MechanicalObject.position.writeableArray() as w:
        end = begin[:3] + np.asarray(frames["position"])
        end = np.hstack((end, np.asarray(frames["rotation"])))
        w[0] = LinearRamp(begin, end, factor)


class AnimationController(Sofa.Core.Controller):
    def __init__(self, target, goals, manager, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.name = "AnimationController"
        self.done = False
        self.target = target
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

        self.animationsTarget = {key : {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]} for key in self.animations.keys()}
        for key, anim in self.animations.items():
            if len(anim["frames"]) > 1:
                arrfl = np.asarray([anim["frames"][i]["fl"] for i in range(len(anim["frames"]))])
                x = np.mean(abs(np.min(arrfl[:, 0]) - np.max(arrfl[:, 0])))
                position = [x, 0, 0]
                rotation = [0, 0, 0, 1]
                self.animationsTarget[key] = {"position" : position, "rotation" : rotation}
    
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
            if self.currentTargetAnim:
                self.manager.removeAnimation(self.currentTargetAnim)
            animation.animate(targetAnimation, {"target" : self.target, "frames" : self.animationsTarget[key]}, duration=frames["duration"], mode="loop")
            self.currentTargetAnim = self.manager.animations[-1]
            
            self.lastKey = key

        if self.index < len(self.animations.values()) - 1:
            self.index += 1
        else:
            self.index = 0

    def onAnimateBeginEvent(self, event):
        if self.done:
            return
        self.done = True
        self.onKeypressedEvent("")

def LinearRamp(beginValue=0.0, endValue=1.0, scale=0.5):
    begin = np.asarray(beginValue, dtype=float)
    end = np.asarray(endValue, dtype=float)
    return (end - begin) * scale + begin


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

    rootnode.addObject(AnimationController(target, goals, manager))
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
        min = urdf.joint_map[jointName].limit.lower 
        max = urdf.joint_map[jointName].limit.upper
        robot.addObject('JointActuator', template='Vec1', name='joint' + str(i), index=i,
                            maxAngle=max, minAngle=min, initAngle=(min+max)/2, 
                            maxAngleVariation=0.025)

def addSliddingActuator(node):
    # Sliding Actuator
    node.addObject('UniformMass', template='Rigid3', totalMass=16)
    node.addObject('SlidingActuator', name="SlideX", template='Rigid3', indices=0,
                        direction=[1, 0, 0, 0, 0, 0], maxDispVariation=0.01,
                        maxForce=1000, minForce=-1000)
    node.addObject('SlidingActuator', name="SlideY", template='Rigid3', indices=0,
                        direction=[0, 1, 0, 0, 0, 0], maxDispVariation=0.01,
                        maxForce=1000, minForce=-1000)
    node.addObject('SlidingActuator', name="SlideZ", template='Rigid3', indices=0,
                        direction=[0, 0, 1, 0, 0, 0], maxDispVariation=0.01,
                        maxForce=1000, minForce=-1000)
    
    node.addObject('SlidingActuator', name="RotateX", template='Rigid3', indices=0,
                        direction=[0, 0, 0, 1, 0, 0], maxDispVariation=0.001,
                        maxForce=100, minForce=-100)
    node.addObject('SlidingActuator', name="RotateY", template='Rigid3', indices=0,
                        direction=[0, 0, 0, 0, 1, 0], maxDispVariation=0.001,
                        maxForce=100, minForce=-100)
    node.addObject('SlidingActuator', name="RotateZ", template='Rigid3', indices=0,
                        direction=[0, 0, 0, 0, 0, 1], maxDispVariation=0.001,
                        maxForce=100, minForce=-100)

def addEffector(robot, goals):
    # Effector: Jambes
    i = 0
    for node in robot.Joints.children:
        if node.name.value.endswith("kn"):
            leg = node.name.value[:2]           
            effector = node.addChild("Effector")
            effector.addObject("MechanicalObject", name="MechanicalObject", position=[0, 0, -0.35])            
            effector.addObject('PositionEffector', indices=0, effectorGoal=goals.getChild(leg).dofs.position.getLinkPath(),
                            weight=1, useDirections=[1, 1, 1])
            effector.addObject("RigidMapping")
            i += 1

def addRootJointEffector(robot, target):
    # Effector: root_joint
    effector = robot.addChild('Effector')
    effector.addObject('MechanicalObject', template="Rigid3", position=[0, 0, 0, 0, 0 ,0, 1],
                       showObject=True, drawMode=2, showObjectScale=0.1)
    effector.addObject('PositionEffector', template="Rigid3", indices=0, effectorGoal=[0, 0, 0, 0, 0, 0, 1],
                       weight=5, useDirections=[1, 1, 1, 0, 1, 0])
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
