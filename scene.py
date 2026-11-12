
def printTree(node, indent=0):
    print(" " * indent + "[" + node.name.value + "]")
    for obj in node.objects:
        print(" " * (indent + 2) + "- " + obj.getClassName() + " : " + obj.name.value)
    for child in node.children:
        printTree(child, indent + 4)

def addCube(simulation):
    cube = simulation.addChild("SpotRobot")
    #cube.addObject('EulerImplicitIntegrationScheme')
    #cube.addObject('SparseLDLSolver')
    cube.addObject('GenericConstraintCorrection', linearSolver=simulation.solver.linkpath, ODESolver=simulation.EulerImplicitIntegrationScheme.linkpath)
    cube.addObject('MechanicalObject', template='Rigid3', position=[-1, 0, 0, 0, 0, 0, 1])
    cube.addObject('UniformMass', totalMass=0.01)
    # SlidingActuator needs a PartialFixedProjectiveConstraint to fix the other directions
    cube.addObject('PartialFixedProjectiveConstraint', template='Rigid3', fixedDirections=[0, 1, 1, 1, 1, 1])
    cube.addObject('SlidingActuator', template='Rigid3', indices=0, direction=[1, 0, 0, 0, 0, 0], maxDispVariation=0.01)

    visual = cube.addChild('Visu')
    visual.addObject('MeshOBJLoader', filename='mesh/cube.obj', scale=0.1)
    visual.addObject('OglModel', src='@MeshOBJLoader')
    visual.addObject('RigidMapping')

def createScene(rootnode):
    import os
    import numpy as np
    from modules.header import addHeader, addSolvers
    from modules.robot import SpotRobot
    from modules.cloud import Cloud
    from modules.miniskinning import SplatAssignment, addSkinningFrames
    import Sofa.ImGui as MyGui
    from math import pi
    from modules.robotconfigurations import spot_ctrl_joint_infos as spotInitConfiguration
    from urchin import URDF

    settings, modelling, simulation = addHeader(rootnode, inverse=True, withCollision=True)
    rootnode.addObject("InteractiveCamera", name="camera", computeZClip=False, zFar=10000)
    rootnode.addObject("PointCloudRenderer", name="renderer", camera=rootnode.camera.linkpath)

    addSolvers(simulation, rayleighStiffness=0)
    #simulation.solver.printLog = True
    #simulation.EulerImplicitIntegrationScheme.printLog = True
    #simulation.GenericConstraintCorrection.printLog = True
    rootnode.VisualStyle.displayFlags = ["showVisual"]

    # Units are in m, kg, s
    # Robot
    simulation.addChild(SpotRobot("data/model.urdf", freeFlyer=True))

    robot = simulation.SpotRobot.Robot
    robot.Joints.universe.activated = False

#
    ## Direct problem
    jointNodes = [n for n in robot.Joints.children
                  if n.name.value not in ("universe", "root_joint")]
    positions = np.copy(robot.getMechanicalState().position.value)
    for i in range(len(positions)):
        jointName = jointNodes[i].name.value
        value = 0 if jointName not in spotInitConfiguration else spotInitConfiguration[jointName].pos_desired
        positions[i] = value
        joint = robot.addObject('JointConstraint', template='Vec1', name=jointName, index=i,
                                valueType="displacement",
                                value=value)
        MyGui.MyRobotWindow.addSetting(jointName, joint.value, -pi, pi)

    tmp = np.copy(robot.dofs.position.value)
    robot.dofs.position.value = positions.tolist()

    # Actuator pour les jambes
    urdf = URDF.load('data/model.urdf')
    for i in range(len(robot.dofs.position)):
        jointName = jointNodes[i].name.value
        min = urdf.joint_map[jointName].limit.lower 
        max = urdf.joint_map[jointName].limit.upper
        robot.addObject('JointActuator', template='Vec1', name='joint' + str(i), index=i,
                            maxAngle=max, minAngle=min, initAngle=0, 
                            maxAngleVariation=0.01)

    # Sliding Actuator
    rootJoint = simulation.SpotRobot.RootJoint
    rootJoint.addObject('UniformMass', template='Rigid3', totalMass=16)
    rootJoint.addObject('PartialFixedProjectiveConstraint', template='Rigid3',
                        fixedDirections=[0, 1, 1, 1, 1, 1])
    rootJoint.addObject('SlidingActuator', template='Rigid3', indices=0,
                        direction=[1, 0, 0, 0, 0, 0], maxDispVariation=0.005)
    

    # Target pour le corps
    target = rootnode.addChild('Target')
    target.addObject('EulerImplicitIntegrationScheme', firstOrder=True)
    target.addObject('CGLinearSolver', tolerance=1e-5, iterations=20, threshold=1e-5)
    target.addObject('MechanicalObject', position=[1, 0, 0], showObject=True, drawMode=2, showObjectScale=0.1)
    target.addObject('PartialFixedProjectiveConstraint', fixedDirections=[0, 1, 1])
    target.addObject('UniformMass', totalMass=0.1)
    target.addObject('UncoupledConstraintCorrection')


    # Goals pour les jambes
    # avant gauche, avant droit, arrière gauche, arrière droit
    positions = [[0.5, 0.3, -0.3], 
                 [0.5, -0.3, -0.3], 
                 [0, 0.3, -0.3], 
                 [0, -0.3, -0.3]]
    goals = rootnode.addChild("Goals")
    for i in range(4):
        goal = goals.addChild('Goal' + str(i))
        goal.addObject('EulerImplicitIntegrationScheme', firstOrder=True)
        goal.addObject('CGLinearSolver', iterations=25, tolerance=1e-5, threshold=1e-5)
        goal.addObject('MechanicalObject', name='dofs', position=positions[i],
                        showObject=True, showObjectScale=0.08, drawMode=2)
        goal.addObject('UncoupledConstraintCorrection')


    # Effector: Jambes
    i = 0
    for node in robot.Joints.children:
        if node.name.value.endswith("kn"):
            effector = node.addChild("Effector")
            effector.addObject("MechanicalObject", name="MechanicalObject", position=[0, 0, -0.35, 0, 0, 0, 1])            
            effector.addObject('PositionEffector', indices=0, effectorGoal=goals.children[i].dofs.findData('position').getLinkPath(),
                            useDirections=[0, 0, 0])
            effector.addObject("RigidMapping")
            i += 1
    
    # Effector: root_joint
    effector = robot.Joints.root_joint.addChild('Effector')
    effector.addObject('MechanicalObject', position=[0, 0, 0],
                       showObject=True, drawMode=2, showObjectScale=0.1)
    effector.addObject('PositionEffector', indices=0, effectorGoal=[0, 0, 0],
                       weight=2, useDirections=[1, 0, 0])
    effector.addObject('RigidMapping')
    effector.PositionEffector.effectorGoal.setParent(target.MechanicalObject.position)
    # Ajoute du gs
    cloud = rootnode.addChild(Cloud("spot", filename="splats/spot_export.ply"))
    rootnode.init()
    rootnode.addObject(SplatAssignment(robot, cloud))

    rootnode.spot.visual.renderer.frames.setParent(robot.Joints.jointsDof.position)
    rootnode.spot.visual.renderer.initFrames.value = robot.Joints.jointsDof.position.value
    rootnode.spot.visual.renderer.frameIndices.setParent(rootnode.SplatAssignment.idx)
    robot.dofs.position.value = tmp

    # Masquer les meshes du robot
    #for link in robot.Joints.children:
    #    if hasattr(link, 'Visual'):
    #        for vis in link.Visual.children:
    #            vis.visualModel.enable.value = False
    return