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

    settings, modelling, simulation = addHeader(rootnode, inverse=False, withCollision=True)
    rootnode.addObject("InteractiveCamera", name="camera", computeZClip=False, zFar=1000)
    rootnode.addObject("PointCloudRenderer", name="renderer", camera=rootnode.camera.linkpath)

    addSolvers(simulation, rayleighStiffness=0)
    rootnode.VisualStyle.displayFlags = ["showVisual"]

    # Units are in m, kg, s
    # Robot
    simulation.addChild(SpotRobot("data/model.urdf"))
    robot = simulation.SpotRobot.Robot

    # Direct problem
    names = robot.Joints.children
    positions = np.copy(robot.getMechanicalState().position.value)
    for i in range(len(positions)):
        jointName = names[i+1].name.value
        value = 0 if jointName not in spotInitConfiguration else spotInitConfiguration[jointName].pos_desired
        positions[i] = value
        joint = robot.addObject('JointConstraint', template='Vec1', name=jointName, index=i,
                                valueType="angle",
                                value=value)
        MyGui.MyRobotWindow.addSetting(jointName, joint.value, -pi, pi)
#
    tmp = np.copy(robot.dofs.position.value)
    robot.dofs.position.value = positions.tolist()

    cloud = rootnode.addChild(Cloud("spot", filename="splats/spot_export.ply"))
    rootnode.init()
    assign = SplatAssignment(robot, cloud)
    assign.assignment()
    rootnode.addObject(assign)

    rootnode.spot.visual.renderer.frames.setParent(robot.Joints.jointsDof.position)
    rootnode.spot.visual.renderer.initFrames.value = robot.Joints.jointsDof.position.value
    rootnode.spot.visual.renderer.frameIndices.setParent(rootnode.SplatAssignment.idx)
    robot.dofs.position.value = tmp

    # Masquer les meshes du robot
    for link in robot.Joints.children:
        if hasattr(link, 'Visual'):
            for vis in link.Visual.children:
                vis.visualModel.enable.value = False

                

    return