# This script is used to compute and visualize the inverse kinematics of the robot in MeshCat.

import pinocchio as pin
import numpy as np
from os.path import dirname, join, abspath
import meshcat_shapes
import argparse

from pinocchio.visualize import MeshcatVisualizer
import time

def parse_urdf():
    """
    Parses the URDF file to build Pinocchio models.
    This function constructs the file paths for the URDF file and its associated
    mesh directory, then uses Pinocchio to build the model.
    Returns:
        tuple: A tuple containing:
            - model (pin.Model): The constant parts of the model.
            - collision_model (pin.GeometryModel): The collision model.
            - visual_model (pin.GeometryModel): The visual model.
            - data (pin.Data): The varying part of the model.
    """
    
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "g1_description")

    model_path = str(pinocchio_model_dir)
    mesh_dir = pinocchio_model_dir
    urdf_filename = "g1_23dof.urdf"
    urdf_model_path = join(model_path, urdf_filename)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    )

    data = model.createData()
    
    return model, collision_model, visual_model, data

def init_viewer(new_window=True):
    """
    Initializes a MeshCat visualizer.
    Starts a new MeshCat server and client, and initializes the viewer with the given model, collision model, and visual model.
    If the MeshCat library is not installed, an ImportError is caught, and an error message is printed before exiting the program.

    Args:
        new_window (bool): A flag to indicate whether to open a new window for the viewer. Default is True.

    Returns:
        MeshcatVisualizer: An instance of the Meshcat visualizer.
    """

    # Start a new MeshCat server and client.
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=new_window)

    # Load the robot in the viewer.
    viz.loadViewerModel()
    return viz

def display_robot_configuration(q, viz):
    """
    Loads the robot model into the viewer and displays its configuration.

    Args:
        q (np.ndarray): The robot configuration.
        viz (MeshcatVisualizer): The Meshcat visualizer.
    """    
    # Display a robot configuration.
    viz.display(q)
    viz.displayVisuals(True)

def get_kinematic_tree(tree_name):
    """
    Retrieve the starting and ending joint IDs and frame IDs for a specified kinematic tree.
    Args:
        tree_name (str): The name of the kinematic tree. 
                         Options are "left_hand", "right_hand", "left_leg", "right_leg".
    Returns:
        tuple: A tuple containing the starting joint ID, ending joint ID, starting frame ID, and ending frame ID.
    """
    
    kinematic_trees = {
    "left_hand": {
        "starting_joint_id": model.getJointId("left_shoulder_pitch_joint"),
        "ending_joint_id": model.getJointId("left_wrist_roll_joint"),
        "starting_frame_id": model.getFrameId("left_shoulder_pitch_joint"),
        "ending_frame_id": model.getFrameId("left_wrist_roll_joint")
    },
    "right_hand": {
        "starting_joint_id": model.getJointId("right_shoulder_pitch_joint"),
        "ending_joint_id": model.getJointId("right_wrist_roll_joint"),
        "starting_frame_id": model.getFrameId("right_shoulder_pitch_joint"),
        "ending_frame_id": model.getFrameId("right_wrist_roll_joint")
    },
    "left_leg": {
        "starting_joint_id": model.getJointId("left_hip_pitch_joint"),
        "ending_joint_id": model.getJointId("left_ankle_roll_joint"),
        "starting_frame_id": model.getFrameId("left_hip_pitch_joint"),
        "ending_frame_id": model.getFrameId("left_ankle_roll_joint")
    },
    "right_leg": {
        "starting_joint_id": model.getJointId("right_hip_pitch_joint"),
        "ending_joint_id": model.getJointId("right_ankle_roll_joint"),
        "starting_frame_id": model.getFrameId("right_hip_pitch_joint"),
        "ending_frame_id": model.getFrameId("right_ankle_roll_joint")
    }
    }
    tree = kinematic_trees[tree_name]
    return tree["starting_joint_id"], tree["ending_joint_id"], tree["starting_frame_id"], tree["ending_frame_id"]

def compute_subtree_joint_velocities(model, data, q_current, starting_joint_id, ending_joint_id, ending_frame_id, v_desired):
    """
    Computes the joint velocities for a given kinematic sub-tree to achieve the desired end-effector velocity.

    Args:
        model (pin.Model): The Pinocchio model.
        data (pin.Data): The Pinocchio data.
        q_current (np.ndarray): The current configuration of the robot.
        starting_joint_id (int): The joint ID where the kinematic sub-tree starts.
        ending_joint_id (int): The joint ID where the kinematic sub-tree ends.
        ending_frame_id (int): The frame ID where the kinematic sub-tree ends.
        v_desired (np.ndarray): The desired velocity of the end-effector [vx, vy, vz, wx, wy, wz].

    Returns:
        np.ndarray: The updated configuration of the robot.
    """
    J = pin.computeFrameJacobian(model, data, q_current, ending_frame_id, pin.LOCAL_WORLD_ALIGNED)

    # Extract the relevant columns for the end effector movement
    J_reduced = J[:, model.joints[starting_joint_id].idx_q - 1 : model.joints[ending_joint_id].idx_q]  

    # Compute the pseudo-inverse of the reduced Jacobian
    J_reduced_pseudo_inv = np.linalg.pinv(J_reduced)

    q_dot_reduced = J_reduced_pseudo_inv @ v_desired
    
    q_dot_full = np.zeros(model.nv)

    q_dot_full[model.joints[starting_joint_id].idx_q - 1 : model.joints[ending_joint_id].idx_q] = q_dot_reduced
    
    q_current = pin.integrate(model, q_current, q_dot_full * 0.1)
    
    return q_current

def compute_whole_body_inverse_kinematics(model, data, q_current, v_desired):
    """
    Calculates the whole body inverse kinematics for the robot.

    Args:
        model (pin.Model): The Pinocchio model.
        data (pin.Data): The Pinocchio data.
        q_current (np.ndarray): The current configuration of the robot.
        v_desired (np.ndarray): The desired velocity of the end-effector [vx, vy, vz, wx, wy, wz].

    Returns:
        np.ndarray: The updated configuration of the robot.
    """
    reference_frame = pin.WORLD

    # Left leg
    _, _, _, left_leg_ending_frame_id = get_kinematic_tree("left_leg")
    J_left_leg = pin.computeFrameJacobian(model, data, q_current, left_leg_ending_frame_id, reference_frame)

    # Right leg
    _, _, _, right_leg_ending_frame_id = get_kinematic_tree("right_leg")
    J_right_leg = pin.computeFrameJacobian(model, data, q_current, right_leg_ending_frame_id, reference_frame)

    # Right hand
    _, _, _, right_hand_ending_frame_id = get_kinematic_tree("right_hand")
    J_right_hand = pin.computeFrameJacobian(model, data, q_current, right_hand_ending_frame_id, reference_frame)

    # Stacking the Jacobians
    J = np.vstack((J_left_leg, J_right_leg, J_right_hand))

    # Stacking the desired velocities
    v_desired = np.hstack((np.zeros(6), np.zeros(6), v_desired))

    # Computing the pseudo-inverse of the Jacobian
    J_pseudo_inv = np.linalg.pinv(J)

    # Computing the joint velocities
    q_dot = J_pseudo_inv @ v_desired

    # Integrating the joint velocities
    q_current = pin.integrate(model, q_current, q_dot * 0.1)

    return q_current

def update_visual_frame(viz, model, data, ending_joint_frame_id, whole_body_ik):
    """
    Updates the visual frame in the viewer.

    Args:
        viz (MeshcatVisualizer): The Meshcat visualizer.
        model (pin.Model): The Pinocchio model.
        data (pin.Data): The Pinocchio data.
        ending_joint_frame_id (int): The frame ID where the kinematic sub-tree ends.
        whole_body_ik (bool): Flag to indicate if whole body inverse kinematics is used.
    """
    # If the whole body is run only the right hand is used
    if whole_body_ik:
        ending_joint_frame_id = model.getFrameId("right_wrist_roll_joint")

    ee_frame = viz.viewer[str(ending_joint_frame_id)]

    oMf_hand = data.oMf[ending_joint_frame_id]
    ee_frame.set_transform(oMf_hand.homogeneous)

    meshcat_shapes.frame(ee_frame)

def simulate_movement(model, 
                      data, 
                      viz, 
                      q_initial, 
                      starting_joint_id, 
                      ending_joint_id,
                      starting_frame_id,
                      ending_frame_id, 
                      v_desired, 
                      duration=1.0, 
                      sleep_time=0.1, 
                      first_time=True, 
                      new_window=False, 
                      q_previous=None,
                      whole_body_ik=False):
    """
    Simulates the movement of a specified kinematic sub-tree for a given duration.
    
    Args:
        model (pin.Model): The Pinocchio model.
        data (pin.Data): The Pinocchio data.
        viz (MeshcatVisualizer): The Meshcat visualizer.
        q_initial (np.ndarray): The initial configuration of the robot.
        starting_joint_id (int): The joint ID where the kinematic sub-tree starts.
        ending_joint_id (int): The joint ID where the kinematic sub-tree ends.
        starting_frame_id (int): The frame ID where the kinematic sub-tree starts.
        ending_frame_id (int): The frame ID where the kinematic sub-tree ends.
        v_desired (np.ndarray): The desired velocity of the end-effector [vx, vy, vz, wx, wy, wz].
        duration (float): The duration of the simulation in seconds.
        sleep_time (float): The sleep time between each simulation step in seconds.
        first_time (bool): Flag to indicate if this is the first time the simulation is run.
        new_window (bool): Flag to indicate if a new window should be opened for the viewer.
        q_previous (np.ndarray, optional): The previous configuration of the robot. Defaults to None.
        whole_body_ik (bool): Flag to indicate if whole body inverse kinematics should be used. Defaults to False.
    """
    # Wait 4 seconds before starting the simulation if launch for first time
    if first_time or new_window:
        time.sleep(4)
    else:
        time.sleep(1)

    # Initialize the configuration
    if q_previous is None:
        q_current = q_initial.copy()
    else:
        q_current = q_previous.copy()

    # Run the simulation for the specified duration
    start_time = time.time()
    
    while time.time() - start_time < duration:
        
        pin.forwardKinematics(model, data, q_current)
        pin.updateFramePlacements(model, data)            

        if whole_body_ik:
            q_current = compute_whole_body_inverse_kinematics(model, data, q_current, v_desired)
        else:
            q_current = compute_subtree_joint_velocities(model, data, q_current, starting_joint_id, ending_joint_id, ending_frame_id, v_desired)

        update_visual_frame(viz, model, data, ending_frame_id, whole_body_ik)
        display_robot_configuration(q_current, viz)
        
        time.sleep(sleep_time)  # Sleep to simulate real-time update
    
    return q_current 

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simulate robot movement with inverse kinematics.")
    parser.add_argument("--whole_body_ik", action="store_true", help="Use whole body inverse kinematics.")
    args = parser.parse_args()

    # Parse the URDF file
    model, collision_model, visual_model, data = parse_urdf()

    # Initialize the viewer
    viz = init_viewer(new_window=True)

    # Robot half sitting with the feet on the meshcat ground    
    q0 = np.array([
        # floating base
        0.0, 0.0, 0.72,  # reference base linear
        0.0, 0.0, 0.0, 1.0,  # reference base quaternion
        # left leg
        -0.6, 0.0, 0.0, 1.2, -0.6, 0.0,
        # right leg
        -0.6, 0.0, 0.0, 1.2, -0.6, 0.0,
        # waist
        0.0,
        # left shoulder
        0.0, 0.0, 0.0, 0.0, 0.0,
        # right shoulder
        0.0, 0.0, 0.0, 0.0, 0.0,
    ])

    display_robot_configuration(q0, viz)

    # Select the desired kinematic tree
    selected_tree = "left_hand"

    starting_joint_id, ending_joint_id, starting_frame_id, ending_frame_id = get_kinematic_tree(selected_tree)

    # Define the desired velocity of the selected kinematic tree or for the right hand if whole body [vx, vy, vz, wx, wy, wz]
    v_desired = np.array([0.1, 0., 0., 0., 0., 0.])  # Example velocity

    q_current = q0.copy()

    # Simulate the movement and update the frame
    q_previous = simulate_movement(model, data, viz, q0, starting_joint_id, ending_joint_id, starting_frame_id, ending_frame_id, v_desired, whole_body_ik=args.whole_body_ik)

