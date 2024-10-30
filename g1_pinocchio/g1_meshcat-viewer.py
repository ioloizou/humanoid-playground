# This script is used to compute and visualize the inverse kinematics of the robot in MeshCat.

# TODO: 
# - Add option for movement to start from current position - Done
# - Make general kinematic sub tree simulate function - Done
# - Add visualization of the frame of the end effector with meshcat_shapes - Not finished
# - Make IK respect world frame, i need to try oMf.action

import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath
import meshcat_shapes

from pinocchio.visualize import MeshcatVisualizer
import time

def parse_urdf():
    """
    Parses the URDF file to build Pinocchio models.
    This function constructs the file paths for the URDF file and its associated
    mesh directory, then uses Pinocchio to build the model.
    Returns:
        tuple: A tuple containing:
            - model (pin.Model): The constat parts of the model.
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
    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=new_window)
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)

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
    Retrieve the starting and ending joint IDs for a specified kinematic tree.
    Args:
        tree_name (str): The name of the kinematic tree. 
                         Options are "left_hand", "right_hand", "left_leg", "right_leg".
    Returns:
        tuple: A tuple containing the starting joint ID and ending joint ID.
    """
    
    kinematic_trees = {
    "left_hand": {
        "starting_joint_id": model.getJointId("left_shoulder_pitch_joint"),
        "ending_joint_id": model.getJointId("left_wrist_roll_joint")
    },
    "right_hand": {
        "starting_joint_id": model.getJointId("right_shoulder_pitch_joint"),
        "ending_joint_id": model.getJointId("right_wrist_roll_joint")
    },
    "left_leg": {
        "starting_joint_id": model.getJointId("left_hip_pitch_joint"),
        "ending_joint_id": model.getJointId("left_ankle_roll_joint")
    },
    "right_leg": {
        "starting_joint_id": model.getJointId("right_hip_pitch_joint"),
        "ending_joint_id": model.getJointId("right_ankle_roll_joint")
    }
    }
    return kinematic_trees[tree_name]["starting_joint_id"], kinematic_trees[tree_name]["ending_joint_id"] 

def update_visual_frame(viz, model, data, ending_joint_id):
  # Need to make joint to frame. 
  ee_frame = viz.viewer["ending_joint_id"]

  oMf_hand = data.oMf[ending_joint_id]
  ee_frame.set_transform(oMf_hand.homogeneous)

  meshcat_shapes.frame(ee_frame)

def simulate_movement(model, data, viz, q_initial, starting_joint_id, ending_joint_id, v_desired, duration=1.0, sleep_time=0.1, first_time=True, new_window=False, q_previous=None):
    """
    Simulates the movement of a specified kinematic sub-tree for a given duration.
    
    Args:
        model (pin.Model): The Pinocchio model.
        data (pin.Data): The Pinocchio data.
        viz (MeshcatVisualizer): The Meshcat visualizer.
        q_initial (np.ndarray): The initial configuration of the robot.
        starting_joint_id (int): The joint ID where the kinematic sub-tree starts.
        ending_joint_id (int): The joint ID where the kinematic sub-tree ends.
        v_desired (np.ndarray): The desired velocity of the end-effector [vx, vy, vz, wx, wy, wz].
        duration (float): The duration of the simulation in seconds.
        sleep_time (float): The sleep time between each simulation step in seconds.
        first_time (bool): Flag to indicate if this is the first time the simulation is run.
        new_window (bool): Flag to indicate if a new window should be opened for the viewer.
        q_previous (np.ndarray, optional): The previous configuration of the robot. Defaults to None.
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

        # Transform the Jacobian from the left wrist to the world frame
        oMf = data.oMf[ending_joint_id]
        oRf = oMf.rotation

        # Calculate the Jacobian from left wrist
        J = pin.computeJointJacobian(model, data, q_current, ending_joint_id)
        
        # Align the Jacobian with the world frame
        J[:3, :] = oRf @ J[:3, :]
        J[3:, :] = oRf @ J[3:, :]            

        # Extract the relevant columns for the left hand movement
        J_reduced = J[:, model.joints[starting_joint_id].idx_q - 1 : model.joints[ending_joint_id].idx_q]  

        # Compute the pseudo-inverse of the reduced Jacobian
        J_reduced_pseudo_inv = np.linalg.pinv(J_reduced)

        q_dot_reduced = J_reduced_pseudo_inv @ v_desired
        
        # print("Reduced q: ", q_dot_reduced)

        q_dot_full = np.zeros(model.nv)

        q_dot_full[model.joints[starting_joint_id].idx_q - 1 : model.joints[ending_joint_id].idx_q] = q_dot_reduced
        
        q_current = pin.integrate(model, q_current, q_dot_full * 0.1)

        update_visual_frame(viz, model, data, ending_joint_id)
        display_robot_configuration(q_current, viz)
        time.sleep(sleep_time)  # Sleep to simulate real-time update
    
    return q_current 

# Parse the URDF file
model, collision_model, visual_model, data = parse_urdf()

# Initialize the viewer
new_window = True
viz = init_viewer(new_window)

# Robot with the feet on the meshcat ground    
q_feet = pin.neutral(model)
q_feet[2] = 0.8

display_robot_configuration(q_feet, viz)

# Choose one of the following kinematic trees: right hand, left hand, right leg or left leg
starting_joint_id, ending_joint_id  = get_kinematic_tree("right_hand")

# Initialize the configuration
q_current = q_feet.copy()

# Define the desired velocity of the left hand [vx, vy, vz, wx, wy, wz]
v_desired = np.array([0.1, 0., 0., 0., 0., 0.])

# Simulating the inverse kinematics
q_previous = simulate_movement(model, data, viz, q_current, starting_joint_id, ending_joint_id, v_desired)
