# This script is used to compute and visualize the inverse kinematics of the robot in MeshCat.

import pinocchio as pin
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import dirname, join, abspath
from IPython import get_ipython

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

def init_viewer():
    """
    Initializes a MeshCat visualizer.
    Starts a new MeshCat server and client, and initializes the viewer with the given model, collision model, and visual model.
    If the MeshCat library is not installed, an ImportError is caught, and an error message is printed before exiting the program.
    Returns:
        MeshcatVisualizer: An instance of the Meshcat visualizer.
    """

    # Start a new MeshCat server and client.
    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=False)
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)

    # Load the robot in the viewer.
    viz.loadViewerModel()
    return viz

def display_robot_configuration(q):
    """
    Loads the robot model into the viewer and displays its neutral configuration.
    This function initializes the viewer with the robot model and sets the robot
    to its default (neutral) configuration for visualization.
    """    
    # Display a robot configuration.
    viz.display(q)
    viz.displayVisuals(True)

def simulate_left_hand_movement(model, data, viz, q_initial, starting_joint_id, ending_joint_id, v_desired, duration=1.0, sleep_time=0.1, first_time=True):
    """
    Simulates the movement of the left hand for a given duration.
    
    Args:
        model (pin.Model): The Pinocchio model.
        data (pin.Data): The Pinocchio data.
        viz (MeshcatVisualizer): The Meshcat visualizer.
        q_initial (np.ndarray): The initial configuration of the robot.
        starting_joint_id (int): The joint ID of the left shoulder.
        ending_joint_id (int): The joint ID of the left wrist.
        duration (float): The duration of the simulation in seconds.
        sleep_time (float): The sleep time between each simulation step in seconds.
    """
    # Wait 4 seconds before starting the simulation if launch for first time
    if first_time:
        time.sleep(4)
    else:
        time.sleep(1)

    # Initialize the configuration
    q_current_left_hand = q_initial.copy()

    # Run the simulation for the specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        
        pin.forwardKinematics(model, data, q_current_left_hand)

        # Transform the Jacobian from the left wrist to the world frame
        oMwr = data.oMf[ending_joint_id]
        oRwr = oMwr.rotation

        # Calculate the Jacobian from left wrist
        J_left_hand = pin.computeJointJacobian(model, data, q_current_left_hand, ending_joint_id)
        
        # Align the Jacobian with the world frame
        J_left_hand[:3, :] = oRwr @ J_left_hand[:3, :]
        J_left_hand[3:, :] = oRwr @ J_left_hand[3:, :]

        # Extract the relevant columns for the left hand movement
        J_left_hand_reduced = J_left_hand[:, model.joints[starting_joint_id].idx_q - 1 : model.joints[ending_joint_id].idx_q]  

        # Compute the pseudo-inverse of the reduced Jacobian
        J_left_hand_pseudo_inv = np.linalg.pinv(J_left_hand_reduced)

        q_dot_left_hand = J_left_hand_pseudo_inv @ v_desired
        
        print("Reduced q: ", q_dot_left_hand)

        q_dot_full_left_hand = np.zeros(model.nv)

        q_dot_full_left_hand[model.joints[starting_joint_id].idx_q - 1 : model.joints[ending_joint_id].idx_q] = q_dot_left_hand
        
        q_current_left_hand = pin.integrate(model, q_current_left_hand, q_dot_full_left_hand * 0.1)

        viz.display(q_current_left_hand)
        time.sleep(sleep_time)  # Sleep to simulate real-time update

model = parse_urdf()[0]
collision_model = parse_urdf()[1]
visual_model = parse_urdf()[2]
data = parse_urdf()[3]

viz = init_viewer()

q_neutral = pin.neutral(model)

display_robot_configuration(q_neutral)
    
# Robot with the feet on the meshcat ground    
q_feet = q_neutral.copy()
q_feet[2] = 0.8

display_robot_configuration(q_feet)

# Get the kinematic tree of the left hand
starting_joint_id = model.getJointId("left_shoulder_pitch_joint")
ending_joint_id = model.getJointId("left_wrist_roll_joint")

# Initialize the configuration
q_current_left_hand = q_feet.copy()

# Define the desired velocity of the left hand
v_left_hand = np.array([0., 0., 0., 1., 0., 0.])

# Call the function with the appropriate arguments
simulate_left_hand_movement(model, data, viz, q_feet, starting_joint_id, ending_joint_id, v_left_hand)
