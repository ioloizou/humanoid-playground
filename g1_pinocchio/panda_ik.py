import pinocchio as pin
import numpy as np

import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
import meshcat_shapes
import time

def init_viewer(new_window=True):
    """
    Initializes a MeshCat visualizer.
    Starts a new MeshCat server and client, and initializes the viewer with the given model, collision model, and visual model.

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

def update_visual_frame(viz, model, data, frame_name):
  """Updates the visual frame in the viewer."""
  
  ee_frame = viz.viewer[frame_name]

  oMf_hand = data.oMf[model.getFrameId(frame_name)]
  ee_frame.set_transform(oMf_hand.homogeneous)

  meshcat_shapes.frame(ee_frame)

def simulate_movement(viz, model, data, q_initial, v_desired, new_window, duration=1, steps=100, q_previous=None):
    """
    Simulates the movement of the robot for a specified duration.

    Args:
        viz (MeshcatVisualizer): The Meshcat visualizer.
        model (pinocchio.Model): The robot model.
        data (pinocchio.Data): The robot data.
        q0 (np.ndarray): The initial configuration of the robot.
        new_window (bool): A flag to indicate whether to open a new window for the viewer.
        duration (float): The duration of the simulation in seconds.
        steps (int): The number of steps to simulate the movement.

    Returns:
        None
    """
    # Wait 4 seconds before starting the simulation if launch for first time
    if new_window:
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
    dt = duration / steps

    while time.time() - start_time < duration:
        pin.forwardKinematics(model, data, q_current)
        pin.updateFramePlacements(model, data)

        # Frame name of the end effector
        frame_name = "panda_hand"
        idx_tool = model.getFrameId(frame_name)

        J = pin.computeFrameJacobian(model, data, q_current, idx_tool, pin.LOCAL_WORLD_ALIGNED) 

        qdot = np.linalg.pinv(J) @ v_desired

        q_current = pin.integrate(model, q_current, qdot * dt)

        update_visual_frame(viz, model, data, frame_name)
        display_robot_configuration(q_current, viz)

        time.sleep(dt)  # Sleep to simulate real-time update
    
    return q_current

robot = example_robot_data.load("panda")

model = robot.model
data = model.createData()
collision_model = robot.collision_model
visual_model = robot.visual_model

# Initialize the viewer
new_window = True
viz = init_viewer(new_window)

# Set the initial configuration of the robot
q0 =np.array([0,-0.785398163397, 0, -2.35619449019, 0,1.57079632679,0.785398163397, 0., 0.])

display_robot_configuration(q0, viz)

# Define the desired velocity of the left hand [vx, vy, vz, wx, wy, wz]
v_desired = np.array([0.5, 0., 0., 0., 0., 0.])

q_current = q0.copy()

q_previous = simulate_movement(viz, model, data, q_current, v_desired, new_window)

# To run on terminal:
# q_previous = simulate_movement(viz, model, data, q_current, np.array([0.1, 0., 0., 0., 0., 0.]), new_window=False, q_previous=q_previous)