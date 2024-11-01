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

  meshcat_shapes.frame(ee_frame, axis_length=0.2)

def compute_ik(model, data, q_current, v_desired, frame_name, dt):
    """
    Computes the next configuration of the robot given the desired velocity using inverse kinematics.

    Args:
    model (pinocchio.Model): The robot model.
    data (pinocchio.Data): The robot data.
    q_current (np.ndarray): The current configuration of the robot.
    v_desired (np.ndarray): The desired velocity of the end effector.
    frame_name (str): The name of the frame to control.
    dt (float): The time step for integration.

    Returns:
    np.ndarray: The next configuration of the robot.
    """
    idx_tool = model.getFrameId(frame_name)
    J = pin.computeFrameJacobian(model, data, q_current, idx_tool, pin.LOCAL_WORLD_ALIGNED)
    qdot = np.linalg.pinv(J) @ v_desired
    q_next = pin.integrate(model, q_current, qdot * dt)
    return q_next

def compute_close_loop_ik(model, data, q_current, oMgoal, frame_name_id, dt):

    # Get the current placement of the end effector
    oMtool = data.oMf[frame_name_id]

    # Compute the error
    toolMgoal = oMtool.actInv(oMgoal)
    error = pin.log6(toolMgoal).vector

    # Compute the Jacobian
    J = pin.computeFrameJacobian(model, data, q_current, idx_tool, pin.LOCAL)

    # Map the Jacobian to the tangent space
    J = pin.Jlog6(toolMgoal.inverse()) @ J
    
    # Compute the pseudo-inverse of the Jacobian
    J_pinv = np.linalg.pinv(J)

    gain = 1.5

    # Compute the joint velocity
    qdot = gain * J_pinv @ error

    # Integrate the joint velocity
    q_next = pin.integrate(model, q_current, qdot * dt)

    # Compute the error norm
    error_norm = np.linalg.norm(error)

    return q_next, error_norm

def simulate_movement(viz, model, data, q_initial, v_desired, new_window, duration=10, steps=1000, q_previous=None):
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

    while (time.time() - start_time < duration):
        pin.forwardKinematics(model, data, q_current)
        pin.updateFramePlacements(model, data)

        # Uncomment the line below to use the simple IK method and comment the next line
        # q_current = compute_ik(model, data, q_current, v_desired, frame_name, dt)

        q_current, error_norm = compute_close_loop_ik(model, data, q_current, oMgoal, idx_tool, dt)
        
        print(f"Error norm: {error_norm}")

        # Break if the error is small
        if error_norm < 1e-2:
            break

        update_visual_frame(viz, model, data, frame_name)
        display_robot_configuration(q_current, viz)

        print(f"Time passed: {time.time() - start_time}")
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

# Update q and frames
pin.forwardKinematics(model, data, q_current)
pin.updateFramePlacements(model, data)

# Frame name of the end effector
frame_name = "panda_hand"

# Set the goal position of the end effector
idx_tool = model.getFrameId(frame_name)
oMtool = data.oMf[idx_tool]
# Add an offset to the current position
oMgoal = pin.SE3(oMtool.rotation, oMtool.translation + np.array([0.1, 0.1, 0.1]))

# Rotate the goal by 180 degrees around the x-axis with a rotation matrix
oMgoal.rotation = oMgoal.rotation @ pin.utils.rotate('x', 3.14)

q_previous = simulate_movement(viz, model, data, q_current, v_desired, new_window)

# To run on terminal:
# q_previous = simulate_movement(viz, model, data, q_current, np.array([0.1, 0., 0., 0., 0., 0.]), new_window=False, q_previous=q_previous)