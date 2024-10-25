# This examples shows how to load and move a robot in meshcat.

import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath

from pinocchio.visualize import MeshcatVisualizer
import time

# Load the URDF model.
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "g1_description")

model_path = str(pinocchio_model_dir)
mesh_dir = pinocchio_model_dir
urdf_filename = "g1_23dof.urdf"
urdf_model_path = join(model_path, urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

data = model.createData()

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

# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)
viz.displayVisuals(True)


# Display the robot with the feet on the meshcat ground.
q1 = q0.copy()
q1[2] = 0.8
viz.display(q1)


# Print the joint names and their index.
print("Joint names and their index:")
for i, name in enumerate(model.names):
    print(f"{i}: {name}")

# Calculating the jacobian from joint 24 to joint 20
q2 = q1.copy()
J = pin.computeFrameJacobian(model, data, q2, 24, pin.LOCAL_WORLD_ALIGNED)

# Taking the last 5 columns of the jacobian right wrist roll to right shoulder pitch joint
# J = J[:, -5:]

# Calculating joints positions based on velocity of the end effector through the jacobian
v = np.ones(6) * 1
J_pseudo_inv = np.linalg.pinv(J)

# Initialize the configuration
q_current = q2.copy()

# Wait 4 second before starting the simulation
time.sleep(4)

# Run the simulation for 1 second
start_time = time.time()
while time.time() - start_time < 1.0:
    q_dot = J_pseudo_inv @ v
    q_current = pin.integrate(model, q_current, q_dot * 0.01)
    viz.display(q_current)
    time.sleep(0.01)  # Sleep to simulate real-time update
