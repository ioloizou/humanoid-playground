# This examples shows how to load and move a robot in meshcat.

import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath

from pinocchio.visualize import MeshcatVisualizer
def main():
    # Load the URDF model.
    # Conversion with str seems to be necessary when executing this file with ipython
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "g1_description")

    model_path = str(pinocchio_model_dir)
    print("Model path: ", model_path)
    mesh_dir = pinocchio_model_dir
    print("Mesh path: ", mesh_dir)
    urdf_filename = "g1_23dof.urdf"
    urdf_model_path = join(model_path, urdf_filename)
    print("URDF model path: ", urdf_model_path)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    )

    # Start a new MeshCat server and client.
    # Note: the server can also be started separately using the "meshcat-server" command in a terminal:
    # this enables the server to remain active after the current script ends.
    #
    # Option open=True pens the visualizer.
    # Note: the visualizer can also be opened seperately by visiting the provided URL.
    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
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

if __name__ == "__main__":
    main()
