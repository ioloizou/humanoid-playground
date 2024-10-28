import pinocchio as pin
import numpy as np
import time

model = pin.buildSampleModelManipulator()

data = model.createData()

q = pin.neutral(model)

print(q)

J = pin.computeJointJacobian(model, data, q, 6)

v = np.zeros(6)
v[0] = 1

qdot = np.linalg.pinv(J) @ v

q = pin.integrate(model, q, qdot*0.01)

print(q)