import numpy as np
import ctu_crs

t_es = np.eye(4)
t_es[:3, :3] = np.diag([-1, 1, -1])
t_es[:3, 3] = [-0.135, 0, 0]

def fk (q: np.ndarray, robot) -> np.ndarray:
    T = robot.fk(q)
    fk_ee = T @ t_es

    return fk_ee

def ik (pose: np.ndarray, robot) -> np.ndarray:
    pose_robot = np.linalg.inv(t_es) @ pose
    q_ik = robot.ik(pose_robot)

    return q_ik

