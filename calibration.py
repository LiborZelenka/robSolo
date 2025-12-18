from ctu_crs import CRS97
import kinematics as k
import numpy as np

homography = []
images = []
positions = []

def capture_calibration_images():
    robot = CRS97()
    robot.initialize()
    q0 = robot.q_home
    robot.soft_home()

    for x in np.arange(0.35, 0.55, 0.05):
        for y in np.arange(-0.20, 0.20, 0.05):
                current_pose = k.fk(robot.get_q(), robot)
                target_pose = current_pose.copy()
                target_pose[0, 3] = x 
                target_pose[1, 3] = y
                target_pose[2, 3] = 0.05
                target_pose[:3, :3] = k.get_rotation_from_z(np.array([0, 0, -1]))

                ik_sols = k.ik(target_pose, robot)

                if len(ik_sols) > 0:
                    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
                    robot.move_to_q(closest_solution)
                    robot.wait_for_motion_stop()

                    images.append(robot.grab_image("calib_img_x{:.2f}_y{:.2f}.png".format(x, y)))
                    positions.append([x, y, 0.05])

                else:
                    print(f"No IK solutions found for target offset ({x}, {y}, {z})!")
                    continue



