from ctu_crs import CRS97
import kinematics as k
import numpy as np
import cv2
import os


class Homography:
    def __init__(self):
        self.H = []
        self.position = []
        self.images = []
        self.robot = CRS97()
        self.robot.soft_home()
        self.robot.initialize()


    def capture_calibration_images(self):

        for x in np.arange(0.35, 0.55, 0.1):
            for y in np.arange(-0.20, 0.20, 0.1):
                    current_pose = k.fk(self.robot.get_q(), self.robot)
                    target_pose = current_pose.copy()
                    target_pose[0, 3] = x 
                    target_pose[1, 3] = y
                    target_pose[2, 3] = 0.05
                    target_pose[:3, :3] = k.get_rotation_from_z(np.array([0, 0, -1]))

                    ik_sols = k.ik(target_pose, self.robot)

                    if len(ik_sols) > 0:
                        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - self.robot.get_q()))
                        self.robot.move_to_q(closest_solution)
                        self.robot.wait_for_motion_stop()

                        self.images.append(self.robot.grab_image())
                        self.position.append([x, y, 0.05])

                    else:
                        print(f"No IK solutions found for target offset ({x}, {y}, {0.05})!")
                        continue
        
        # Create directory to save images if it doesn't exist
        save_dir = os.path.join(os.path.dirname(__file__), 'homographyImages')
        os.makedirs(save_dir, exist_ok=True)

        # Save captured images to the directory
        for idx, img in enumerate(self.images):
            cv2.imwrite(os.path.join(save_dir, f'image_{idx}.png'), img)
    
    def calculate_homography(self):
 
        images = np.asarray(self.images)
        assert images.shape[0] == len(self.position) #ensure we have the same amount of images and positions

        circle_centers = []
        for img in images:
            bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bw = cv2.GaussianBlur(bw, (5, 5), 0)
            circles = cv2.HoughCircles(bw, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=70, minRadius=5, maxRadius=150)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                print(f"Detected {circles.shape[1]} circles")
                print(f"Circle parameters: {circles[0][0]}")
                center = (circles[0][0][0], circles[0][0][1])  # x, y coordinates of the circle center
                circle_centers.append(center)  # Take the first detected circle


        print(f"Total detected circles: {len(circle_centers)}")
        assert len(circle_centers) == len(self.position), "Number of detected circles does not match number of hoop positions"  

        defined_centers = []
        for pos in self.position:
            defined_centers.append(pos[:2])  # Take only x and y coordinates
        

        print (f"Defined centers: {defined_centers}")
        print (f"Detected centers: {circle_centers}")

        # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.

        homography_matrix, _ = cv2.findHomography(np.array(circle_centers), np.array(defined_centers), cv2.RANSAC, 5.0)

        self.H = homography_matrix
        print(f"Calculated Homography Matrix:\n{self.H}")
        np.save('homography_matrix.npy', self.H)


if __name__ == "__main__":
    homography = Homography()
    homography.capture_calibration_images()
