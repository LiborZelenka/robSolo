import numpy as np

# Define your tool offsets here as 4x4 matrices
# T_es: Transformation from End-effector to Sensor/Tool
T_es = np.eye(4)
T_es[:3, 3] = [-0.135, 0.0, 0.0]  # Example: 10 cm offset along Z-axis
T_es[:3, :3] = np.diag([1, 1, 1])  # Example rotation adjustment

# T_se: Inverse (Sensor to End-effector)
T_se = np.linalg.inv(T_es)

def fk(q: np.ndarray, robot):
    """Returns the position of the tool in absolute coordinates."""
    # robot.fk(q) returns the flange pose
    robot_fk = robot.fk(q) 
    # Apply tool offset: T_world_tool = T_world_flange * T_flange_tool
    return robot_fk @ T_es

def q_valid(q: np.ndarray, robot):
    """Checks joint limits and ensures tool is above a safety height."""
    # Access limits directly from the robot object if available
    limit_min = robot.q_min 
    limit_max = robot.q_max
    
    # Check joint boundaries
    within_limits = np.all((q >= limit_min) & (q <= limit_max))
    
    # Forward kinematics to check Z-height safety
    pose = robot.fk(q)
    z_height = pose[2, 3]
    
    return within_limits and z_height > 0.05

def ik(target_pose: np.ndarray, robot):
    """
    Finds IK solutions by rotating around the Z-axis to find valid approach angles.
    """
    ik_solutions = []
    num_steps = 20
    thetas = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)

    for theta in thetas:
        # Create rotation matrix around Z
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
        
        # Apply rotation to the target pose and convert back to flange coordinates
        # T_flange = T_target * R_z * T_tool_to_flange
        T_rotated = target_pose @ R_z
        T_flange = T_rotated @ T_se
        
        sols = robot.ik(T_flange)
        
        for s in sols:
            if q_valid(s, robot):
                ik_solutions.append(s)

    if len(ik_solutions) == 0:
        print("IK ERROR: No valid solution found for pose.")
        return []

    # Return unique solutions sorted by distance to a reference (optional)
    return np.unique(np.array(ik_solutions), axis=0)