import numpy as np
import pybullet as p
import pybullet_data as pd
import cv2


class SimulationEnvironment:
    def __init__(self, is_render=True):
        self.is_render = is_render
        self.object_ids = []
        self.urdf_path = '../../ASSET/urdf/'

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 240)

        # Camera parameters
        self.camera_params = {
            "width": 640,
            "height": 480,
            "fov": 42,
            "near": 0.1,
            "far": 100.0,
            "view_matrix": p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.15, 0, 0],
                distance=0.4,
                yaw=90,
                pitch=-90,
                roll=0,
                upAxisIndex=2,
            ),
            "projection_matrix": p.computeProjectionMatrixFOV(
                fov=42, aspect=1.33, nearVal=0.1, farVal=100.0
            ),
        }
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

        self.low_scale = np.array([0.03, -0.14, 0.0, - np.pi / 2, 0])
        self.high_scale = np.array([0.27, 0.14, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]
        self.table_boundary = 0.03
        self.total_offset = [0.016, -0.20 + 0.016, 0]

        baseid = p.loadURDF(self.urdf_path + "plane.urdf", useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        textureId = p.loadTexture(self.urdf_path + "floor_white.png")
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5,
                         angularDamping=0.5)
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        lineColorRGB = [0,0,0])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        lineColorRGB = [0,0,0])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        lineColorRGB = [0,0,0])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        lineColorRGB = [0,0,0])


    def load_objects(self, object_list):
        """
        Load objects into the simulation.
        :param object_list: List of dictionaries with object configurations. Each dictionary should have:
            - 'urdf_path': Path to the URDF file of the object.
            - 'position': A list or tuple [x, y, z] specifying the object's initial position.
            - 'orientation': A list or tuple [roll, pitch, yaw] specifying the object's orientation.
        """
        for obj in object_list:
            position = obj["position"]
            orientation = p.getQuaternionFromEuler(obj["orientation"])
            obj_id = p.loadURDF(obj["urdf_path"], basePosition=position, baseOrientation=orientation)
            self.object_ids.append(obj_id)
        print(f"Loaded {len(object_list)} objects into the simulation.")

    def capture_image(self):
        """Capture an image from the simulation's camera."""
        _, _, img, _, _ = p.getCameraImage(
            width=self.camera_params["width"],
            height=self.camera_params["height"],
            viewMatrix=self.camera_params["view_matrix"],
            projectionMatrix=self.camera_params["projection_matrix"],
        )
        # Convert RGB to BGR for OpenCV display
        img = img[:, :, :3]
        img = img[..., ::-1]
        return img

    def display_image(self, img):
        """Display an image using OpenCV."""
        cv2.imshow("Simulation Camera View", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_simulation(self, steps=240):
        """Run the simulation for a specified number of steps."""
        for _ in range(steps):
            p.stepSimulation()

    def cleanup(self):
        """Clean up the simulation."""
        p.disconnect()
        print("Simulation ended.")


# Example usage
if __name__ == "__main__":
    # Create the simulation environment
    env = SimulationEnvironment(is_render=True)

    # Define objects to load
    objects = [
        {
            "urdf_path": "cube_small.urdf",
            "position": [0.1, 0.1, 0.05],
            "orientation": [0, 0, 0],  # Roll, pitch, yaw
        },
        {
            "urdf_path": "sphere_small.urdf",
            "position": [-0.1, -0.1, 0.05],
            "orientation": [0, 0, 0],
        },
        {
            "urdf_path": "duck_vhacd.urdf",
            "position": [0.0, 0.0, 0.1],
            "orientation": [0, 0, 0],
        },
    ]

    # Load objects into the simulation
    env.load_objects(objects)

    # Run simulation for a few steps to stabilize objects
    env.run_simulation(steps=240)

    # Capture and display an image from the simulation
    img = env.capture_image()
    env.display_image(img)

    # Clean up the simulation
    env.cleanup()
