import copy
import torch
import numpy as np
import math
from transforms3d import euler

import lgp.env.alfred.segmentation_definitions as segdef
from lgp.env.alfred.alfred_action import AlfredAction
from lgp.env.alfred.alfred_observation import AlfredObservation
from lgp.env.privileged_info import PrivilegedInfo

from lgp.models.alfred.hlsm.alfred_perception_model import AlfredSegmentationAndDepthModel

import lgp.paths

# TODO: Move to config
PERCEPTION_DEVICE = "cuda"


class PoseInfo():
    """
    Given all the different inputs from AI2Thor event, constructs a pose matrix and a position vector
    to add to the observation.
    """

    def __init__(self,
                 cam_horizon_deg,
                 cam_pos_enu,
                 rot_3d_enu_deg,
                 ):
        self.cam_horizon_deg = cam_horizon_deg
        self.cam_pos_enu = cam_pos_enu
        self.rot_3d_enu_deg = rot_3d_enu_deg

    def is_close(self, pi: "PoseInfo"):
        horizon_close = math.isclose(self.cam_horizon_deg, pi.cam_horizon_deg, abs_tol=1e-3, rel_tol=1e-3)
        cam_pos_close = [math.isclose(a, b) for a,b in zip(self.cam_pos_enu, pi.cam_pos_enu)]
        rot_close = [math.isclose(a, b) for a,b in zip(self.rot_3d_enu_deg, pi.rot_3d_enu_deg)]
        all_close = horizon_close and cam_pos_close[0] and cam_pos_close[1] and cam_pos_close[2] and rot_close[0] and rot_close[1] and rot_close[2]
        return all_close

    @classmethod
    def from_ai2thor_event(cls, event):
        # Unity uses a left-handed coordinate frame with X-Z axis on ground, Y axis pointing up.
        # We want to convert to a right-handed coordinate frame with X-Y axis on ground, and Z axis pointing up.
        # To do this, all you have to do is swap Y and Z axes.

        cam_horizon_deg = event.metadata['agent']['cameraHorizon']

        # Translation from world origin to camera/agent position
        cam_pos_dict_3d_unity = event.metadata['cameraPosition']
        # Remap Unity left-handed frame to ENU right-handed frame (X Y Z -> X Z -Y)
        cam_pos_enu = [cam_pos_dict_3d_unity['z'],
                       -cam_pos_dict_3d_unity['x'],
                       -cam_pos_dict_3d_unity['y']]

        # ... rotation to agent frame (x-forward, y-left, z-up)
        rot_dict_3d_unity = event.metadata['agent']['rotation']
        rot_3d_enu_deg = [rot_dict_3d_unity['x'], rot_dict_3d_unity['z'], rot_dict_3d_unity['y']]

        return PoseInfo(cam_horizon_deg=cam_horizon_deg,
                        cam_pos_enu=cam_pos_enu,
                        rot_3d_enu_deg=rot_3d_enu_deg)

    @classmethod
    def create_new_initial(cls):
        cam_horizon_deg = 30.0
        cam_pos_enu = [0.0, 0.0, -1.576]
        rot_3d_enu_deg = [0.0, 0.0, 0.0]
        return PoseInfo(cam_horizon_deg=cam_horizon_deg,
                        cam_pos_enu=cam_pos_enu,
                        rot_3d_enu_deg=rot_3d_enu_deg)

    def simulate_successful_action(self, action: AlfredAction):
        MOVE_STEP = 0.25
        PITCH_STEP = 15
        YAW_STEP = 90

        if action.action_type == "RotateLeft":
            self.rot_3d_enu_deg[2] = (self.rot_3d_enu_deg[2] - YAW_STEP) % 360
        elif action.action_type == "RotateRight":
            self.rot_3d_enu_deg[2] = (self.rot_3d_enu_deg[2] + YAW_STEP) % 360
        elif action.action_type == "MoveAhead":
            # TODO: Solve this with a geometry equation instead
            if math.isclose(self.rot_3d_enu_deg[2] % 360, 270):
                self.cam_pos_enu[1] += MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 90):
                self.cam_pos_enu[1] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 180):
                self.cam_pos_enu[0] -= MOVE_STEP
            elif math.isclose(self.rot_3d_enu_deg[2] % 360, 0) or math.isclose(self.rot_3d_enu_deg[2] % 360, 360):
                self.cam_pos_enu[0] += MOVE_STEP
            else:
                raise ValueError("Agent doesn't appear to be on a 90-degree grid! This is not supported")
        elif action.action_type == "LookDown":
            self.cam_horizon_deg = self.cam_horizon_deg + PITCH_STEP
        elif action.action_type == "LookUp":
            self.cam_horizon_deg = self.cam_horizon_deg - PITCH_STEP

    def get_agent_pos(self, device="cpu"):
        cam_pos = [
            -self.cam_pos_enu[0],
            -self.cam_pos_enu[1],
            self.cam_pos_enu[2]
        ]
        cam_pos = torch.tensor(cam_pos, device=device, dtype=torch.float32)
        return cam_pos

    def get_pose_mat(self):
        cam_pos_enu = torch.tensor(self.cam_pos_enu)
        # Translation from world origin to camera/agent position
        T_world_to_agent_pos = np.array([[1, 0, 0, cam_pos_enu[0]],
                                         [0, 1, 0, cam_pos_enu[1]],
                                         [0, 0, 1, cam_pos_enu[2]],
                                         [0, 0, 0, 1]])

        # ... rotation to agent frame (x-forward, y-left, z-up)
        rot_3d_enu_rad = [math.radians(r) for r in self.rot_3d_enu_deg]
        R_agent = euler.euler2mat(rot_3d_enu_rad[0], rot_3d_enu_rad[1], rot_3d_enu_rad[2])
        T_agent_pos_to_agent = np.asarray([[R_agent[0, 0], R_agent[0, 1], R_agent[0, 2], 0],
                                           [R_agent[1, 0], R_agent[1, 1], R_agent[1, 2], 0],
                                           [R_agent[2, 0], R_agent[2, 1], R_agent[2, 2], 0],
                                           [0, 0, 0, 1]])

        # .. transform to camera-forward frame (x-right, y-down, z-forward) that ignores camera pitch
        R_agent_to_camflat = euler.euler2mat(0, math.radians(90), math.radians(-90))
        T_agent_to_camflat = np.asarray(
            [[R_agent_to_camflat[0, 0], R_agent_to_camflat[0, 1], R_agent_to_camflat[0, 2], 0],
             [R_agent_to_camflat[1, 0], R_agent_to_camflat[1, 1], R_agent_to_camflat[1, 2], 0],
             [R_agent_to_camflat[2, 0], R_agent_to_camflat[2, 1], R_agent_to_camflat[2, 2], 0],
             [0, 0, 0, 1]])

        # .. transform to camera frame (x-right, y-down, z-forward) that also incorporates camera pitch
        R_camflat_to_cam = euler.euler2mat(math.radians(self.cam_horizon_deg), 0, 0)
        T_camflat_to_cam = np.asarray([[R_camflat_to_cam[0, 0], R_camflat_to_cam[0, 1], R_camflat_to_cam[0, 2], 0],
                                       [R_camflat_to_cam[1, 0], R_camflat_to_cam[1, 1], R_camflat_to_cam[1, 2], 0],
                                       [R_camflat_to_cam[2, 0], R_camflat_to_cam[2, 1], R_camflat_to_cam[2, 2], 0],
                                       [0, 0, 0, 1]])

        # compose into a transform from world to camera
        T_world_to_cam = T_camflat_to_cam @ T_agent_to_camflat @ T_agent_pos_to_agent @ T_world_to_agent_pos
        T_world_to_cam = torch.from_numpy(T_world_to_cam).unsqueeze(0)
        return T_world_to_cam


class InventoryInfo():
    def __init__(self, inventory_objects):
        self.inventory_object_ids = inventory_objects

    @classmethod
    def create_empty_initial(cls):
        return InventoryInfo([])

    @classmethod
    def from_ai2thor_event(cls, event):
        # For each object in the inventory, mark the corresponding dimension in the inventory vector with a 1.0
        inventory_objects = []
        for object in event.metadata['inventoryObjects']:
            object_str = object['objectType'].split("_")[0]
            object_id = segdef.object_string_to_intid(object_str)
            inventory_objects.append(object_id)
        return InventoryInfo(inventory_objects)

    def simulate_successful_action(self, action, latest_observaton):
        if action.action_type == "PickupObject":
            selecton = (action.argument_mask[None, None] * latest_observaton.semantic_image.to(action.argument_mask.device)).sum(dim=(2, 3))
            arg_id = selecton.argmax(dim=1)[0].item()
            # TODO: Is it better to get inventory object class from seg_image and depth_estimate or from
            # whatever action the agent was trying to execute before.
            if len(self.inventory_object_ids) == 0:
                self.inventory_object_ids.append(arg_id)
        elif action.action_type == "PutObject":
            self.inventory_object_ids = []

    def get_inventory_vector(self, device="cpu"):
        num_objects = segdef.get_num_objects()
        inv_vector = torch.zeros([num_objects], device=device, dtype=torch.uint8)
        for object_id in self.inventory_object_ids:
            inv_vector[object_id] = 1
        return inv_vector

    def summarize(self):
        summary = f"Inventory with: {[segdef.object_intid_to_string(i) for i in self.inventory_object_ids]}"
        return summary


class StateTracker():
    """
    Converts raw RGB images and executed actions to AlfredObservation instances that eval:
    - Segmentation
    - Depth
    - Pose
    - Inventory information
    """

    def __init__(self,
                 reference_seg=False,
                 reference_depth=False,
                 reference_pose=False,
                 reference_inventory=False,
                 hparams=None,
                 fov=60):
        self.first_event = None
        self.latest_event = None
        self.latest_observation = None
        self.latest_action = None
        self.latest_extra_events = []

        self.pose_info = None
        self.inventory_info = None

        self.reference_seg = reference_seg
        self.reference_depth = reference_depth
        self.reference_pose = reference_pose
        self.reference_inventory = reference_inventory

        self.fov = fov

        self.seg_model = None
        self.depth_model = None
        if self.reference_seg:
            self.seg_model = None
        else:
            self.seg_model = AlfredSegmentationAndDepthModel(hparams).to(PERCEPTION_DEVICE)
            self.seg_model.load_state_dict(torch.load(lgp.paths.get_segmentation_model_path()))
            self.seg_model.eval()
        if self.reference_depth:
            self.depth_model = None
        else:
            self.depth_model = AlfredSegmentationAndDepthModel(hparams).to(PERCEPTION_DEVICE)
            self.depth_model.load_state_dict(torch.load(lgp.paths.get_depth_model_path()))
            self.depth_model.eval()

    def reset(self, event):
        # First reset everything
        self.latest_event = event
        self.first_event = event
        self.latest_action = None
        self.latest_observation = None

        # Initialize pose and inventory
        if self.reference_pose:
            self.pose_info = PoseInfo.from_ai2thor_event(event)
        else:
            self.pose_info = PoseInfo.create_new_initial()

        if self.reference_inventory:
            self.inventory_info = InventoryInfo.from_ai2thor_event(event)
        else:
            self.inventory_info = InventoryInfo.create_empty_initial()

        # Make the first observation
        self.latest_observation = self._make_observation()

    def log_action(self, action: AlfredAction):
        self.latest_action = action

    def log_event(self, event):
        self.latest_event = event
        #self.latest_observation = self._make_observation()

    def log_extra_events(self, events):
        self.latest_extra_events = events
        self.latest_observation = self._make_observation()

    def get_observation(self) -> AlfredObservation:
        return self.latest_observation

    def _make_observation(self) -> AlfredObservation:
        event = self.latest_event

        # RGB Image:
        # Add batch dimension to each image
        if event.frame is not None:
            rgb_image = torch.from_numpy(event.frame.copy()).permute((2, 0, 1)).unsqueeze(0).half() / 255
        else:
            rgb_image = torch.zeros((1, 3, 300, 300))

        # Depth
        if self.reference_depth:
            depth_image = torch.from_numpy(event.depth_frame.copy()).unsqueeze(0).unsqueeze(0) / 1000
        else:
            _, pred_depth = self.depth_model.predict(rgb_image.float().to(PERCEPTION_DEVICE))
            depth_image = pred_depth.to("cpu") # TODO: Maybe skip this? We later move it to GPU anyway

        # Segmentation
        if self.reference_seg:
            semantic_image = StateTracker._extract_reference_semantic_image(event)
            semantic_image = semantic_image.unsqueeze(0)
        else:
            pred_seg, _ = self.seg_model.predict(rgb_image.float().to(PERCEPTION_DEVICE))
            semantic_image = pred_seg

        # Simple error detection from RGB image changes
        action_failed = False
        if self.latest_observation is not None:
            assert self.latest_action is not None, "Didn't log an action, but got two observations in a row?"
            rgb_diff = (rgb_image - self.latest_observation.rgb_image).float().abs().mean()
            if rgb_diff < 1e-4:
                print(f"Action: {self.latest_action}, RGB Diff: {rgb_diff}. Counting as failed.")
                action_failed = True
            else:
                pass
                #print(f"Action: {self.latest_action} Success with RGB Diff: {rgb_diff}")

        # Use dead-reckoning to estimate the pose, and track state of the inventory
        if not action_failed and self.latest_action is not None:
            self.pose_info.simulate_successful_action(self.latest_action)
            oinv = copy.deepcopy(self.inventory_info)
            self.inventory_info.simulate_successful_action(self.latest_action, self.latest_observation)
            if len(oinv.inventory_object_ids) != len(self.inventory_info.inventory_object_ids):
                print(self.inventory_info.summarize())

        # Pose
        if self.reference_pose:
            self.pose_info = PoseInfo.from_ai2thor_event(event)

        T_world_to_cam = self.pose_info.get_pose_mat()
        cam_horizon_deg = [self.pose_info.cam_horizon_deg]
        agent_pos = self.pose_info.get_agent_pos()

        # Inventory
        if self.reference_inventory:
            self.inventory_info = InventoryInfo.from_ai2thor_event(event)
        inventory_vector = self.inventory_info.get_inventory_vector()
        inventory_vector = inventory_vector.unsqueeze(0)

        privileged_info = PrivilegedInfo(event)
        observation = AlfredObservation(rgb_image,
                                         depth_image,
                                         semantic_image,
                                         inventory_vector,
                                         T_world_to_cam,
                                         self.fov,
                                         cam_horizon_deg,
                                         privileged_info)

        # TODO: Use pose instead:
        observation.set_agent_pos(agent_pos)
        if action_failed:
            observation.set_error_causing_action(self.latest_action)

        # Add extra RGB frames from smooth navigation
        if self.latest_extra_events:
            extra_frames = [torch.from_numpy(e.frame.copy()).permute((2, 0, 1)).unsqueeze(0).half() / 255 for e in self.latest_extra_events]
            observation.extra_rgb_frames = extra_frames

        return observation

    @classmethod
    def _extract_reference_semantic_image(cls, event, device="cpu"):
        """
        The segmentation images that come from AI2Thor have unstable color<->object mappings.
        Instead, we can build up a one-hot object image from the dictionary of class masks
        """
        num_objects = segdef.get_num_objects()
        h, w = event.frame.shape[0:2]
        seg_image = torch.zeros([num_objects, h, w], dtype=torch.int16, device=device)

        inventory_obj_strs = set()
        for object in event.metadata['inventoryObjects']:
            inventory_obj_string = object['objectType'].split("_")[0]
            inventory_obj_strs.add(inventory_obj_string)

        for obj_str, class_mask in event.class_masks.items():
            obj_int = segdef.object_string_to_intid(obj_str)
            class_mask_t = torch.from_numpy(class_mask.astype(np.int16)).to(device)
            seg_image[obj_int] = torch.max(seg_image[obj_int], class_mask_t)
        return seg_image.type(torch.ByteTensor)

    @classmethod
    def _extract_reference_inventory_vector(cls, event, device="cpu"):
        num_objects = segdef.get_num_objects()
        inv_vector = torch.zeros([num_objects], device=device, dtype=torch.uint8)
        # For each object in the inventory, mark the corresponding dimension in the inventory vector with a 1.0
        for object in event.metadata['inventoryObjects']:
            object_str = object['objectType'].split("_")[0]
            object_id = segdef.object_string_to_intid(object_str)
            inv_vector[object_id] = 1
        return inv_vector