import cv2
import numpy as np
import torch
import os


from lgp.utils.utils import standardize_image, save_frames

from lgp.models.alfred.voxel_grid import VoxelGrid

from lgp.utils import viz

ANIMATE = False


def extract_frames(rollout, address : str, standardize=False):
    # Address is a ::-separated string indicating the variable in the rollout to extract
    frames = []
    address = address.split("::")

    # TODO: This is hacky
    if rollout[-1]["done"]:
        rollout = rollout[:-1]

    for sample in rollout:#[:-1]:
        data = sample
        found = True
        for tok in address:
            try:
                data = data[tok]
            except KeyError:
                print(f"Couldn't locate key: {tok} in address: {address}."
                      f"Available keys at this level: {data.keys() if hasattr(data, 'keys') else 'ERR: data not a dict'}")
                found = False

        # Low-level skills will have longer trajectory traces than high-level policies.
        # Specific skills will only have traces when they are active.
        # If the specific data is not found, repeat the previous frame if the previous frame exists.
        # otherwise continue to the next sample
        if found:
            frames.append(data)
        else:
            print(f"ERR: Copied of: {address}")
            if len(frames) > 0:
                frames.append(frames[-1])

    if len(frames) == 0:
        return []

    # Repeat the first frame ahead of the rollout if the number of frames is less than the length of rollout
    frames = [frames[0]] * (len(rollout) - len(frames)) + frames

    if standardize:
        frames = [standardize_image(f[0], normalize=False, uint8=False) for f in frames]

    return frames


def draw_text(text, width_px):
    bg = np.zeros((50, width_px, 3))
    cv2.putText(bg, text, (5, 15), cv2.QT_FONT_NORMAL, color=(255, 255, 255), fontScale=0.3)
    return bg


def draw_on_img(text, img_or_shape_or_heightpx, color=(0, 0, 0), ol_color=(255, 255, 255), bg_color=(255, 255, 255), max_text_h=None, line_num=0):
    whr = 0.5

    if isinstance(img_or_shape_or_heightpx, int):
        height_px = img_or_shape_or_heightpx
        width_px = int(float(len(text) * height_px) * whr)
        img = np.ones((height_px, width_px, 3), dtype=np.uint8) * np.asarray(bg_color)[None, None, :]
    elif isinstance(img_or_shape_or_heightpx, tuple):
        img = np.ones([img_or_shape_or_heightpx[0], img_or_shape_or_heightpx[1], 3], dtype=np.uint8) * np.asarray(bg_color)[None, None, :]
    else:
        img = img_or_shape_or_heightpx

    h_px, w_px, _ = img.shape
    h_from_w = int(float(w_px / whr) / float(len(text)))
    h_nom = min(h_px, h_from_w)
    if max_text_h is not None:
        h_nom = min(max_text_h, h_nom)

    font_scale = float(h_nom) / 20.0
    if line_num < 0:
        line_num = int(h_px / h_nom) + line_num

    origin = (int(h_nom * 0.25), int(h_nom * (line_num + 0.75)))

    img = img.astype(np.uint8)
    img = cv2.putText(img, text, origin, cv2.FONT_HERSHEY_PLAIN, lineType=cv2.LINE_AA, color=ol_color, fontScale=font_scale, thickness=4)
    img = cv2.putText(img, text, origin, cv2.FONT_HERSHEY_PLAIN, lineType=cv2.LINE_AA, color=color, fontScale=font_scale, thickness=1)
    return img


def dynamic_voxel_viz(rollout, live=False):
    state_reprs = extract_frames(rollout, "agent_trace::obs_func::state_repr")
    repr_diffs = extract_frames(rollout, "agent_trace::obs_func::observed_repr")
    subgoals = extract_frames(rollout, "agent_trace::hl_agent::action_proposal::subgoals")
    r_images = extract_frames(rollout, "agent_trace::skills::PickupObject::gofor::goto::reward_map")
    v_images = extract_frames(rollout, "agent_trace::skills::PickupObject::gofor::goto::v_image")

    frames = []

    for i in range(len(state_reprs)):
        state_repr = state_reprs[i]
        repr_diff = repr_diffs[i]
        subgoal = subgoals[i]
        r_image = r_images[i]
        v_image = v_images[i]
        has_goal = r_image.max().item() > 0.1
        gx, gy = r_image[0, 0].max(1).values.argmax(0).item(), r_image[0, 0].max(0).values.argmax(0).item()

        device = state_repr.data.data.device

        origin = state_repr.data.origin
        voxel_size = state_repr.data.voxel_size

        all_rgb_voxelgrid = state_repr.make_rgb_voxelgrid()
        current_rgb_voxelgrid = repr_diff.make_rgb_voxelgrid()

        current_base_color = torch.tensor([1.0, 1.0, 1.0], device=device)[None, :, None, None, None]
        past_base_color = torch.tensor([1.0, 1.0, 1.0], device=device)[None, :, None, None, None]
        all_rgb_voxelgrid.data = all_rgb_voxelgrid.data * 0.5 + past_base_color * 0.5
        current_rgb_voxelgrid.data = current_rgb_voxelgrid.data * 1.0#0.8 + current_base_color * 0.0
        #all_rgb_voxelgrid.data = all_rgb_voxelgrid.data * 0.7 + past_base_color * 0.3 * (1 - current_rgb_voxelgrid.occupancy) + current_base_color * 0.3 * current_rgb_voxelgrid.occupancy
        all_rgb_voxelgrid.data = all_rgb_voxelgrid.data * (1 - current_rgb_voxelgrid.occupancy) + current_rgb_voxelgrid.data * current_rgb_voxelgrid.occupancy

        if subgoal is not None:
            action_mask = subgoal.argument_mask
            yellow_action = action_mask.repeat((1, 3, 1, 1, 1)).clone()
            yellow_action[:, 2, :, :, :] = -1 * (2 * yellow_action[:, 0, :, :, :])

            #red_goal = torch.zeros_like(yellow_action)

            combined_rgb = torch.clamp(all_rgb_voxelgrid.data.data + yellow_action, 0, 1)
            print(f"Mask sum: {action_mask.sum().item()}, max: {action_mask.max().item()}")
        else:
            combined_rgb = all_rgb_voxelgrid.data.data

        combined_rgb_voxelgrid = VoxelGrid(combined_rgb, all_rgb_voxelgrid.occupancy, voxel_size, origin)

        if has_goal:
            # Mark the target pose on the voxel map
            combined_rgb_voxelgrid.data[:, 0, gx, gy, :5] = 1.0
            combined_rgb_voxelgrid.data[:, 1, gx, gy, :5] = 0.0
            combined_rgb_voxelgrid.data[:, 2, gx, gy, :5] = 0.0
            combined_rgb_voxelgrid.occupancy[:, :, gx, gy, :5] = 1.0
            # Mark the floor on the voxel map
            #combined_rgb_voxelgrid.data[0, 0, :, :, 0] = (combined_rgb_voxelgrid.data[0, 0, :, :, 0] + v_image[0, 0].to(device).clamp(0, 1) * 0.5).clamp(0, 1)
            #combined_rgb_voxelgrid.occupancy[0, :, :, :, 0] = 1.0

        # Mark agent position with a blue voxel
        ax, ay, az = state_repr.get_pos_xyz_vx()
        az = -az
        combined_rgb_voxelgrid.data[0, :, int(ax.item()), int(ay.item()), :int(az.item())] = torch.tensor([[0], [0], [0]], device=device).repeat((1, int(az.item())))
        combined_rgb_voxelgrid.occupancy[0, :, int(ax.item()), int(ay.item()), :int(az.item())] = 1.0

        # Mark origin as unobserved (currently a lot of black points with no occupancy are landing there)
        ox, oy, oz = state_repr.get_origin_xyz_vx()
        combined_rgb_voxelgrid.occupancy[:, :, ox, oy, oz] = 0.0

        import lgp.utils.render3d as r3d
        if live:
            r3d.view_voxel_grid(combined_rgb_voxelgrid)
        else:
            #r3d.view_voxel_grid(combined_rgb_voxelgrid)
            frame = r3d.render_voxel_grid(combined_rgb_voxelgrid)
            frames.append(frame)
    return frames


def export_media(media, vizdir, name, fps_settings={}, start_t=0):
    outdir = os.path.join(vizdir, name)
    for media_name, frames in media.items():
        mediadir = os.path.join(outdir, media_name)
        if media_name in fps_settings:
            fps = fps_settings[media_name]
        else:
            fps = 1
        save_frames(frames, mediadir, start_t)
        #save_mp4(frames, f"{mediadir}.mp4", fps=fps)
        #if gif:
        #save_gif(frames, f"{mediadir}.gif")


def visualize_corl21(rollout, vizdir, rollout_name, start_t=0):
    # First collect all the media
    media = {}
    fps_settings = {}

    #rollout = rollout[1:]

    # Observations
    observations = extract_frames(rollout, "observation")
    media["rgb_img"] = [standardize_image(obs.represent_as_image(rgb=True, semantic=False, depth=False)[0],
                            scale=2.0, normalize=False, uint8=True) for obs in observations]
    media["semantic_img"] = [standardize_image(obs.represent_as_image(rgb=False, semantic=True, depth=False)[0],
                            scale=2.0, normalize=True, uint8=True) for obs in observations]
    media["depth_img"] = [standardize_image(obs.represent_as_image(rgb=False, semantic=False, depth=True)[0] / 5,
                            scale=2.0, normalize=False, uint8=True) for obs in observations]

    # Actions
    actions = extract_frames(rollout, "action")
    media["action_args"] = [standardize_image(action.represent_as_image(), scale=2, normalize=True, uint8=True)
                        for action in actions]
    media["ov_actions"] = [0.5 * rgb + 0.5 * act * np.asarray([[[1, 0, 0]]]) for rgb, act in zip(media["rgb_img"], media["action_args"])]
    media["action_types"] = [draw_on_img(a.action_type, (100, 600)) for a in actions]

    # Tasks
    tasks = extract_frames(rollout, "task")
    media["tasks"] = []
    for task, obs_img in zip(tasks, media["rgb_img"]):
        task_image = draw_text(str(task), obs_img.shape[1])
        media["tasks"].append(task_image)

    # VIN value images:
    v_images = extract_frames(rollout, "agent_trace::skills::PickupObject::gofor::goto::v_image")
    v_images = [standardize_image(v, scale=5, uint8=True) for v in v_images]
    media["v_images"] = v_images

    # VIN occupancy image:
    occupancy_2d = extract_frames(rollout, "agent_trace::skills::PickupObject::gofor::goto::occupancy_2d")
    occupancy_2d = [standardize_image(v[0], scale=5, uint8=True) for v in occupancy_2d]
    media["occupancy_2d"] = occupancy_2d

    # VIN reward images:
    r_images = extract_frames(rollout, "agent_trace::skills::PickupObject::gofor::goto::reward_map")
    r_images = [standardize_image(r[0].repeat((3, 1, 1)), scale=5, uint8=True) for r in r_images]
    media["r_images"] = r_images

    # High-level action
    hl_actions_raw = extract_frames(rollout, "agent_trace::hl_agent::hl_action")
    hl_actions_text = [f"({a.type_str()}, {a.arg_str()})" if a is not None else "   " for a in hl_actions_raw]
    #W = media["rgb_img"][0].shape[1]
    hl_actions = [draw_on_img(a, (100, 600), color=(0, 0, 0), bg_color=(255, 255, 255)) for a in hl_actions_text]
    media["hl_actions"] = hl_actions

    # Action mask buildng
    fpv_argument_mask = extract_frames(rollout, "agent_trace::skills::PickupObject::interact::fpv_argument_mask", standardize=True)
    fpv_voxel_argument_mask = extract_frames(rollout, "agent_trace::skills::PickupObject::interact::fpv_voxel_argument_mask", standardize=True)
    fpv_semantic_argument_mask = extract_frames(rollout, "agent_trace::skills::PickupObject::interact::fpv_semantic_argument_mask", standardize=True)
    llc_flow_states = extract_frames(rollout, "agent_trace::skills::PickupObject::interact::llc_flow_state")
    action_intermediate_masks = [viz.vstack([a, b, c]) for a, b, c in zip(fpv_argument_mask, fpv_voxel_argument_mask, fpv_semantic_argument_mask)]
    media["action_intermediate_masks"] = viz.b_unify_size(action_intermediate_masks)

    # Voxelmap:
    voxel_frames = dynamic_voxel_viz(rollout, live=False)
    media["voxels"] = voxel_frames

    # The money visualization:
    extra_rgb_frames = extract_frames(rollout, "agent_trace::obs_func::extra_rgb_frames")

    subgoals = extract_frames(rollout, "agent_trace::hl_agent::action_proposal::subgoal")

    task_str = str(rollout[0]["task"])

    comb_frames = []
    for i in range(len(media["rgb_img"])):
        rgb_img = media["rgb_img"][i]
        rgb_images = extra_rgb_frames[i]
        is_motion = [len(rgb_images) > 1 for _ in rgb_images]
        rgb_images = [standardize_image(img, scale=2, uint8=True) for img in rgb_images] + [rgb_img]
        is_motion = is_motion + [False]

        for j, rgb_img in enumerate(rgb_images):
            if j >= len(rgb_images) - 2:
                k = i
            else:
                k = i - 1
            k = max(min(k, len(media["rgb_img"]) - 1), 0)
            seg_img = media["semantic_img"][k]
            depth_img = media["depth_img"][k]
            vox_img = media["voxels"][k]
            act_img = media["action_args"][k]
            subgoal = subgoals[k]
            actbg = media["rgb_img"][k]
            subgoal_text = f"({subgoal.type_str()}, {subgoal.arg_str()})" if subgoal is not None else " "
            action_text = actions[k].type_str()
            llc_flow_state = llc_flow_states[k]
            frame_is_motion = is_motion[j]

            v_img = media["v_images"][k]
            v_img = draw_on_img("VIN Value Function", v_img, max_text_h=30)

            a_mask_fpv_vox = standardize_image(fpv_voxel_argument_mask[k], scale=2, uint8=True)
            a_mask_fpv_seg = standardize_image(fpv_semantic_argument_mask[k], scale=2, uint8=True)
            a_mask_fpv_vox = actbg * 0.4 + a_mask_fpv_vox * np.asarray([1, 0, 0])[None, None, :] * 0.6
            a_mask_fpv_seg = actbg * 0.4 + a_mask_fpv_seg * np.asarray([1, 0, 0])[None, None, :] * 0.6

            a_mask_fpv_seg = draw_on_img("Subgoal Argument", a_mask_fpv_seg, max_text_h=50, line_num=0)
            a_mask_fpv_seg = draw_on_img("   Semantic Class Mask", a_mask_fpv_seg, max_text_h=50, line_num=1)
            a_mask_fpv_vox = draw_on_img("Subgoal Argument", a_mask_fpv_vox, max_text_h=50, line_num=0)
            a_mask_fpv_vox = draw_on_img("   Projected Voxel Mask", a_mask_fpv_vox, max_text_h=50, line_num=1)
            rcol = viz.vstack([v_img, a_mask_fpv_seg, a_mask_fpv_vox])

            # Renormalize depth (-ish)
            while depth_img.max() < 127:
                depth_img = depth_img * 2 + 1

            seg_img = draw_on_img("Segmentation", seg_img, max_text_h=30)
            depth_img = draw_on_img("Depth", depth_img, max_text_h=30)

            ds = viz.hstack([seg_img, depth_img])
            did_interact = act_img.sum() > 0.5
            if did_interact and not is_motion[j]:
                rgbact = 0.5 * rgb_img + 0.5 * act_img * np.asarray([[[1, 0, 0]]])
                rgbact = draw_on_img("RGB Input", rgbact, max_text_h=30)
                rgbact = draw_on_img("Action Argument", rgbact, max_text_h=30, color=(255, 0, 0), line_num=1)
            else:
                rgbact = rgb_img
                rgbact = draw_on_img("RGB Input              ", rgbact, max_text_h=30)
            lcol = viz.vstack([rgbact, ds])

            ccol = (vox_img[:, 300:-300, :] * 255).astype(np.uint8)
            ccol = draw_on_img("Semantic Voxel Map", ccol, color=(255, 255, 255), ol_color=(0, 0, 0), max_text_h=30, line_num=0)
            ccol = draw_on_img("Black Pillar: Agent", ccol, color=(0, 0, 0), ol_color=(0, 0, 0), max_text_h=30, line_num=1)
            ccol = draw_on_img("Red Pillar: Navigation Goal", ccol, color=(255, 0, 0), ol_color=(0, 0, 0), max_text_h=30, line_num=2)
            ccol = draw_on_img("Yellow: Subgoal Argument Mask", ccol, color=(255, 255, 0), ol_color=(0, 0, 0), max_text_h=30, line_num=3)

            # Mark action text
            ccol = draw_on_img(f"Task: {task_str}", ccol, color=(255, 255, 255), ol_color=(0, 0, 0), max_text_h=52, line_num=-3)
            ccol = draw_on_img(f"Subgoal: {subgoal_text}", ccol, color=(204, 255, 153), ol_color=(0, 0, 0), max_text_h=50, line_num=-2)
            ccol = draw_on_img(f"Action: {action_text}", ccol, color=(153, 204, 255), ol_color=(0, 0, 0), max_text_h=50, line_num=-1)
            ccol = draw_on_img(f"                             LLC State: {llc_flow_state}", ccol, color=(150, 150, 150), ol_color=(50, 50, 50), max_text_h=50, line_num=-1)

            comb = viz.hstack([lcol, ccol, rcol])

            # Keep to a Full-HD resolution - more than that is a bit much
            comb = viz.resize_to_width(comb, 1920)

            comb_frames.append(comb)
            # Repeat non-motion frames so that people have time to look at the masks
            if not is_motion[j]:
                for _ in range(5):
                    comb_frames.append(comb)
            # Additionally repeat interaction frames even more
            if did_interact and not is_motion[j]:
                for _ in range(10):
                    comb_frames.append(comb)
    media["rgb_voxel_comb"] = comb_frames
    fps_settings["rgb_voxel_comb"] = 10

    fps_settings["rgb_voxel_comb"] = 10
    # Temporary:
    media = {
        "rgb_voxel_comb": media["rgb_voxel_comb"],
    }
    export_media(media, vizdir, rollout_name, fps_settings, start_t)
    return len(comb_frames) + start_t


def visualize_rollout(rollout, vizdir, rollout_name, start_t=0):
    return visualize_corl21(rollout, vizdir, rollout_name, start_t=start_t)
