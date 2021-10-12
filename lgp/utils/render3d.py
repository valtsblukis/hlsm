import time
import numpy as np
import open3d as o3d

from lgp.models.alfred.voxel_grid import VoxelGrid


def render_aligned_point_cloud(points, scene_image, animate=False):
    np_points = points.permute((0, 2, 3, 1)).view((-1, 3)).detach().numpy()
    np_colors = scene_image.permute((0, 2, 3, 1)).view((-1, 3)).detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    nparray = render_geometries(pcd, animate=animate)
    return nparray


def view_voxel_grid(voxel_grid: VoxelGrid):
    geometry, centroid = voxelgrid_to_geometry(voxel_grid)
    show_geometries(geometry)


def render_voxel_grid(voxel_grid: VoxelGrid, animate=False):
    geometry, centroid = voxelgrid_to_geometry(voxel_grid)
    frame_or_frames = render_geometries(geometry, animate, centroid=centroid)
    return frame_or_frames


def voxelgrid_to_geometry(voxel_grid: VoxelGrid):
    coord_grid = voxel_grid.get_centroid_coord_grid()
    occupied_mask = voxel_grid.occupancy > 0.5
    # Make sure occupied mask aligns with the data that it is used to index
    occupied_coords = coord_grid[occupied_mask.repeat((1, 3, 1, 1, 1))]
    occupied_data = voxel_grid.data[occupied_mask.repeat((1, voxel_grid.data.shape[1], 1, 1, 1))]
    # Build a PointCloud representation of the VoxelGrid
    pcd = o3d.geometry.PointCloud()
    np_points = occupied_coords.view([3, -1]).permute((1, 0)).detach().cpu().numpy()
    centroid = np_points.sum(0) / (np_points.shape[0] + 1e-10)
    # TODO: Render semantic channels in RGB
    np_colors = occupied_data.view([3, -1]).permute((1, 0)).detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    o3dvoxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_grid.voxel_size)
    return o3dvoxels, centroid


def render_geometries(geometry, animate=False, num_frames=18, centroid=None):
    vis = o3d.visualization.Visualizer()
    size = 900
    vis.create_window(width=size, height=size, visible=True)


def show_geometries(geometry):
    o3d.visualization.draw_geometries([geometry])


def get_topdown_extrinsics():
    extrinsics = [
        -0.0012036729144499355,
        0.99999927558549517,
        -0.0,
        0.0,
        0.99999927558549517,
        0.0012036729144499355,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -1.0,
        0.0,
        1.3062022702693259,
        0.45217550214016428,
        10.14637563106116,
        1.0
    ]
    intrinsics = {
        "height": 1080,
        "intrinsic_matrix":
            [
                935.30743608719376,
                0.0,
                0.0,
                0.0,
                935.30743608719376,
                0.0,
                959.5,
                539.5,
                1.0
            ],
        "width": 1920
    }
    T_extrinsics = np.asarray(extrinsics).reshape((4, 4)).T
    intrinsics["intrinsic_matrix"] = np.asarray(intrinsics["intrinsic_matrix"]).reshape((3, 3)).T
    return T_extrinsics, intrinsics


# Good for val unseen kitchen
def get_side_extrnsics():
    extrinsics = [
        0.44710135406621537,
        -0.64330548915175789,
        0.62149692422358327,
        0.0,
        -0.8944782407754126,
        -0.3238875800558948,
        0.30822964209313075,
        0.0,
        0.003009314121165479,
        -0.6937253657700243,
        -0.72023333782585097,
        0.0,
        1.3569806463340897,
        1.9256778598660822,
        5.1905799138297688,
        1.0
    ],
    intrinsics = {
        "height": 1080,
        "intrinsic_matrix":
            [
                935.30743608719376,
                0.0,
                0.0,
                0.0,
                935.30743608719376,
                0.0,
                959.5,
                539.5,
                1.0
            ],
        "width": 1920
    }
    T_extrinsics = np.asarray(extrinsics).reshape((4, 4)).T
    intrinsics["intrinsic_matrix"] = np.asarray(intrinsics["intrinsic_matrix"]).reshape((3, 3)).T
    return T_extrinsics, intrinsics


def get_side_extrnsics():
    extrinsics = [
        -0.7874044031349311,
        -0.4048262636087101,
        0.46487633002373213,
        0.0,
        -0.61643103850314951,
        0.51384392026509473,
        -0.59663824917325547,
        0.0,
        0.0026609572810071258,
        -0.7563597833697997,
        -0.65415043943051698,
        0.0,
        -0.44183541050442121,
        0.95600381984128813,
        8.4431486580534347,
        1.0
    ]
    intrinsics = {
        "height": 1080,
        "intrinsic_matrix":
            [
                1483.6378065054962,
                0.0,
                0.0,
                0.0,
                1483.6378065054962,
                0.0,
                959.5,
                539.5,
                1.0
            ],
        "width": 1920
    }
    T_extrinsics = np.asarray(extrinsics).reshape((4, 4)).T
    intrinsics["intrinsic_matrix"] = np.asarray(intrinsics["intrinsic_matrix"]).reshape((3, 3)).T

    return T_extrinsics, intrinsics


def render_geometries(geometry, animate=False, num_frames=18, centroid=None):
    vis = o3d.visualization.Visualizer()

    T_extrinsic, intrinsics = get_topdown_extrinsics()  # (centroid)

    vis.create_window(width=intrinsics["width"], height=intrinsics["height"], visible=True)
    time.sleep(0.1)

    vis.add_geometry(geometry)
    vis.update_geometry(geometry)
    vis.update_renderer()
    ctr = vis.get_view_control()
    viewportcam = ctr.convert_to_pinhole_camera_parameters()

    if T_extrinsic is not None:
        viewportcam.extrinsic = T_extrinsic
    if "intrinsic_matrix" in intrinsics:
        viewportcam.intrinsic.intrinsic_matrix = intrinsics["intrinsic_matrix"]
    ctr.convert_from_pinhole_camera_parameters(viewportcam)

    if animate:
        frames = []

        def frame_callback(vis):
            ctr = vis.get_view_control()
            ctr.rotate(0, -10.0)
            picture = vis.capture_screen_float_buffer(False)
            frames.append(np.asarray(picture))
            if len(frames) > num_frames:
                vis.register_animation_callback(None)
                vis.destroy_window()
            return False

        vis.register_animation_callback(frame_callback)
        vis.run()
        return frames
    else:
        # vis.run()
        vis.update_renderer()
        time.sleep(0.2)
        picture = vis.capture_screen_float_buffer(do_render=True)
        nparray = np.asarray(picture)
        vis.destroy_window()

    #show_image(nparray, "frame3d", scale=1, waitkey=True)

    return nparray