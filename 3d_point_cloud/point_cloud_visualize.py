import argparse, os
import numpy as np
import h5py
from tqdm import tqdm
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")


def visualize_3d(path):

    if path.split('.')[-1] == "obj":
        mesh = o3d.io.read_triangle_mesh(f"{path}", enable_post_processing=True)
        mesh_bbox = mesh.get_axis_aligned_bounding_box()
        mesh_bbox.color = (0, 0, 1)
        print(f"Axis Aligned bbox extent: {mesh_bbox.get_extent()}")
        print(f"Axis Aligned bbox center: {mesh_bbox.get_center()}")
        print(f"Axis Aligned bbox corner points: {np.asarray(mesh_bbox.get_box_points())}")
        print(f"Axis Aligned bbox max bound: {mesh_bbox.get_max_bound()}")
        print(f"Axis Aligned bbox min bound: {mesh_bbox.get_min_bound()}")
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh_bbox, mesh])
    elif path.split('.')[-1] == "ply":
        pcd = o3d.io.read_point_cloud(f"{path}")
        pcd_bbox = pcd.get_axis_aligned_bounding_box()
        pcd_bbox.color = (0, 0, 1)
        print(f"Axis Aligned bbox extent: {pcd_bbox.get_extent()}")
        print(f"Axis Aligned bbox center: {pcd_bbox.get_center()}")
        print(f"Axis Aligned bbox corner points: {np.asarray(pcd_bbox.get_box_points())}")
        print(f"Axis Aligned bbox max bound: {pcd_bbox.get_max_bound()}")
        print(f"Axis Aligned bbox min bound: {pcd_bbox.get_min_bound()}")
        o3d.visualization.draw_geometries([pcd_bbox, pcd])
    elif path.split('.')[-1] == "hdf5":
        hf = h5py.File(f"{path}")
        xyz = np.asarray(hf.get('xyz'))
        rgb = np.asarray(hf.get('rgb'))

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd_bbox = pcd.get_axis_aligned_bounding_box()
        pcd_bbox.color = (0, 0, 1)
        print(f"Axis Aligned bbox extent: {pcd_bbox.get_extent()}")
        print(f"Axis Aligned bbox center: {pcd_bbox.get_center()}")
        print(f"Axis Aligned bbox corner points: {np.asarray(pcd_bbox.get_box_points())}")
        print(f"Axis Aligned bbox max bound: {pcd_bbox.get_max_bound()}")
        print(f"Axis Aligned bbox min bound: {pcd_bbox.get_min_bound()}")
        o3d.visualization.draw_geometries([pcd_bbox, pcd])
        



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
        help="Path to 3D sample to visualize.")
    args = parser.parse_args()
    print(args)

    # Pass path to the sample
    visualize_3d(args.path)