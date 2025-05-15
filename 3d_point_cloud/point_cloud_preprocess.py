import argparse, os
import numpy as np
import pandas as pd
import json
import copy
import h5py
from tqdm import tqdm
import open3d as o3d
import trimesh
import pymeshlab as pyml
import rasterio.warp
import warnings
warnings.filterwarnings("ignore")



def merge_texture_mesh(src_path, dest_path):
    block_count = 0
    for block in sorted(os.listdir(f"{src_path}")):
        print(f"Block count: {block_count}")
        print(f"Block ID: {block}")
        block_count += 1
        subblock_count = 0
        for sub_block in sorted(os.listdir(f"{src_path}/{block}")):
            print(f"Sub Block count: {subblock_count}")
            print(f"Sub Block ID: {sub_block}")
            subblock_count += 1
            if os.path.exists(f"{src_path}/{block}/{sub_block}/{sub_block}_merged_mesh.obj"):
                continue
            if not os.path.exists(f"{src_path}/{block}/{sub_block}"):
                os.makedirs(f"{src_path}/{block}/{sub_block}")
            ms = pyml.MeshSet()
            ms.set_verbosity(False)
            if not os.path.exists(f"{dest_path}/{block}/{sub_block}"):
                os.makedirs(f"{dest_path}/{block}/{sub_block}")
            for item in tqdm(os.listdir(f"{src_path}/{block}/{sub_block}")):
                if ".obj" in item:
                    ms.load_new_mesh(f"{src_path}/{block}/{sub_block}/{item}")
                    ms.set_texture(textname=f"{src_path}/{block}/{sub_block}/{os.path.splitext(item)[0]}_0.jpg")
            # print(len(ms))
            ms.flatten_visible_layers()
            # print(len(ms))
            # print("Mesh flattened")
            ms.save_current_mesh(f"{dest_path}/{block}/{sub_block}/{sub_block}_merged_mesh.obj")
            # print("Mesh saved")


def convert_mesh_to_pcd(src_path, dest_path, sampling_count=1000000, seed=42):
    block_count = 0
    for block in sorted(os.listdir(f"{src_path}")):
        print(f"Block count: {block_count}")
        print(f"Block ID: {block}")
        block_count += 1
        subblock_count = 0
        for sub_block in sorted(os.listdir(f"{src_path}/{block}")):
            print(f"Sub Block count: {subblock_count}")
            print(f"Sub Block ID: {sub_block}")
            subblock_count += 1
            if os.path.exists(f"{dest_path}/{block}/{sub_block}/{sub_block}_pcd.ply"):
                continue
            if not os.path.exists(f"{dest_path}/{block}/{sub_block}"):
                os.makedirs(f"{dest_path}/{block}/{sub_block}")
            mesh_new_tri = trimesh.load_mesh(f"{src_path}/{block}/{sub_block}/{sub_block}_merged_mesh.obj")
            mesh_concat = mesh_new_tri.dump(True)
            pcd = trimesh.sample.sample_surface(mesh_concat, count=sampling_count, sample_color=True, seed=seed)
            new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[0]))
            colors = pcd[2]
            new_colors = []
            for i in range(len(colors)):
                new_colors.append([colors[i][0]/255.0, colors[i][1]/255.0, colors[i][2]/255.0])
            new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
            o3d.io.write_point_cloud(f"{dest_path}/{block}/{sub_block}/{sub_block}_pcd.ply", new_pcd)



def create_3d_samples(src_path, dest_path, data_csv_path, args, crop_size=100):
    
    data_df = pd.read_csv(data_csv_path)
    
    print(data_df.head)
    neighbors = json.load(open(f"{args.path}/meta_data/sub_block_neighbors.json"))
    grouped_data = data_df.groupby("250mx250m")
    utm_data = pd.read_csv(f"{args.path}/meta_data/sub_block_bbox_utm.csv")
    print(f"Total num of 250mx250m groups: {len(grouped_data)}")
    count = 1
    for name, group in grouped_data:
        print(f"Current group: {name}, count: {count}")
        count += 1
        sub_block = name
        mh_count, flag = 0, 0
        merge_neigh_pcd = o3d.geometry.PointCloud()
        for item in neighbors[sub_block]:
            neigh_block = utm_data[utm_data['250mx250m'] == item]
            if len(neigh_block) > 0:
                if os.path.exists(f"{src_path}/{list(neigh_block['2kmx2km'])[0]}/{item}/{item}_pcd.ply"):
                    pcd_new = o3d.io.read_point_cloud(f"{src_path}/{list(neigh_block['2kmx2km'])[0]}/{item}/{item}_pcd.ply")
                    merge_neigh_pcd += pcd_new
                    mh_count += 1
                else: print(f"{src_path}/{list(neigh_block['2kmx2km'])[0]}/{item}/{item}_pcd.ply  does not exist!!!")
            else:
                print(f"mh_count: {mh_count}, skipping")
                flag = 1
                break
        if flag == 1:
            continue
        print(f"mh_count: {mh_count}")
        bbox = merge_neigh_pcd.get_axis_aligned_bounding_box()
        bbox_min_bound = bbox.get_min_bound()
        merge_neigh_pcd = copy.deepcopy(merge_neigh_pcd).translate((-1*bbox_min_bound[0], -1*bbox_min_bound[1], 0))
        bbox = merge_neigh_pcd.get_axis_aligned_bounding_box()

        for index, row in tqdm(group.iterrows()):
            block = row['2kmx2km']
            if not os.path.exists(f"{dest_path}/{block}/{sub_block}/{row['Grd_ID']}_pcd_{crop_size}m.hdf5"):
                if not os.path.exists(f"{dest_path}/{block}/{sub_block}"):
                    os.makedirs(f"{dest_path}/{block}/{sub_block}")

                utm_img = rasterio.warp.transform(CRS.from_epsg(4326), CRS.from_epsg(3879), [float(row['Longitude'])], [float(row['Latitude'])])
                utm_block = utm_data[utm_data['250mx250m'] == str(sub_block)]
                img_utm_diff = [abs(float(utm_img[0][0])-float(list(utm_block['Upper_left_utm'])[0].split(',')[0][1:])), 
                        250-abs(float(utm_img[1][0])-float(list(utm_block['Upper_left_utm'])[0].split(',')[1][1:-1]))]

                img_coord = [img_utm_diff[0] + bbox.get_min_bound()[0] + 250, img_utm_diff[1] + bbox.get_min_bound()[1] + 250]

                bbox_min_bound = bbox.get_min_bound()
                bbox_max_bound = bbox.get_max_bound()
                bbox_points = [[img_coord[0]-crop_size//2, img_coord[1]-crop_size//2, bbox_min_bound[2]], [img_coord[0]+crop_size//2, img_coord[1]-crop_size//2, bbox_min_bound[2]],
                        [img_coord[0]-crop_size//2, img_coord[1]+crop_size//2, bbox_min_bound[2]], [img_coord[0]-crop_size//2, img_coord[1]-crop_size//2, bbox_max_bound[2]],
                        [img_coord[0]+crop_size//2, img_coord[1]+crop_size//2, bbox_max_bound[2]], [img_coord[0]-crop_size//2, img_coord[1]+crop_size//2, bbox_max_bound[2]],
                        [img_coord[0]+crop_size//2, img_coord[1]-crop_size//2, bbox_max_bound[2]], [img_coord[0]+crop_size//2, img_coord[1]+crop_size//2, bbox_min_bound[2]]]
                bbox_cropped = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_points))
                bbox_cropped.color = (1, 0, 0)

                # PCD Cropped
                pcd_cropped = merge_neigh_pcd.crop(bbox_cropped)

                new_pcd_crop = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pcd_cropped.points)))
                new_pcd_crop.colors = o3d.utility.Vector3dVector(np.asarray(pcd_cropped.colors))

                # Save as HDF5 file
                xyz = np.asarray(new_pcd_crop.points)
                rgb = np.asarray(new_pcd_crop.colors)
                with h5py.File(f"{dest_path}/{block}/{sub_block}/{row['Grd_ID']}_pcd_{crop_size}m.hdf5", 'w') as f:
                    dset_xyz = f.create_dataset('xyz', data=xyz)
                    dset_rgb = f.create_dataset('rgb', data=rgb)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
        help="Path to 3D data main folder.")
    parser.add_argument("--sampling_count", type=int, default=1000000,
        help="Number of points to sample from Texture Mesh to create Point Cloud samples")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for Trimesh Point Cloud sampling")
    parser.add_argument("--crop_size", type=int, default=100,
        help="3D Point Cloud crop size in meters (crop_size X crop_size).")
    args = parser.parse_args()
    print(args)

    # Merge the parts L21 texture meshes of all the blocks into single mesh OBJ file with textures
    merge_texture_mesh(f"{args.path}/raw_texture_mesh", f"{args.path}/merged_texture_mesh")

    # Convert the Merged Textured mesh to Point Cloud Samples
    convert_mesh_to_pcd(f"{args.path}/merged_texture_mesh", f"{args.path}/converted_pcd", sampling_count=args.sampling_count, seed=args.seed)

    # Create 3D samples for the Train-set by cropping in (crop_size X crop_size) m^2 sizes, samples saved in HDF5 format
    create_3d_samples(f"{args.path}/converted_pcd", f"{args.path}/3d_samples", f"../meta_data/CVH3D_train_set.csv", args, crop_size=args.crop_size)

    # Create 3D samples for the Validation-set by cropping in (crop_size X crop_size) m^2 sizes
    create_3d_samples(f"{args.path}/converted_pcd", f"{args.path}/3d_samples", f"../meta_data/CVH3D_validation_set.csv", args, crop_size=args.crop_size)

