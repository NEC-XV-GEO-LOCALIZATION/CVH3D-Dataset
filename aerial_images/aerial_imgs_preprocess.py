import argparse, os
import numpy as np
import pandas as pd
import cv2
import json
import pyproj
import rasterio
from tqdm import tqdm


def convert_aerial_image(src_path, dest_path):
    for filename in tqdm(sorted(os.listdir(src_path))):
        if os.path.exists(f"{dest_path}/{filename.split('.')[0]}.png"):
            continue
        aer_img = cv2.imread(f"{src_path}/{filename}")
        cv2.imwrite(f"{dest_path}/{filename.split('.')[0]}.png", aer_img)


def code_gps_matrix(src_path, dest_path):
    for filename in sorted(os.listdir(src_path)):
        if os.path.exists(f"{dest_path}/{filename.split('.')[0]}.npy"):
            continue
        with rasterio.open(f"{src_path}/{filename}") as src:
            print(filename)
            print(src.width, src.height)
            print(src.crs)
            print(src.transform)
            print(np.asarray(src.transform, dtype=np.float32))
            print(src.count)
            print(src.indexes)
            print(src.bounds)
            w, h = src.width, src.height
            gps_matrix = np.zeros((w, h, 2))
            trans_matrix = np.asarray(src.transform, dtype=np.float32)
            a=float(trans_matrix[0])
            d=float(trans_matrix[1])
            b=float(trans_matrix[3])
            e=float(trans_matrix[4])
            c=float(trans_matrix[2])
            f=float(trans_matrix[5])
            transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326")  # Conversion from Local EPSG(3067) to Global EPSG(4326)
            for i in tqdm(range(w-1)):
                j = np.array([x for x in range(0, h-1)])
                gps_y = a*j+b*i+c
                gps_x = d*j+e*i+f
                old_pts = [(gps_y[k], gps_x[k]) for k in range(0, h-1)]
                new_pts = [pt for pt in transformer.itransform(old_pts)]
                j = np.array([x for x in range(0, h-1)])
                gps_matrix[i,j,0] = [new_pts[pt][0] for pt in range(len(new_pts))]
                gps_matrix[i,j,1] = [new_pts[pt][1] for pt in range(len(new_pts))]
            
            print(gps_matrix[0][:5])
            np.save(f"{dest_path}/{filename.split('.')[0]}.npy", gps_matrix)
        


def load_all_img_npy(aer_img_path, aer_mat_path):

    dir_all_imgs = sorted(os.listdir(f'{aer_img_path}'))

    for item in dir_all_imgs:
        str_name = item.split('.')[0]
        globals()[str_name + '_img'] = cv2.imread(f'{aer_img_path}/{item}')
        print(globals()[str_name + '_img'].shape)
        print(str_name + '_img')
    print('All Images Loaded')

    dir_all_numpys = sorted(os.listdir(f"{aer_img_path}"))

    for item in dir_all_numpys:
        str_name = item.split('.')[0]
        globals()[str_name + '_npy'] = np.load(f"{aer_img_path}/{item}")
        print(str_name + '_npy')
        print(globals()[str_name + '_npy'].shape)
    print('All numpy Loaded')


def create_pair(sat_img, matrix_gps, x_gps, y_gps, path_save, img_name, sat_size, precision):

    h, w, c = sat_img.shape
    location = np.argwhere(((np.around(matrix_gps, decimals = precision) == [round(float(x_gps), precision),round(float(y_gps), precision)]).all(axis = 2)))
    if len(location) == 0:
        return False
    if location.shape[1] == 2:
        cent_a = int(0.5*(min(location[:,0])+max(location[:,0])))
        cent_b = int(0.5*(min(location[:,1])+max(location[:,1])))
        cropped_img = sat_img[(cent_a-sat_size//2):(cent_a+sat_size//2),(cent_b-sat_size//2):(cent_b+sat_size//2),:]
        if np.prod(cropped_img.size) == 3*sat_size*sat_size:
            cv2.imwrite(f'{path_save}/{img_name}', cropped_img)
            return True
    return False

def create_expanded_sat_imgs(csv_path, aer_img_path, aer_mat_path, sat_size, sat_save_path):
    load_all_img_npy(aer_img_path, aer_mat_path)

    data_df = pd.read_csv(f"{csv_path}")
    print(data_df.head)
    res_df = pd.DataFrame(columns=data_df.columns)
    for index, row in tqdm(data_df.iterrows()):
        if os.path.exists(f"{sat_save_path}/{row['Sat_ID']}/{row['Grd_ID']}_sat.jpg"):
            continue
        sat_img = globals()[row['Sat_ID'] + '_img']
        npy_mat = globals()[row['Sat_ID'] + '_npy']

        dir_save = f"{sat_save_path}/{row['Sat_ID']}"
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        x_gps = float(row['Latitude'])
        y_gps = float(row['Longitude'])
        tmp_bool = create_pair(sat_img, npy_mat, x_gps, y_gps, dir_save, f"{row['Grd_ID']}_sat.png", sat_size)
        if tmp_bool == False:
            raise Exception(f"Crop size {sats_size} out of Aerial image bounds!")
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
        help="Path to aerial data main folder.")
    parser.add_argument("--sat_size", type=int, default=400,
        help="Aerial image crop size (sat_size X sat_size) in pixels. To follow the paper, use 400 pixels")
    parser.add_argument("--precision", type=int, default=4,
        help="Specifies GPS decimal precision for searching aerial pair within a satellite image.")
    args = parser.parse_args()
    print(args)


    # Convert the raw '.jp2' aerial image to PNG format
    convert_aerial_image(f"{args.path}/raw_images", f"{args.path}/aerial_pngs")

    # Create GPS Matrix for each aerial image
    code_gps_matrix(f"{args.path}/raw_images", f"{args.path}/aerial_matrices")

    # Create Train-set aerial image
    create_expanded_sat_imgs("../meta_data/CVH3D_train_set.csv", f"{args.path}/aerial_pngs", f"{args.path}/aerial_matrices",
        args.sat_size, f"{args.path}/sat_images")
    
    # Create Validation-set aerial image
    create_expanded_sat_imgs("../meta_data/CVH3D_validation_set.csv", f"{args.path}/aerial_pngs", f"{args.path}/aerial_matrices",
        args.sat_size, f"{args.path}/sat_images")

