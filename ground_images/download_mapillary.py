
import argparse, os
import pandas as pd
import multiprocessing
import time
import mapillary.interface as mly
import requests



def downl_img(item):
    img_id =int(item)
    if os.path.exists(os.path.join(f"{dirPath}", f"{img_id}.jpg")):
        return
        
    img_url = mly.image_thumbnail(image_id=img_id, resolution=args.img_size)
    img_bin = requests.get(img_url)

    img_filepath = os.path.join(f"{dirPath}", f"{img_id}.jpg")
    with open(img_filepath, "wb") as file:
        file.write(img_bin.content)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
        help="Path to download folder.")
    parser.add_argument("--token", type=str, required=True,
        help="Pass your Mapillary account access token.")
    parser.add_argument("--img_size", type=int, default=1024,
        help="Image size to be downloaded, Options: [256|1024|2048]")
    parser.add_argument("--workers", type=int, default=0,
        help="Number of processes to use for parallel download of images.")
    args = parser.parse_args()
    print(args)

    if args.workers == 0:
        args.workers = multiprocessing.cpu_count()//2
        
    MLY_ACCESS_TOKEN = args.token
    mly.set_access_token(MLY_ACCESS_TOKEN)


    # Download Train-set images
    start = time.time()
    data_df = pd.read_csv("../meta_data/CVH3D_train_set.csv")
    data = list(data_df['Grd_ID'])
    dirPath = f"{args.path}/images"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    print(f"Total number of images: {len(data)}")
    print(data[:5])

    try:
        p = multiprocessing.Pool(args.workers)
        p.map(downl_img, data)
        p.close()
    except Exception as e:
        pass


    print()
    print(f"Total number of images downloaded: {len(os.listdir(dirPath))}")
    print(f"\nTotal time taken: {time.time()-start} secs")



    # Download Validation-set images
    start = time.time()
    data_df = pd.read_csv("../meta_data/CVH3D_validation_set.csv")
    data = list(data_df['Grd_ID'])
    dirPath = f"{args.path}/images"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    print(f"Total number of images: {len(data)}")
    print(data[:5])

    try:
        p = multiprocessing.Pool(args.workers)
        p.map(downl_img, data)
        p.close()
    except Exception as e:
        pass
    

    print()
    print(f"Total number of images downloaded: {len(os.listdir(dirPath))}")
    print(f"\nTotal time taken: {time.time()-start} secs")
