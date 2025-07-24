import argparse
import os
from PIL import Image
from tqdm import tqdm
from datasets.hdbscanplaces_datasets import HDBScanPlaces, get_focal_point, get_angle
PANO_WIDTH = int(512 * 6.5)


def get_crop(pano_path, focal_point):
    cropped_width = int(90.0 / 360 * PANO_WIDTH)
    obs_point = pano_path.split("@")[1:3]
    angle = - get_angle(focal_point, obs_point) % 360
    crop_offset = int((angle / 360 * PANO_WIDTH) % PANO_WIDTH)
    yaw = int(pano_path.split("@")[9])
    north_yaw_in_degrees = (180 - yaw) % 360
    yaw_offset = int((north_yaw_in_degrees / 360) * PANO_WIDTH)
    offset = (yaw_offset + crop_offset - cropped_width * 0.5) % PANO_WIDTH
    pano_pil = Image.open(pano_path)
    if offset + cropped_width <= PANO_WIDTH:
        pil_crop = pano_pil.crop((offset, 0, offset + cropped_width, 512))
    else:
        crop1 = pano_pil.crop((offset, 0, PANO_WIDTH, 512))
        crop2 = pano_pil.crop((0, 0, cropped_width - (PANO_WIDTH - offset), 512))
        pil_crop = Image.new('RGB', (cropped_width, 512))
        pil_crop.paste(crop1, (0, 0))
        pil_crop.paste(crop2, (crop1.size[0], 0))
    pil_crop = pil_crop.resize((512, 512))
    return pil_crop

def extract_images(args):
    hdb_ds = HDBScanPlaces(dataset_folder=args.dataset_folder)
    for angle in [0, 90]:
        for current_utms, current_group_paths in zip(hdb_ds.group_utms, hdb_ds.group_paths):
            for utms, group_path in tqdm(zip(current_utms, current_group_paths), total=len(current_utms)):
                focal_point = get_focal_point(utms, args.focal_dist, angle=angle)
                for path in group_path:
                    pano_path = args.dataset_folder + "/" + path
                    pil_crop = get_crop(pano_path, focal_point)
                    pil_crop.save(f'{args.extracted_folder}/{angle}/{path}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_folder", type=str, default="/home/jc/jc/RAL2025/datasets/sfxl")
    parser.add_argument("--extracted_folder", type=str, default="/mnt/data/sfxl")
    parser.add_argument("--focal_dist", type=int, default=15)

    args = parser.parse_args()

    subfolders = os.listdir(args.dataset_folder)
    for angle in [0, 90]:
        for folder in subfolders:
            os.makedirs(f'{args.extracted_folder}/{angle}/{folder}', exist_ok=True)
    extract_images(args)
