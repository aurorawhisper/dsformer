import os
import math
import torch
import random
import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as tfm
from scipy.spatial.distance import cdist
from tqdm import tqdm
import hdbscan
from sklearn.neighbors import NearestNeighbors
import datasets.dataset_utils as dataset_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

PANO_WIDTH = int(512 * 6.5)


def get_angle(focal_point, obs_point):
    obs_e, obs_n = float(obs_point[0]), float(obs_point[1])
    focal_e, focal_n = focal_point
    side1 = focal_e - obs_e
    side2 = focal_n - obs_n
    angle = - math.atan2(side1, side2) / math.pi * 90 * 2
    return angle


def get_eigen_things(utm_coords):
    mu = utm_coords.mean(0)
    norm_data = utm_coords - mu
    eigenvectors, eigenvalues, v = np.linalg.svd(norm_data.T, full_matrices=False)
    return eigenvectors, eigenvalues, mu


def rotate_2d_vector(vector, angle):
    assert vector.shape == (2,)
    theta = np.deg2rad(angle)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    rotated_point = np.dot(rot_mat, vector)
    return rotated_point


def get_focal_point(utm_coords, meters_from_center=10, angle=0):
    """Return the focal point from a set of utm coords"""
    B, D = utm_coords.shape
    assert D == 2
    eigenvectors, eigenvalues, mu = get_eigen_things(utm_coords)
    direction = rotate_2d_vector(eigenvectors[1], angle)
    focal_point = mu + direction * meters_from_center
    return focal_point


class HDBScanPlaces(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, focal_dist=15, current_group=0, min_images_per_class=10, angle=0):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        dataset_folder : str, the path of the folder with the train images.
        focal_dist : int, distance (in meters) between the center of the class and
            the focal point. The center of the class is computed as the
            mean of the positions of the images within the class.
        current_group : int, which one of the groups to consider.
        min_images_per_class : int, minimum number of image in a class.
        angle : int, the angle formed between the line of the first principal
            component, and the line that connects the center of gravity of the
            images to the focal point.
        """
        super().__init__()
        # for hdbscan clustering
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
        self.min_cluster_size = 15
        self.min_samples = 15
        self.block_width = 10
        self.block_num = 5
        self.neigh_radius = 7.5
        self.radius_dist = 40.0
        self.cropped_fov = 90.0
        self.min_images_per_class = min_images_per_class
        self.focal_dist = focal_dist
        self.current_group = current_group
        self.dataset_folder = dataset_folder

        filename = f"cache/sfxl_bw{self.block_width}_bn{self.block_num}_foc{focal_dist}_mipc{min_images_per_class}.torch"
        self.filename = filename
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            logging.info(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, filename)

        if current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        self.group_paths, self.group_utms, self.group_centroids = torch.load(filename)
        self.current_group_paths = self.group_paths[current_group]
        self.current_group_utms = self.group_utms[current_group]
        self.focal_point_per_class = []
        for utms in self.current_group_utms:
            focal_point = get_focal_point(utms, focal_dist, angle=angle)
            self.focal_point_per_class.append(focal_point)

    def __getitem__(self, class_num):
        # This function takes as input the class_num instead of the index of
        # the image. This way each class is equally represented during training.
        focal_point = self.focal_point_per_class[class_num]
        pano_path = self.dataset_folder + "/" + random.choice(self.current_group_paths[class_num])
        crop = self.get_crop(pano_path, focal_point)
        return crop, class_num, pano_path

    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.current_group_paths)

    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.current_group_paths[c]) for c in range(len(self))])

    def get_crop(self, pano_path, focal_point):
        cropped_width = int(self.cropped_fov / 360 * PANO_WIDTH)
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
        pil_crop = tfm.functional.to_tensor(pil_crop)

        return pil_crop

    def initialize(self, dataset_folder, filename):
        logging.debug(f"Searching training images in {dataset_folder}")

        images_paths = np.array(dataset_utils.read_images_paths(dataset_folder))
        logging.debug(f"Found {len(images_paths)} images")

        logging.debug("For each image, get its UTM east, UTM north from its path")
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north
        utmeast_utmnorth = [(m[1], m[2]) for m in images_metadatas]
        utmeast_utmnorth = np.array(utmeast_utmnorth).astype(float)

        east_values = utmeast_utmnorth[:, 0]
        north_values = utmeast_utmnorth[:, 1]
        # Data Structure: [group0[class0, class1, ...], group1[class0, class1, ...]]
        logging.debug(f"Clustering images in east <--> west ...")
        raw_east_utms, raw_east_paths, raw_east_centroids = self.Block_HDBSCAN_Clustering(east_values,
                                                                                          utmeast_utmnorth,
                                                                                          images_paths)
        logging.debug(f"Clustering images in north <--> south ...")
        raw_north_utms, raw_north_paths, raw_north_centroids = self.Block_HDBSCAN_Clustering(north_values,
                                                                                             utmeast_utmnorth,
                                                                                             images_paths)

        # Remove dense centroids based on centroids distances
        east_valid_indices = []

        for centroids in raw_east_centroids:
            valid_centroids_indices = []
            distances = cdist(centroids, centroids)
            for i, dis in enumerate(distances):
                if len(np.where((dis < self.radius_dist) & (dis > 0))[0]) == 0:
                    valid_centroids_indices.append(i)
            east_valid_indices.append(np.array(valid_centroids_indices))
        valid_east_utms = [raw_east_utms[i][east_valid_indices[i]] for i in range(self.block_num)]
        valid_east_paths = [raw_east_paths[i][east_valid_indices[i]] for i in range(self.block_num)]
        valid_east_centroids = [raw_east_centroids[i][east_valid_indices[i]] for i in range(self.block_num)]

        north_valid_indices = []
        for centroids in raw_north_centroids:
            valid_centroids_indices = []
            distances = cdist(centroids, centroids)
            for i, dis in enumerate(distances):
                if len(np.where((dis < self.radius_dist) & (dis > 0))[0]) == 0:
                    valid_centroids_indices.append(i)
            north_valid_indices.append(np.array(valid_centroids_indices))
        valid_north_utms = [raw_north_utms[i][north_valid_indices[i]] for i in range(self.block_num)]
        valid_north_paths = [raw_north_paths[i][north_valid_indices[i]] for i in range(self.block_num)]
        valid_north_centroids = [raw_north_centroids[i][north_valid_indices[i]] for i in range(self.block_num)]

        north_valid_indices2 = []
        for valid_north_centroid, valid_east_centroid in zip(valid_north_centroids, valid_east_centroids):
            valid_centroids_indices = []
            distances = cdist(valid_north_centroid, valid_east_centroid)
            for i, dis in enumerate(distances):
                if len(np.where((dis < self.radius_dist) & (dis > 0))[0]) == 0:
                    valid_centroids_indices.append(i)
            north_valid_indices2.append(np.array(valid_centroids_indices))
        group_paths = [np.concatenate([valid_east_paths[i], valid_north_paths[i][north_valid_indices2[i]]]) for i in
                       range(self.block_num)]
        group_utms = [np.concatenate([valid_east_utms[i], valid_north_utms[i][north_valid_indices2[i]]]) for i in
                      range(self.block_num)]
        group_centroids = [np.concatenate([valid_east_centroids[i], valid_north_centroids[i][north_valid_indices2[i]]])
                           for i in range(self.block_num)]

        torch.save((group_paths, group_utms, group_centroids), filename)

    def get_block_indices(self, min_v, max_v, m, raw_values):
        indices = []
        for utm_v in range(min_v, max_v, m * self.block_num):
            indices.append(np.where((raw_values >= utm_v) & (raw_values <= utm_v + m)))
        return indices

    def Block_HDBSCAN_Clustering(self, values, utmeast_utmnorth, images_paths):
        all_centroids = []
        all_paths = []
        all_utms = []
        all_indices = []
        min_v, max_v = round(min(values)), round(max(values))
        for i in range(self.block_num):
            all_indices.append(self.get_block_indices(i * self.block_width + min_v, max_v, self.block_width, values))
        neigh = NearestNeighbors(algorithm='brute')
        # HDBSCAN Clustering
        clustering = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                     min_samples=self.min_samples,
                                     metric='euclidean')
        for block_id, indices in enumerate(all_indices):
            centroids = []
            utms = []
            paths = []
            for indice in tqdm(indices, desc=f'Block_{block_id}: '):
                if len(indice[0]) == 0:
                    continue
                block_utms = utmeast_utmnorth[indice]
                block_paths = images_paths[indice]
                block_preds = clustering.fit_predict(block_utms)
                for i in range(max(block_preds)):
                    tmp_utms = block_utms[block_preds == i]
                    tmp_paths = block_paths[block_preds == i]
                    neigh.fit(tmp_utms)
                    _, I = neigh.radius_neighbors(tmp_utms, self.neigh_radius)
                    densities = np.array([len(p) for p in I])
                    if max(densities) >= self.min_images_per_class:
                        max_indices = np.where(densities == max(densities))[0]
                        highest_density_indice = max_indices[
                            np.argmin(np.mean(cdist(tmp_utms[max_indices], tmp_utms), axis=1))]
                        highest_density_point = tmp_utms[highest_density_indice]
                        tmp_utms = tmp_utms[I[highest_density_indice]]
                        tmp_paths = tmp_paths[I[highest_density_indice]]
                        centroids.append(highest_density_point)
                        utms.append(tmp_utms)
                        paths.append(tmp_paths)
            all_utms.append(np.array(utms, dtype=object))
            all_paths.append(np.array(paths, dtype=object))
            all_centroids.append(np.array(centroids).astype(float))
        return all_utms, all_paths, all_centroids
