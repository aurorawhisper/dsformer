from torch.utils.data import Dataset
import os
from os.path import join
from glob import glob
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import numpy as np
from datasets.mapillary_sls_main.mapillary_sls.datasets.msls import MSLS
from sklearn.neighbors import NearestNeighbors
import scipy.io as scio
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


class TestDataset(Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache1).
    """
    def __init__(self, args, dataset_name='msls', split=None):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(args.dataset_folder, dataset_name)
        self.images_paths = []
        self.base_transform = transforms.Compose([
                              transforms.Resize(args.resize),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
        if dataset_name == 'msls':
            if not os.path.exists(self.dataset_folder):
                raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
            if not os.path.exists(os.path.join(self.dataset_folder, 'npys')):
                print('npys not found, create:', os.path.join(self.dataset_folder, 'npys'))
                os.mkdir(os.path.join(self.dataset_folder, 'npys'))
                _ = MSLS(root_dir=self.dataset_folder, save=True, mode='val', posDistThr=args.soft_positives_dist_threshold)
                _ = MSLS(root_dir=self.dataset_folder, save=True, mode='test')
                #_ = MSLS(root_dir=self.dataset_folder, save=True, mode='train')
            self.qIdx = np.load(os.path.join(self.dataset_folder, 'npys', 'msls_' + split + '_qIdx.npy'))
            self.database_paths = np.load(os.path.join(self.dataset_folder, 'npys', 'msls_' + split + '_dbImages.npy'))
            self.queries_paths = np.load(os.path.join(self.dataset_folder, 'npys', 'msls_' + split + '_qImages.npy'))
            self.pIdx = np.load(os.path.join(self.dataset_folder, 'npys', 'msls_' + split + '_pIdx.npy'), allow_pickle=True)
            self.nonNegIdx = np.load(os.path.join(self.dataset_folder, 'npys', 'msls_' + split + 'nonNegIdx.npy'), allow_pickle=True)
            self.queries_paths = self.queries_paths[self.qIdx]
            self.soft_positives_per_query = self.pIdx
            # Remove queries without soft positives
            queries_without_any_soft_positives = \
                np.where(np.array([len(p) for p in self.soft_positives_per_query], dtype=object) == 0)[0]
            if len(queries_without_any_soft_positives) != 0:
                logging.info(f"There are {len(queries_without_any_soft_positives)} queries without any positives " +
                             "within the training set. They won't be considered as they're useless for testing.")
                self.soft_positives_per_query = np.delete(self.soft_positives_per_query,
                                                          queries_without_any_soft_positives)
                self.queries_paths = np.delete(self.queries_paths, queries_without_any_soft_positives)
        elif dataset_name.startswith('pitts'):
            self.dataset_folder = join(args.dataset_folder, 'Pittsburgh250k')
            if not os.path.exists(self.dataset_folder):
                raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
            matdata = scio.loadmat(f'{self.dataset_folder}/netvlad_v100_datasets/datasets/{dataset_name}_{split}.mat')['dbStruct'].item()
            self.database_paths = [f[0].item() for f in matdata[1]]
            self.database_paths = [join(self.dataset_folder, path) for path in self.database_paths]
            self.database_utms = matdata[2].T
            self.queries_paths = [f[0].item() for f in matdata[3]]
            self.queries_paths = [join(self.dataset_folder, 'queries_real', path) for path in self.queries_paths]
            self.queries_utms = matdata[4].T
            # Find soft_positives_per_query, which are within soft_positives_dist_threshold (25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                                 radius=args.soft_positives_dist_threshold,
                                                                 return_distance=False)
        elif dataset_name == 'tokyo247':
            if not os.path.exists(self.dataset_folder):
                raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
            matdata = scio.loadmat(f'{self.dataset_folder}/database_gsv_vga/netvlad_v100_datasets/datasets/{dataset_name}.mat')['dbStruct'].item()
            self.database_paths = [f[0].item() for f in matdata[1]]
            self.database_paths = [join(self.dataset_folder, 'database_gsv_vga', path[:-3] + 'png')
                                   for path in self.database_paths]
            self.database_utms = matdata[2].T
            self.queries_paths = [f[0].item() for f in matdata[3]]
            self.queries_paths = [join(self.dataset_folder, 'query/247query', path)
                                  for path in self.queries_paths]
            self.queries_utms = matdata[4].T
            
            # Find soft_positives_per_query, which are within soft_positives_dist_threshold (25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                                 radius=args.soft_positives_dist_threshold,
                                                                 return_distance=False)
        elif dataset_name == 'nordland':
            if not os.path.exists(self.dataset_folder):
                raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
            self.database_paths = sorted(glob(f"{self.dataset_folder}/database/*.*", recursive=True),
                                         key=lambda x: int(os.path.basename(x).split('.')[0]))
            self.queries_paths = sorted(glob(f"{self.dataset_folder}/queries/*.*", recursive=True),
                                        key=lambda x: int(os.path.basename(x).split('.')[0]))
            self.soft_positives_per_query = []
            image_nums = len(self.queries_paths)
            for i in range(image_nums):
                self.soft_positives_per_query.append(list(range(max(0, i-2), min(image_nums, i+3))))

        if not self.images_paths:
            self.images_paths = list(self.database_paths) + list(self.queries_paths)
            self.database_num = len(self.database_paths)
            self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = self.base_transform(img)
        return img, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        if self.dataset_name == 'gsv_cities':
            return f"< {self.__class__.__name__}, {self.dataset_name} - #images: {len(self.images_paths)}; >"
        else:
            return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; " \
                   f"#queries: {self.queries_num} >"

    def get_positives(self):
        return self.soft_positives_per_query



