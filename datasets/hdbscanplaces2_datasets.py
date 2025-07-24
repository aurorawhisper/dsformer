import torch
import random
import logging
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as tfm
ImageFile.LOAD_TRUNCATED_IMAGES = True

PANO_WIDTH = int(512 * 6.5)


class HDBScanPlaces2(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, current_group=0, min_images_per_class=10, angle=0):
        super().__init__()
        self.block_width = 10
        self.block_num = 5
        self.focal_dist = 15
        self.dataset_folder = dataset_folder

        filename = f"cache/sfxl_bw{self.block_width}_bn{self.block_num}_foc{self.focal_dist}_mipc{min_images_per_class}.torch"
        if current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        self.group_paths, _, _ = torch.load(filename)
        self.current_group_paths = self.group_paths[current_group]
        self.angle = angle

    def __getitem__(self, class_num):
        image_path = self.dataset_folder + f"/{self.angle}/" + random.choice(self.current_group_paths[class_num])
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = tfm.functional.to_tensor(pil_img)
        return pil_img, class_num, image_path

    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.current_group_paths[c]) for c in range(len(self))])

    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.current_group_paths)

