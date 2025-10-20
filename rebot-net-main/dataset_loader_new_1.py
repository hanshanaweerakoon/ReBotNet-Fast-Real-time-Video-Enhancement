# dataset_loader_new.py

#data
import numpy as np
import random
import torch
from pathlib import Path
import torch.utils.data as data
# NOTE: Assuming utils_video and basicsr.data.degradations are available via sys.path
import utils.utils_video as utils_video 
from basicsr.data import degradations as degradations
import math
import cv2

# CRITICAL FIX: Global variable required by the indexing logic.
val_partition = set() 

# --- New Global Target Size ---
TARGET_SIZE = 384
# ---

def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts."""
    img = np.clip(img, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img


def random_bivariate_Gaussian(kernel_size,
                              sigma_x_range,
                              sigma_y_range,
                              rotation_range,
                              noise_range=None,
                              isotropic=True):
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels."""
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    
    # Use the full random range directly from the input ranges
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        rotation = 0
        sigma_y = sigma_x

    # Assuming basicsr.data.degradations.bivariate_Gaussian exists and works
    kernel = degradations.bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    
    kernel = kernel / np.sum(kernel)
    return kernel

def random_mixed_kernels(kernel_list,
                              kernel_prob,
                              kernel_size=21,
                              sigma_x_range=(0.6, 5),
                              sigma_y_range=(0.6, 5),
                              rotation_range=(-math.pi, math.pi),
                              betag_range=(0.5, 8),
                              betap_range=(0.5, 8),
                              noise_range=None): 
    """Randomly generate mixed kernels using full random ranges."""
    
    kernel_type = 'aniso' # Default to anisotropic Gaussian for simplicity

    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    # ... (Other kernel types like 'generalized_iso' would use similar random sampling)
    
    return kernel


class PortraitVideoRecurrentTrainDataset(data.Dataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', TARGET_SIZE)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.filename_tmpl = opt.get('filename_tmpl', '08d')
        self.filename_ext = opt.get('filename_ext', 'png')
        self.num_frame = opt['num_frame']

        keys = []
        total_num_frames = [] 
        start_frames = [] 
        
        # *** CRITICAL FIX 3: Removed deterministic rand_num generation ***
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _, start_frame = line.split(' ')
                # Append one entry for every frame in the video clip
                keys.extend([f'{folder}' for i in range(int(start_frame), int(start_frame)+int(frame_num))])
                total_num_frames.extend([int(frame_num) for i in range(int(frame_num))])
                start_frames.extend([int(start_frame) for i in range(int(frame_num))])


        self.keys = []
        self.total_num_frames = [] 
        self.start_frames = []
        
        # Data partitioning for train/test mode remains the same
        if opt['test_mode']:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])

        # -----------------------------------------------------
        # *** CRITICAL FIX 4: Process io_backend_opt in __init__ ***
        # This prevents the KeyError in DataLoader workers.
        # -----------------------------------------------------
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        
        # Save type and kwargs, modifying the options dict once (if necessary)
        self.io_backend_type = self.io_backend_opt.pop('type', None)
        self.io_backend_kwargs = self.io_backend_opt
        
        if self.io_backend_type == 'lmdb':
            self.is_lmdb = True
            # LMDB client setup logic (adjusting self.io_backend_kwargs)...

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)

        # Degradation parameters
        self.blur_kernel_size = 15
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = (0.1, 3.0) 
        self.downsample_range = (0.8, 2.5)
        self.jpeg_range = (70, 100)

        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                            f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        # -----------------------------------------------------
        # *** CRITICAL FIX 5: Initialize FileClient safely ***
        # Uses pre-processed type and kwargs, avoiding concurrency errors.
        # -----------------------------------------------------
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_type, **self.io_backend_kwargs)

        key = self.keys[index] 
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        
        category_folder, sequence_folder = key.split('/') 

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(start_frames)
        
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        img_lqs = []
        img_gts = []

        for neighbor in neighbor_list:
            
            # Path calculation
            if self.is_lmdb:
                img_lq_path = f'{category_folder}/{sequence_folder}/{neighbor:08d}' 
                img_gt_path = f'{category_folder}/{sequence_folder}/{neighbor:08d}' 
            else:
                img_lq_path = self.lq_root / 'sequences' / category_folder / sequence_folder / f'im{neighbor}.png' 
                img_gt_path = self.gt_root / 'sequences' / category_folder / sequence_folder / f'im{neighbor}.png'

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            # CRITICAL RESIZE 1: Resize GT image to 384x384
            img_gt = cv2.resize(img_gt, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            img_gts.append(img_gt)

            # ------------------------ generate lq image ------------------------ #
            
            # *** CRITICAL FIX 6: Degradation is now applied with true per-frame randomness ***
            
            # 1. Blur Kernel Generation (uses internal randomness)
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, 
                [-math.pi, math.pi],
                noise_range=None) 
            img_lq = cv2.filter2D(img_gt, -1, kernel)
            
            # 2. Downsample and JPEG parameters are now fully random per frame
            scale_random = random.uniform(self.downsample_range[0], self.downsample_range[1])
            jpeg_rand = random.uniform(self.jpeg_range[0], self.jpeg_range[1])

            # Perform the random blind degradation downsample
            img_lq = cv2.resize(img_lq, 
                                (int(TARGET_SIZE / scale_random), int(TARGET_SIZE / scale_random)), 
                                interpolation=cv2.INTER_LINEAR)
            
            if self.jpeg_range is not None:
                img_lq = add_jpg_compression(img_lq, jpeg_rand)

            # CRITICAL RESIZE 3: Resize LQ image back up to 384x384 (full input size)
            img_lq = cv2.resize(img_lq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            
            img_lqs.append(img_lq)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot']) 

        img_results = utils_video.img2tensor(img_results)
        
        half_len = len(img_results) // 2 
        img_gts = torch.stack(img_results[half_len:], dim=0)
        img_lqs = torch.stack(img_results[:half_len], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)