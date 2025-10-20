# /rebot-net-main/utils/utils_video.py

# ======================================================================
# 1. I/O (Input/Output) and General Image Utilities
# ======================================================================

# Import FileClient (Still correct location)
from basicsr.utils.file_client import FileClient

# Import image processing fundamentals (Still correct location)
from basicsr.utils.img_util import (
    imfrombytes,  # Decodes image bytes into a numpy array
    img2tensor    # Converts numpy array to PyTorch tensor
)

# Import Data Augmentation/Transform functions (NEW LOCATION)
from basicsr.data.transforms import (
    augment,              # Handles random flip/rotation augmentation
    paired_random_crop    # Performs aligned cropping on LQ and GT images
)