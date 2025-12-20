from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time, image_path, cam_id = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                if image is not None:
                    # Default to an all-ones mask of shape (1, H, W)
                    height = image.shape[1]
                    width = image.shape[2]
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                image_path = getattr(caminfo, 'image_path', None)
                cam_id = getattr(caminfo, 'cam_id', None)
                
                if image is not None:
                    # Default to an all-ones mask of shape (1, H, W)
                    height = image.shape[1]
                    width = image.shape[2]
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              image_path=image_path, cam_id=cam_id)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
