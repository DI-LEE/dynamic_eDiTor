#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
from typing import Any
sys.path.append('./src')
sys.path.append('./src/4DGaussians')
import numpy as np
import re
import random
import os
import shutil
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from tqdm import tqdm
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils.scene_utils import render_training_image
from datetime import datetime
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from PIL import Image

'''Qwen'''
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, QwenImageTransformer2DModel
from qwen_pipeline.pipeline import QwenImageEditPipeline

# Custom attention processor for multi-keyframe editing
from utils_ours.qwen_exattn_token_local import register_extended_attn
from utils_ours.pipeline_qwenimage_edit_local import qwenimage_pipeline_ours
# batch processing
from utils_ours.camtime_grid_utils import build_camtime_grid, generate_subgrid

from render import render_sets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_raft_weights = Raft_Large_Weights.DEFAULT
RAFT = raft_large(weights=_raft_weights, progress=True).to(device).eval()

import lpips
lpips_model = lpips.LPIPS(net='alex').to(device).eval()

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

@torch.no_grad()
def compute_flow_raft(img1_pil: Image.Image, img2_pil: Image.Image) -> torch.Tensor:
    """
    Compute optical flow from img1 -> img2 using RAFT.
    Returns: flow tensor of shape [1, 2, H, W] (in pixels).
    """
    # Convert to tensors [0,1], shape [1,3,H,W]
    im1 = TF.to_tensor(img1_pil).unsqueeze(0).to(device)
    im2 = TF.to_tensor(img2_pil).unsqueeze(0).to(device)

    # Ensure same spatial size (resize img2 to img1 if needed)
    H, W = im1.shape[-2:]
    if im2.shape[-2:] != (H, W):
        im2 = F.interpolate(im2, size=(H, W), mode="bilinear", align_corners=False)

    # RAFT prefers dims divisible by 8 — pad then unpad the output flow
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)

    im1_pad = F.pad(im1, pad, mode="replicate")
    im2_pad = F.pad(im2, pad, mode="replicate")

    # Forward pass: returns list of refinements; take last
    flow_list = RAFT(im1_pad, im2_pad)   # list of [B,2,H_pad,W_pad]
    flow = flow_list[-1]
    # Unpad back to original size
    flow = flow[..., :H, :W]             # [1,2,H,W]
    return flow


def init_qwen_pipeline(resgister_extended_attn=False, layer_range=None, cache_root=None, keep_last=2):
    """Initialize and configure the Qwen Image Edit Pipeline.
    
    Args:
        resgister_extended_attn: Whether to register extended attention processor
        last_n_layers: Apply to last N layers
        first_n_layers: Apply to first N layers
        middle_n_layers: Apply to middle N layers
        layer_range: Tuple of (start, end) for layer range
    """
    model_id = "Qwen/Qwen-Image-Edit"
    torch_dtype = torch.bfloat16
    device = "cuda"
    torch.cuda.empty_cache() 

    quantization_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    ).to(device)

    # register custom transformer block
    if resgister_extended_attn:
        register_extended_attn(
            transformer,
            layer_range=layer_range,
        )

    # import ipdb; ipdb.set_trace()
    quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    ).to(device)

    pipe = QwenImageEditPipeline.from_pretrained(
        model_id, 
        transformer=transformer,
        text_encoder=text_encoder, 
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    

    # optionally load LoRA weights to speed up inference
    pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")

    pipe.set_progress_bar_config(disable=True)

    return pipe

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer, 
                         prompt=None, first_idx=None, write_local_log=False, 
                         densification_iterations=None, editing_iterations=None, vanilla_editing=False):
    first_iter = 0

    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = opt.iterations + editing_iterations
    print(f"final_iter: {final_iter}")
    first_iter += 1

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = [i for i in train_cams]

    if checkpoint:
        first_iter = int(checkpoint) + 1
    count = 0
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")

    if stage == "edit":
        edited_dataset = {}

        # Get training cameras (returns a Dataset object)
        viewpoint_stack = scene.getTrainCameras()
        # Get FPS-adjusted dataset object
        fps_adjusted_stack = scene.getFPSAdjustedTrainCameras(target_fps=dataset.edit_fps, original_fps=30) # fps_adjusted_stack[0].cam_id -> 'cam01' #fps_adjusted_stack[0].time -> 0.0 // fps_adjusted_stack[3].time -> 0.3

        print(f"Total frames: {len(viewpoint_stack)}, Sampling FPS: {dataset.edit_fps}")
        print(f"FPS-adjusted frames: {len(fps_adjusted_stack)}")

        generator = torch.Generator(device="cuda").manual_seed(42)

        if write_local_log:
            log_folder = os.path.join(args.model_path, 'log', datetime.now().strftime('%Y_%m_%d_%H-%M-%S'))
            os.makedirs(log_folder, exist_ok=True)

        # Build grid
        train_cameras = []
        test_cameras = []
        if dataset.square_window:
            cam_ids_sorted, time_length, grid = build_camtime_grid(fps_adjusted_stack)
            total_windows = (time_length - 1) // 1 + (len(cam_ids_sorted) - 2) * ((time_length - 1) // 2)
        
        num_samples = len(fps_adjusted_stack)
        print(f"Editing {num_samples} frames (using fps_adjusted_stack)")

        # Initialize editing module
        if vanilla_editing:
            Editing_module = init_qwen_pipeline()
        else:
            if dataset.layer_range and isinstance(dataset.layer_range, str):
                layer_range_tuple = tuple(map(int, dataset.layer_range.split(',')))
            else:
                layer_range_tuple = dataset.layer_range
            Editing_module = init_qwen_pipeline(
                resgister_extended_attn=True, 
                layer_range=layer_range_tuple,
            )
        
        # Start 2D editing
        if vanilla_editing:
            description = "Vanilla Editing"
        else:
            description = f"ExtendedAttn Editing (layer range: {dataset.layer_range})"

        to_tensor = T.ToTensor()
        current_cam = None
        cache_root = os.path.join(args.model_path, 'attn_cache')
        os.makedirs(cache_root, exist_ok=True)

        original_folder_path = os.path.join(args.model_path, 'original_images')
        os.makedirs(original_folder_path, exist_ok=True)

        bottom_token_replace = False
        is_cache_bottom = False

        for i, windows in enumerate[list[tuple[int, int, Any]]](tqdm(generate_subgrid(grid), desc=description, total=total_windows)):
            cam_id = windows[0][2].cam_id
            if current_cam is None or current_cam != cam_id:
                current_cam = cam_id
                is_first = True
                token_replace = False

                is_cache_bottom = True
                bottom_token_replace = True
            else:
                is_first = False
                token_replace = True

                is_cache_bottom = False
                bottom_token_replace = False

            batch_images = []
            batch_times = []
            batch_cam_ids = []
            batch_flows = []
            batch_flows_backward = []

            for cam_idx, time, camera in windows:
                src_image = camera.original_image
                image_height = src_image.shape[1]
                image_width = src_image.shape[2]
                if isinstance(src_image, torch.Tensor):
                    to_pil = T.ToPILImage()
                    src_image_pil = to_pil(torch.clamp(src_image.detach().cpu(), 0, 1))
                else:
                    src_image_pil = src_image
                batch_images.append(src_image_pil)
                batch_times.append(time)
                batch_cam_ids.append(camera.cam_id)
            
            if token_replace:
                batch_flows.append(compute_flow_raft(batch_images[2], batch_images[0]))
                batch_flows.append(compute_flow_raft(batch_images[3], batch_images[1]))
                batch_flows_backward.append(compute_flow_raft(batch_images[0], batch_images[2]))
                batch_flows_backward.append(compute_flow_raft(batch_images[1], batch_images[3]))

            batch_size = len(batch_images)
            prompts = [prompt] * batch_size

            with torch.inference_mode():
                edited_output = qwenimage_pipeline_ours(
                    Editing_module,
                    image=batch_images,
                    prompt=prompts,
                    generator=generator,
                    num_inference_steps=8,
                    height=768,
                    width=768,
                    cache_dir=cache_root,
                    token_replace=token_replace,
                    ref_flows=batch_flows,
                    ref_flows_backward=batch_flows_backward,
                    is_cache_bottom=is_cache_bottom,
                    bottom_token_replace=bottom_token_replace,
                )

            edited_images = edited_output # [<PIL.Image.Image image mode=RGB size=1184x896 at 0x7FD3D4727EE0>, <PIL.Image.Image image mode=RGB size=1184x896 at 0x7FD3D4727100>, <PIL.Image.Image image mode=RGB size=1184x896 at 0x7FD3D4725060>]

            # for j, ((cam_idx, time, camera)) in enumerate(windows):
            for j, ((cam_idx, time, camera), edited_image_pil) in enumerate(zip(windows, edited_images)):
                if not isinstance(edited_image_pil, torch.Tensor):
                    edited_image = to_tensor(edited_image_pil) # torch.Size([3, 896, 1184])
                    edited_image = TF.resize(edited_image, (image_height, image_width), antialias=True) # torch.Size([3, 507, 676])

                if edited_image.dim() == 2:
                    edited_image = edited_image.unsqueeze(0)
                if edited_image.shape[0] == 1 and camera.original_image.shape[0] == 3:
                    edited_image = edited_image.repeat(3, 1, 1)
                edited_image = edited_image.to(camera.original_image.device)

                src_image = camera.original_image
                original_image_filename = os.path.join(original_folder_path, f'{camera.cam_id}_time{time}.png')
                if not os.path.exists(original_image_filename):
                    torchvision.utils.save_image(src_image, original_image_filename)

                if camera.uid not in edited_dataset:
                    edited_dataset[camera.uid] = [edited_image, camera]
                    train_cameras.append(camera)
                    
        for filename in os.listdir(cache_root):
            file_path = os.path.join(cache_root, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared files in {cache_root}")

        # Time reporting removed
        
    edited_image_idx_list = list(edited_dataset.keys())
    N = len(edited_image_idx_list)
    order = torch.randperm(N).tolist()
    for iteration in range(first_iter, final_iter+1):
        k = (iteration - first_iter) % N
        if iteration % N == 0:
            order = torch.randperm(N).tolist()
        viewpoint_stack_idx = order[k]
        current_cam_key = edited_image_idx_list[viewpoint_stack_idx]

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1

                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time

                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # dynerf's branch
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        edited_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        viewpoint_cam = edited_dataset[current_cam_key][1]

        now_stage = stage if stage != "edit" else "fine"
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=now_stage,cam_type=scene.dataset_type)
        rendered_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image_height = rendered_image.shape[1]
        image_width = rendered_image.shape[2]
        images.append(rendered_image.unsqueeze(0))

        if stage == "edit":
            edited_image = edited_dataset[current_cam_key][0].cuda()
            edited_images.append(edited_image.unsqueeze(0))
            edited_image_tensor = torch.cat(edited_images,0)

        if scene.dataset_type!="PanopticSports":
            gt_image = viewpoint_cam.original_image.cuda()
        else:
            gt_image  = viewpoint_cam['image'].cuda()
        gt_images.append(gt_image.unsqueeze(0))
        gt_image_tensor = torch.cat(gt_images,0)

        radii_list.append(radii.unsqueeze(0))
        visibility_filter_list.append(visibility_filter.unsqueeze(0))
        viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)

        if iteration % 100 == 0:
            os.makedirs(os.path.join(args.model_path, 'vis_combined'), exist_ok=True)
            vis_combined = torch.cat([gt_image[None], edited_image[None], rendered_image[None]], dim = 0).float()
            torchvision.utils.save_image(vis_combined, os.path.join(args.model_path, 'vis_combined', str(iteration).zfill(6) + '.png'), nrow=vis_combined.shape[0])
        
        # Loss
        # breakpoint()
        if stage == "edit":
            Ll1 = l1_loss(image_tensor, edited_image_tensor[:,:3,:,:])
            psnr_ = psnr(image_tensor, edited_image_tensor).mean().double()
            ssim_ = ssim(image_tensor, edited_image_tensor[:,:3,:,:])
            lpips_ = lpips_loss(image_tensor, edited_image_tensor[:,:3,:,:], lpips_model)
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        
        loss = Ll1 + tv_loss

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,edited_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 1 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], "fine", scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter or stage == 'edit' and iteration > opt.iterations + densification_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                elif stage == "fine" or stage == "edit":
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter)  
                else:
                    opacity_threshold = opt.opacity_threshold_fine_after
                    densify_threshold = opt.densify_grad_threshold_after

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if stage != "edit":
                    if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                        gaussians.grow(5,5,scene.model_path,iteration,stage)
                    if iteration % opt.opacity_reset_interval == 0:
                        print("reset opacity")
                        gaussians.reset_opacity()
                    
            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
        
    # Final execution time reporting removed

    return train_cameras, test_cameras


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, 
            checkpoint_iterations, checkpoint, debug_from, expname, editing_flag=None, prompt=None,
            first_idx=None, write_local_log=None, densification_iterations=None, editing_iterations=None, vanilla_editing=False):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    timer.start()
    train_cameras = []
    test_cameras = []
    if editing_flag is not None:
        ################# for editing #################
        scene = Scene(dataset, gaussians, load_iteration=checkpoint, shuffle=False)
        train_cameras, test_cameras = scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "edit", tb_writer, opt.iterations + opt.edit_iterations, timer, 
                                prompt, first_idx, write_local_log, densification_iterations, editing_iterations, vanilla_editing)
    else:
        scene = Scene(dataset, gaussians, load_iteration=checkpoint)
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer)
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations, timer)
    
    return train_cameras, test_cameras

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # Save a frozen copy of this script to the output directory
    try:
        current_file = os.path.abspath(__file__)
        shutil.copyfile(current_file, os.path.join(args.model_path, "frozen.py"))
    except Exception as e:
        print(f"Warning: failed to save frozen script copy: {e}")
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--seed', type=int, default=1004)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 14000, 15000, 16000, 20000, 24000, 29000, 30000, 34000,40000, 50000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument('--editing_flag',  action='store_true', default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--write_local_log', action='store_true', default=True)
    parser.add_argument('--densification_iterations', type=int, default=10000)
    parser.add_argument('--editing_iterations', type=int, default=10000)
    parser.add_argument('--vanilla_editing', action='store_true', default=False)
    parser.add_argument('--rendering_flag', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    setup_seed(args.seed)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train_cameras, test_cameras = training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, \
            args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, \
            editing_flag=args.editing_flag, prompt=args.prompt, \
            first_idx=args.idx, write_local_log=args.write_local_log, \
            densification_iterations=args.densification_iterations, editing_iterations=args.editing_iterations, vanilla_editing=args.vanilla_editing)
    print("\nTraining complete.")

    if args.rendering_flag:
        iteration = args.start_checkpoint + args.editing_iterations
        render_sets(lp.extract(args), hp.extract(args), iteration, pp.extract(args), 1, 0, 0)
    
    print("\nAll done.")

