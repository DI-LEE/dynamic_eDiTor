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
sys.path.append('./src/Deformable-3D-Gaussians')
import os
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import numpy as np
import re
import random
import shutil
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim#, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from tqdm import tqdm
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams#, ModelHiddenParams
from torch.utils.data import DataLoader
# from utils.timer import Timer
# from utils.loader_utils import FineSampler, get_stamp_list
# from utils.scene_utils import render_training_image
import copy
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
from utils_ours.qwen_exattn_token_gpu import register_extended_attn
# from utils_ours.qwen_exattn_token_local import register_extended_attn
from utils_ours.pipeline_qwenimage_edit_gpu import qwenimage_pipeline_ours
# from utils_ours.pipeline_qwenimage_edit_local import qwenimage_pipeline_ours
# batch processing
from utils_ours.process_blockwise import build_camtime_grid, generate_square_windows, generate_grid_windows_for_optflow, build_mono_grid, generate_mono_windows

from render import render_sets
# from render_evaluation import render_sets_for_metric, render_sets_for_metric_train_cameras

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, load_checkpoint=None, iteration=None, prompt=None, first_idx=0, write_local_log=False, vanilla_editing=False):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    if load_checkpoint is not None:
        deform.load_weights(dataset.model_path, iteration=iteration)
        scene = Scene(dataset, gaussians, load_iteration=load_checkpoint, shuffle=False)
    else:
        scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras()
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    
    start_iter = load_checkpoint + 1 if load_checkpoint is not None else 1

    mono_grid = build_mono_grid(viewpoint_stack) # len(mono_grid) = 180 for mochi-high-five
    # import ipdb; ipdb.set_trace()
    mono_windows = generate_mono_windows(mono_grid)

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
    
    edited_dataset = {}
    generator = torch.Generator(device="cuda").manual_seed(42)

    to_tensor = T.ToTensor()
    is_first = True
    token_replace = None
    prev_right_cache = None
    for i, windows in enumerate[list[tuple[int, int, Any]]](tqdm(mono_windows, desc=description)):
        if is_first:
            token_replace = False
        else:
            token_replace = True

        batch_images = []
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

        batch_size = len(batch_images)
        prompts = [prompt] * batch_size

        if token_replace and len(windows) == 4:
            batch_flows.append(compute_flow_raft(batch_images[2], batch_images[1]))
            batch_flows.append(compute_flow_raft(batch_images[3], batch_images[1]))
            batch_flows_backward.append(compute_flow_raft(batch_images[1], batch_images[2]))
            batch_flows_backward.append(compute_flow_raft(batch_images[1], batch_images[3]))
            
        with torch.inference_mode():
            edited_output, right_cache = qwenimage_pipeline_ours(
                Editing_module,
                image=batch_images,
                prompt=prompts,
                generator=generator,
                num_inference_steps=8,
                height=768,
                width=768,
                token_replace=token_replace,
                ref_flows=batch_flows,
                ref_flows_backward=batch_flows_backward,
                prev_right_cache=prev_right_cache if token_replace else None,
            )

        edited_images = edited_output

        if is_first:
            is_first = False

        prev_right_cache = right_cache
        right_cache = None
        torch.cuda.empty_cache()

        for j, ((cam_idx,time, camera), edited_image_pil) in enumerate(zip(windows, edited_images)):
            if not isinstance(edited_image_pil, torch.Tensor):
                edited_image = to_tensor(edited_image_pil) # torch.Size([3, 896, 1184])
                edited_image = TF.resize(edited_image, (image_height, image_width), antialias=True) # torch.Size([3, 507, 676])

            if edited_image.dim() == 2:
                edited_image = edited_image.unsqueeze(0)
            if edited_image.shape[0] == 1 and camera.original_image.shape[0] == 3:
                edited_image = edited_image.repeat(3, 1, 1)
            edited_image = edited_image.to(camera.original_image.device)

            if camera.uid not in edited_dataset:
                edited_dataset[camera.uid] = [edited_image, camera]

                if vanilla_editing:
                    os.makedirs(os.path.join(args.model_path, 'vanilla_edited_images'), exist_ok=True)
                    filename = f'{camera.image_name}.png'
                    torchvision.utils.save_image(edited_image, os.path.join(args.model_path, 'vanilla_edited_images', filename))
                else:
                    edited_folder_name = f'edited_{dataset.layer_range}_full'
                    os.makedirs(os.path.join(args.model_path, edited_folder_name), exist_ok=True)
                    filename = f'{camera.image_name}.png'
                    torchvision.utils.save_image(edited_image, os.path.join(args.model_path, edited_folder_name, filename))

    prev_right_cache = None
    right_cache = None
    del Editing_module
    torch.cuda.empty_cache()
    
    edited_image_idx_list = list(edited_dataset.keys())
    num_edited_images = len(edited_image_idx_list)
    order = torch.randperm(num_edited_images).tolist()
#==========================================================================================================
    # Editing loop starts from opt.iterations + 1
    edit_start_iter = opt.iterations + 1
    print(f"[DEBUG] start_iter: {start_iter}, opt.iterations: {opt.iterations}")
    print(f"[DEBUG] num_edited_images: {num_edited_images}")
    print(f"[DEBUG] edited_image_idx_list: {edited_image_idx_list}")
    print(f"[DEBUG] Editing range: {edit_start_iter} to {opt.iterations + opt.edit_iterations}")

    for iteration in tqdm(range(edit_start_iter, opt.iterations + opt.edit_iterations + 1), desc="optimizing progress"):

        k = (iteration - edit_start_iter) % num_edited_images
        if k == 0:
            order = torch.randperm(num_edited_images).tolist()
        viewpoint_stack_idx = order[k]
        current_cam_key = edited_image_idx_list[viewpoint_stack_idx]

        iter_start.record()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam = edited_dataset[current_cam_key][1]
    
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        edited_image = edited_dataset[current_cam_key][0].cuda()
        # edited_image_tensor = torch.cat(edited_image,0)

        if iteration % 100 == 0:
            os.makedirs(os.path.join(args.model_path, 'vis_combined'), exist_ok=True)
            vis_combined = torch.cat([gt_image[None], edited_image[None], image[None]], dim = 0).float()
            torchvision.utils.save_image(vis_combined, os.path.join(args.model_path, 'vis_combined', str(iteration).zfill(6) + '.png'), nrow=vis_combined.shape[0])        

        # Loss
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(image, edited_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, edited_image))
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations + opt.edit_iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))



    if write_local_log:
        log_folder = os.path.join(args.model_path, 'log', datetime.now().strftime('%Y_%m_%d_%H-%M-%S'))
        os.makedirs(log_folder, exist_ok=True)

    print('Editing Finished!')

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, is_edit=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)

                    if not is_edit:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.edit_image.to("cuda"), 0.0, 1.0)

                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6999)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 60001, 5000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--load_checkpoint', type=int, default=None)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--write_local_log', action='store_true', default=False)
    parser.add_argument('--vanilla_editing', action='store_true', default=False)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
             args.save_iterations, load_checkpoint=args.load_checkpoint, iteration=args.iterations, \
            prompt=args.prompt, first_idx=args.idx, write_local_log=args.write_local_log, vanilla_editing=args.vanilla_editing)
    
    render_sets(lp.extract(args), 70000, pp.extract(args), args.skip_train, args.skip_test, args.mode)
    print("\nTraining complete.")