import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# ============================================================
# DashGaussian 多分辨率课程调度
# ============================================================

def compute_dash_switching_iters(cameras_by_scale, total_iters):
    """
    DashGaussian (arXiv:2503.18402) frequency-guided resolution switching.
    χ(F) = mean DFT amplitude energy across sample cameras.
    switching_iter[scale] = total_iters * χ(F_scale) / χ(F_full)
    """
    full_scale = min(cameras_by_scale.keys())  # smallest scale value = highest res
    chi = {}
    for scale, cameras in cameras_by_scale.items():
        sample = cameras[:min(50, len(cameras))]
        energy = 0.0
        for cam in sample:
            img = cam.original_image  # (3, H, W) already on data_device
            img_cuda = img.cuda() if img.device.type == 'cpu' else img
            fft = torch.fft.fft2(img_cuda.float())
            energy += torch.abs(fft).mean().item()
        chi[scale] = energy / max(len(sample), 1)

    chi_full = chi[full_scale]
    switching_iters = {}
    for scale in sorted(cameras_by_scale.keys()):
        switching_iters[scale] = int(total_iters * chi[scale] / chi_full)
    switching_iters[full_scale] = total_iters  # full-res stage ends at total_iters

    print("[DashGaussian] Frequency energy per scale:")
    for s in sorted(switching_iters.keys(), reverse=True):
        print(f"  scale={s:.1f}  χ={chi[s]:.4f}  switch_iter={switching_iters[s]}")
    return switching_iters


def get_current_dash_scale(iteration, switching_iters):
    """Return current resolution scale (largest first = lowest res first)."""
    for scale in sorted(switching_iters.keys(), reverse=True):
        if iteration <= switching_iters[scale]:
            return scale
    return min(switching_iters.keys())


def compute_dash_budget(p_init, p_fin, r_scale, iteration, total_iters):
    """
    DashGaussian Gaussian count budget:
    P_i = P_init + (P_fin - P_init) / r^(2 - i/S)
    Suppresses primitive growth at low resolution, releases at full resolution.
    """
    exponent = 2.0 - iteration / max(total_iters, 1)  # 2.0 → 1.0
    denom = r_scale ** exponent                         # e.g. 4^2=16 early, 1^1=1 at end
    return p_init + (p_fin - p_init) / max(denom, 1e-6)


# ============================================================
# Loss 方案实现
# ============================================================

def fregs_loss(image, gt_image, iteration, max_iterations, lambda_freq=0.01):
    """方案一：FreGS 频率域正则化
    渐进式从低频到高频：前期只监督低频，后期逐渐加入高频。
    """
    # 当前迭代保留的频率比例 (0.1 -> 1.0)
    freq_ratio = 0.1 + 0.9 * (iteration / max_iterations)

    pred_fft = torch.fft.fft2(image)
    gt_fft   = torch.fft.fft2(gt_image)

    # 构造低通掩码
    H, W = image.shape[-2], image.shape[-1]
    mask = torch.zeros(H, W, device=image.device)
    h_cut = max(1, int(H * freq_ratio / 2))
    w_cut = max(1, int(W * freq_ratio / 2))
    mask[:h_cut, :w_cut] = 1
    mask[-h_cut:, :w_cut] = 1
    mask[:h_cut, -w_cut:] = 1
    mask[-h_cut:, -w_cut:] = 1

    pred_amp = torch.abs(pred_fft) * mask
    gt_amp   = torch.abs(gt_fft)   * mask

    return lambda_freq * F.l1_loss(pred_amp, gt_amp)


def build_vgg():
    """构建 VGG-19 特征提取器（只用第一个 conv block）"""
    import torchvision
    vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    extractor = torch.nn.Sequential(*list(vgg.features.children())[:4]).cuda().eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    # ImageNet 归一化
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    return extractor, mean, std


def ahgs_loss(image, gt_image, iteration, max_iterations, vgg_extractor, vgg_mean, vgg_std,
              lambda_per=0.1, vgg_max_size=2048, full_res_start_iter=None):
    """方案二：AH-GS VGG 感知 loss + 衰减调度
    权重衰减：
      - 非 dash 路径或未提供 full_res_start_iter：按总 iter 线性衰减 (1 - k/iterations)
      - dash 路径：仅在 scale=1.0（全分辨率）阶段启用，在该阶段内部从 1 衰减到 0；
        warm-up 阶段 VGG 输入是低频图，特征无意义，权重直接置 0。
    显存控制：进 VGG 前把长边下采样到 vgg_max_size，避免 4K 下显存爆炸。
    """
    if full_res_start_iter is not None:
        if iteration < full_res_start_iter:
            return torch.tensor(0.0, device=image.device)
        progress = (iteration - full_res_start_iter) / max(max_iterations - full_res_start_iter, 1)
        decay = max(0.0, 1.0 - progress)
    else:
        decay = 1.0 - iteration / max_iterations
    if decay <= 0:
        return torch.tensor(0.0, device=image.device)

    pred = image.unsqueeze(0)
    gt   = gt_image.detach().unsqueeze(0)

    H, W = pred.shape[-2], pred.shape[-1]
    long_edge = max(H, W)
    if long_edge > vgg_max_size:
        scale = vgg_max_size / long_edge
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        pred = F.interpolate(pred, size=(new_h, new_w), mode="bilinear", align_corners=False)
        gt   = F.interpolate(gt,   size=(new_h, new_w), mode="bilinear", align_corners=False)

    pred = (pred - vgg_mean) / vgg_std
    gt   = (gt   - vgg_mean) / vgg_std
    pred_feat = vgg_extractor(pred)
    with torch.no_grad():
        gt_feat = vgg_extractor(gt)
    return lambda_per * decay * F.mse_loss(pred_feat, gt_feat)


def depth_reg_loss(render_pkg, lambda_depth=0.01):
    """方案三：深度图 TV 正则化（边缘感知）
    在纹理均匀区域鼓励深度平滑，减少 floater。
    """
    depth = render_pkg.get("depth", None)
    if depth is None:
        return torch.tensor(0.0)
    # depth: [1, H, W]
    dx = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
    dy = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])
    return lambda_depth * (dx.mean() + dy.mean())


def normal_reg_loss(render_pkg, lambda_normal=0.01):
    """方案四：从渲染深度图推导法线，加平滑正则
    深度梯度构成法线场，鼓励法线方向连续。
    """
    depth = render_pkg.get("depth", None)
    if depth is None:
        return torch.tensor(0.0)
    # depth: [1, H, W]
    d = depth[0]  # [H, W]
    dz_dx = d[:, 1:] - d[:, :-1]   # [H, W-1]
    dz_dy = d[1:, :] - d[:-1, :]   # [H-1, W]

    # 法线 = normalize([-dz_dx, -dz_dy, 1])，只取 x/y 分量做平滑
    normal_x = F.pad(dz_dx, (0, 1))  # [H, W]
    normal_y = F.pad(dz_dy, (0, 0, 0, 1))  # [H, W]

    smooth_x = torch.abs(normal_x[:, 1:] - normal_x[:, :-1]).mean()
    smooth_y = torch.abs(normal_y[1:, :] - normal_y[:-1, :]).mean()
    return lambda_normal * (smooth_x + smooth_y)


# ============================================================
# 训练主函数（在原版 train.py 基础上修改 loss 部分）
# ============================================================

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, loss_type,
             wandb_project=None, wandb_name=None, wandb_run=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # 初始化 wandb
    if wandb_run is None and WANDB_FOUND and wandb_project:
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            dir=dataset.model_path,
            config={
                "loss_type": loss_type,
                "iterations": opt.iterations,
                "lambda_dssim": opt.lambda_dssim,
                "source_path": dataset.source_path,
                "model_path": dataset.model_path,
            },
            resume="allow",
        )
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    if opt.use_dash:
        dash_resolution_scales = [float(2**i) for i in range(opt.dash_r_stages)]  # [1.0, 2.0, 4.0]
        scene = Scene(dataset, gaussians, resolution_scales=dash_resolution_scales)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # AH-GS: 预加载 VGG
    vgg_extractor = vgg_mean = vgg_std = None
    if loss_type == "ahgs":
        print("[INFO] Loading VGG-19 for perceptual loss...")
        vgg_extractor, vgg_mean, vgg_std = build_vgg()

    # DashGaussian: 频率引导分辨率调度 + 致密化预算
    dash_switching_iters = None
    dash_cameras_by_scale = None
    dash_prev_scale = None
    dash_full_res_start_iter = None  # 仅给 AHGS decay 用，position LR 不再依赖它重启
    dash_p_init = None
    dash_p_fin_momentum = None
    if opt.use_dash:
        print("[DashGaussian] Computing frequency-guided switching iterations...")
        dash_cameras_by_scale = {
            s: scene.getTrainCameras(scale=s).copy()
            for s in dash_resolution_scales
        }
        dash_switching_iters = compute_dash_switching_iters(dash_cameras_by_scale, opt.iterations)
        dash_prev_scale = max(dash_resolution_scales)  # start at lowest resolution
        dash_p_init = gaussians.get_xyz.shape[0]
        # p_fin momentum 初始值：max_gaussians（设为0时回退到 p_init*10）
        # 论文里 p_fin 是动态 momentum 估计的训练结束时目标上限；
        # 原来写成 p_init 导致 p_fin-p_init=0，budget 公式完全失效。
        dash_p_fin_momentum = float(opt.max_gaussians) if opt.max_gaussians > 0 else float(dash_p_init) * 10.0

    viewpoint_stack   = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log      = 0.0
    ema_Ll1depth_for_log  = 0.0

    # 总训练步数 = 主训练段（dash/LR/loss 节奏都以此为参考）+ 纯精修尾段（锁死高斯球，不再 densify/reset_opacity）
    total_iters = opt.iterations + max(int(getattr(opt, "refine_extra_iters", 0)), 0)
    progress_bar = tqdm(range(first_iter, total_iters), desc=f"Training [{loss_type}]")
    first_iter += 1
    for iteration in range(first_iter, total_iters + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer,
                                       use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # DashGaussian: 分辨率切换处理 + position LR 连续衰减
        if opt.use_dash:
            dash_current_scale = get_current_dash_scale(iteration, dash_switching_iters)
            if dash_current_scale != dash_prev_scale:
                old_scale = dash_prev_scale
                print(f"\n[DashGaussian] iter {iteration}: switching resolution scale {old_scale:.1f} → {dash_current_scale:.1f}")
                viewpoint_stack = dash_cameras_by_scale[dash_current_scale].copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))

                # Fix ①+③：预防性分裂。切换后 radii ≈ old_radii * (old_scale/new_scale)。
                # 目标：切换后屏幕半径 ≤ 20px @ 新 scale 像素空间（比常规阈值 40 更激进，因为切换是一次性机会）
                # 反推：old_radii ≤ 20 * (new_scale/old_scale) = 20 / ratio
                ratio = old_scale / dash_current_scale  # >1
                screen_thresh_old = 20.0 / ratio  # 旧 scale 像素空间下的阈值
                n_force_split = gaussians.force_split_by_screen_size(screen_thresh_old, N=2)
                print(f"[DashGaussian] iter {iteration}: prophylactic split {n_force_split} oversized gaussians "
                      f"(ratio {ratio:.1f}, thresh {screen_thresh_old:.1f}px @ scale={old_scale:.1f})")

                # Fix ②：梯度/像素/半径累加器一并清零，新 scale 决策完全用新 scale 信号
                gaussians.reset_densification_stats()

                dash_prev_scale = dash_current_scale
                # 记录全分辨率开始时间（仅给 AHGS decay 重新参数化用）
                if dash_current_scale == 1.0 and dash_full_res_start_iter is None:
                    dash_full_res_start_iter = iteration
                    print(f"[DashGaussian] iter {iteration}: full resolution reached")

            # position LR 一直按总 iter 连续衰减，不再 reset，避免切换瞬间打散低频骨架
            gaussians.update_learning_rate(iteration)
        else:
            gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            if opt.use_dash:
                viewpoint_stack = dash_cameras_by_scale[dash_prev_scale].copy()
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        try:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                                use_trained_exp=dataset.train_test_exp,
                                separate_sh=SPARSE_ADAM_AVAILABLE)
        except torch.cuda.OutOfMemoryError as e:
            # rasterizer 的 binning buffer 在巨型高斯/高 tile 覆盖下会爆。
            # 不让训练崩：清缓存 + 紧急裁掉 5% 最大的高斯，跳过当前 iter。
            torch.cuda.empty_cache()
            n_before = gaussians.get_xyz.shape[0]
            n_killed = gaussians.emergency_prune_largest(frac=0.05)
            print(f"\n[OOM-Guard] iter {iteration}: render OOM ({str(e).splitlines()[0]}), "
                  f"emergency-pruned {n_killed}/{n_before} largest gaussians, skip iter")
            torch.cuda.empty_cache()
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            continue

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"])

        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        # ---- 基础 Loss ----
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # ---- 额外 Loss ----
        # 进度按主训练段（opt.iterations）计算；纯精修尾段（iter > opt.iterations）clamp 到 1.0
        # 避免 ahgs_loss 的 decay = 1 - iter/max 变负、freq_ratio 越界等问题
        loss_iter = min(iteration, opt.iterations)
        if loss_type == "fregs":
            loss = loss + fregs_loss(image, gt_image, loss_iter, opt.iterations)

        elif loss_type == "ahgs":
            loss = loss + ahgs_loss(image, gt_image, loss_iter, opt.iterations,
                                    vgg_extractor, vgg_mean, vgg_std,
                                    full_res_start_iter=dash_full_res_start_iter if opt.use_dash else None)

        elif loss_type == "depth_reg":
            loss = loss + depth_reg_loss(render_pkg)

        elif loss_type == "normal_reg":
            loss = loss + normal_reg_loss(render_pkg)

        # ---- 深度监督（原版保留）----
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            n_before = gaussians.get_xyz.shape[0]
            n_killed = gaussians.emergency_prune_largest(frac=0.05)
            print(f"\n[OOM-Guard] iter {iteration}: backward OOM ({str(e).splitlines()[0]}), "
                  f"emergency-pruned {n_killed}/{n_before} largest gaussians, skip iter")
            torch.cuda.empty_cache()
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            continue
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log     = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth    + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                if wandb_run is not None:
                    log_dict = {
                        "train/ema_loss": ema_loss_for_log,
                        "train/l1_loss": Ll1.item(),
                        "train/total_loss": loss.item(),
                        "train/ema_depth_loss": ema_Ll1depth_for_log,
                        "train/iter_time_ms": iter_start.elapsed_time(iter_end),
                        "train/num_gaussians": scene.gaussians.get_xyz.shape[0],
                    }
                    if opt.use_dash:
                        log_dict["train/dash_resolution_scale"] = dash_prev_scale
                    wandb_run.log(log_dict, step=iteration)
            if iteration == total_iters:
                progress_bar.close()

            training_report(tb_writer, wandb_run, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations,
                            scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            densify_active = iteration < opt.densify_until_iter or opt.use_dash
            # 点数到达预算上限后，冻结点数：跳过所有致密化/剪枝/不透明度重置，只做纯参数训练。
            # 等价于原版 3DGS "densify_until_iter 之后不再增长高斯" 的语义。
            if opt.lock_after_budget:
                # v4 latch（粘性）：一旦触发永久保持，opacity_reset 也随 densify_active 一起关闭。
                # 触发条件二选一：(a) 点数达到 max_gaussians；(b) 进入纯精修尾段（iter > opt.iterations）。
                if not getattr(training, "_budget_locked", False):
                    hit_budget = opt.max_gaussians > 0 and gaussians.get_xyz.shape[0] >= opt.max_gaussians
                    in_refine_phase = iteration > opt.iterations
                    if hit_budget or in_refine_phase:
                        training._budget_locked = True
                        reason = (f"reached max_gaussians={opt.max_gaussians}" if hit_budget
                                  else f"entered refine_extra phase (iter > {opt.iterations})")
                        print(f"\n[Budget-Lock] iter {iteration}: {reason}, "
                              f"entering pure refinement (no densify / no opacity_reset)")
                if getattr(training, "_budget_locked", False):
                    densify_active = False
            else:
                # v3 原路径：每步重算，无 latch
                budget_reached = opt.max_gaussians > 0 and gaussians.get_xyz.shape[0] >= opt.max_gaussians
                if budget_reached:
                    if not getattr(training, "_budget_logged", False):
                        print(f"\n[Budget] iter {iteration}: reached max_gaussians={opt.max_gaussians}, "
                              f"freezing point count and switching to pure parameter training")
                        training._budget_logged = True
                    densify_active = False
            if densify_active:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radii,
                                                  resolution_scale=dash_current_scale if opt.use_dash else 1.0)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if opt.use_dash:
                        # DashGaussian: budget 按当前分辨率压制上限，低分辨率阶段分母大所以上限低
                        p_i = compute_dash_budget(dash_p_init, dash_p_fin_momentum,
                                                  dash_prev_scale, iteration, opt.iterations)
                        densify_rate = max(0.0, (p_i - gaussians.get_xyz.shape[0]) / max(gaussians.get_xyz.shape[0], 1))
                        densify_rate = min(densify_rate, 0.2)  # 原版上限：每步最多增长 20%
                        # size_threshold: 4K 下 40px 等价；max_radii2D 是当前 scale 下的像素半径，
                        # 所以阈值要除以当前 resolution_scale（scale=1.0→40, 2.0→20, 4.0→10）。
                        size_thresh_vs = 40.0 / dash_current_scale
                        n_natural = gaussians.prune_and_densify(opt.densify_grad_threshold, 0.005,
                                                                scene.cameras_extent, size_thresh_vs, radii,
                                                                densify_rate=densify_rate)
                        # momentum 更新（原版自适应模式）：max_gaussians > 0 时 p_fin 固定，此处是死代码，保留以便切换模式
                        if opt.max_gaussians <= 0:
                            dash_p_fin_momentum = max(dash_p_fin_momentum,
                                                      0.98 * dash_p_fin_momentum + 1.0 * float(n_natural))
                        if opt.max_gaussians > 0 and gaussians.get_xyz.shape[0] > opt.max_gaussians:
                            gaussians.prune_to_budget(opt.max_gaussians)
                    else:
                        # 非 dash 路径按原版 3DGS：opacity_reset 之后才启用屏幕尺寸阈值
                        size_thresh_vs = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                    scene.cameras_extent, size_thresh_vs, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < total_iters:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, wandb_run, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0; psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image    = image[...,    image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    l1_test   += l1_loss(image, gt_image).mean().double()
                    psnr_test  += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test   /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb_run is not None:
                    wandb_run.log({
                        f"eval/{config['name']}_l1_loss": float(l1_test),
                        f"eval/{config['name']}_psnr": float(psnr_test),
                    }, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        if wandb_run is not None:
            wandb_run.log({
                "scene/num_gaussians": scene.gaussians.get_xyz.shape[0],
            }, step=iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Custom loss training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default="baseline",
                        choices=["baseline", "fregs", "ahgs", "depth_reg", "normal_reg"],
                        help="Loss function variant")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(f"Loss type: {args.loss_type}")
    print(f"Optimizing {args.model_path}")

    safe_state(args.quiet)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint,
             args.debug_from, args.loss_type,
             wandb_project=args.wandb_project,
             wandb_name=args.wandb_name)

    print("\nTraining complete.")
