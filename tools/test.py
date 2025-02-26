import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.uniad.apis.custom_test import my_custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pickle

warnings.filterwarnings("ignore")

activations = {} 

def tensor_to_heatmap(feature_map: torch.Tensor) -> np.ndarray:
    """
    feature_map: 形状が (C, H, W) のTensor (1枚のカメラ分)を想定。
    戻り値: ヒートマップ（カラー）のBGR画像 (H, W, 3)
    """
    # ---- 1. チャネル方向を平均して1枚にする (H, W)
    # 必要に応じて mean/max/sum などを切り替える
    heatmap = feature_map.mean(dim=0)  # -> shape: (H, W)

    # ---- 2. テンソル → numpy & 正規化
    heatmap = heatmap.detach().cpu().numpy()
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)

    # ---- 3. カラーマップを適用 (OpenCVはBGR)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def overlay_heatmap_on_image(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    image_bgr: 元画像 (H, W, 3)
    heatmap_bgr: ヒートマップ (H, W, 3)
    alpha: ヒートマップ重ねる割合
    戻り値: ヒートマップを重ね合わせたBGR画像
    """
    # サイズが違う場合は合わせる
    if (image_bgr.shape[0] != heatmap_bgr.shape[0]) or (image_bgr.shape[1] != heatmap_bgr.shape[1]):
        heatmap_bgr = cv2.resize(heatmap_bgr, (image_bgr.shape[1], image_bgr.shape[0]))

    # addWeighted でオーバーレイ
    overlaid = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlaid

def project_points_to_image(points_phys, lidar2img, img_hw):
    """
    points_phys: shape (M,3) in physical coords
    lidar2img: shape(4,4)
    img_hw: (height, width)
    return: (u,v, mask) each shape(M,)
    """
    M = points_phys.shape[0]
    # hom
    ones = points_phys.new_ones((M,1))
    points_4D = torch.cat([points_phys, ones], dim=-1)  # (M,4)
    camcoords = points_4D @ lidar2img.T  # (M,4)
    eps = 1e-5
    z_ = camcoords[:,2].clamp(min=eps)
    u_ = camcoords[:,0] / z_
    v_ = camcoords[:,1] / z_
    
    H,W = img_hw
    mask = (u_>=0)&(v_>=0)&(u_<W)&(v_<H)
    return u_, v_, mask

def draw_points_with_weights(img_bgr, us, vs, weights, mask):
    """
    各pixel(u,v)に weightsでdot可視化
    """
    print("us.shape:", us.shape)
    print("vs.shape:", vs.shape)
    print("weights.shape:", weights.shape)
    print("mask.shape:", mask.shape)

    # shapeが(1,M)だった場合を想定
    if us.dim() == 2 and us.shape[0] == 1:
        us = us.squeeze(0)        # (M,) にする
        vs = vs.squeeze(0)        # (M,)
        weights = weights.squeeze(0)
        mask = mask.squeeze(0)

    print("us.shape (after):", us.shape)
    print("mask.shape (after):", mask.shape)

    vis = img_bgr.copy()
    wmin, wmax = weights.min(), weights.max()
    rng = wmax - wmin + 1e-6
    for i in range(len(us)):
        if not mask[i]:
            continue
        x_i = int(us[i].item())
        y_i = int(vs[i].item())
        if x_i<0 or y_i<0 or x_i>=vis.shape[1] or y_i>=vis.shape[0]:
            continue
        w_01 = (weights[i]-wmin)/rng
        color = (0, int(255*w_01), int(255*(1-w_01)))  # green->red
        cv2.circle(vis, (x_i,y_i), 2, color, -1)
    return vis

def visualize_bevformer_encoder_attention(
    ref_3d, # shape (1, 4, 40000, 3)
    reference_points_cam, # shape (6, 1, 40000, 4, 2))
    bev_mask, # shape (6, 1, 40000, 4)
    sampling_offsets, # shape (6, 9675, 8, 4, 2, 4, 2)
    attention_weights, # shape (6, 9675, 8, 4, 8)
    pc_range, # [x_min, y_min, z_min, x_max, y_max, z_max]
    img_metas, # list of dict. each dict has 'lidar2img', 'img_shape', etc
    info_i=None,
    cam_names=[],
    cam_out_path=[],
):
    """
    簡易例: 
    1) ref_3d -> 物理座標
    2) offsetで x,yを微妙にシフト (実際のMSDeformAttn3Dとは異なる可能性大)
    3) lidar2img投影
    4) attention_weightsでカラードット
    """
    ref_3d = ref_3d.cuda()
    sampling_offsets = sampling_offsets.cuda()
    attention_weights = attention_weights.cuda()
    pc_range = torch.tensor(pc_range, device=ref_3d.device)

    print('lets shape')
    print(ref_3d.shape)
    print(reference_points_cam.shape)
    print(bev_mask.shape)
    print(sampling_offsets.shape)
    print(attention_weights.shape)
    print(pc_range.shape)
    print(img_metas[0]['lidar2img'].shape)
    print(len(img_metas[0]['lidar2img']))


    B, D, N, _ = ref_3d.shape
    # B=1仮定
    ref_3d_flat = ref_3d[0].view(-1, 3).clone()  # => (D*N,3)
    x_min,y_min,z_min, x_max,y_max,z_max = pc_range
    ref_3d_flat[:,0] = ref_3d_flat[:,0]*(x_max - x_min)+x_min
    ref_3d_flat[:,1] = ref_3d_flat[:,1]*(y_max - y_min)+y_min
    ref_3d_flat[:,2] = ref_3d_flat[:,2]*(z_max - z_min)+z_min

    B_, N_, nH, nL, nP, _2 = sampling_offsets.shape
    sampling_offsets_flat = sampling_offsets[0].view(N_*nH*nL*nP, 2)
    attention_weights_flat = attention_weights[0].view(N_*nH*nL*nP)

    ref_3d_big = ref_3d_flat.unsqueeze(1).repeat(1, nH*nL*nP, 1)  # => (N_, nH*nL*nP, 3)
    ref_3d_big = ref_3d_big.view(-1,3)

    # flatten sampling_offsets
    # shape (B,N, nHead,nLevel,nPoint,2) => (N*nHead*nLevel*nPoint,2)
    sampling_offsets_flat = sampling_offsets[0].view(-1,2)
    attn_weights_flat = attention_weights[0].view(-1)
    
    # 例: min(...) だけ対応付け
    M = min(ref_3d_flat.size(0), sampling_offsets_flat.size(0))
    ref_3d_flat = ref_3d_flat[:M]
    dxdy = sampling_offsets_flat[:M]
    w_ = attn_weights_flat[:M]
    
    # XYにdx,dy足す(実際にはnormalizerやlevel別処理が必要)
    # ref_3d_flat[:,0] += dxdy[:,0]
    # ref_3d_flat[:,1] += dxdy[:,1]

    # カメラごとに投影
    for cam_idx, cam_meta in enumerate(img_metas):
        # カメラパラメータ
        lidar2img = torch.tensor(cam_meta['lidar2img'], dtype=torch.float32, device=ref_3d_flat.device)
        (H,W,_) = 900, 1600, 3
        u_, v_, mask = project_points_to_image(ref_3d_big, lidar2img, (H,W))
        # 画像読み込み
        img_path = info_i['cams'][cam_names[cam_idx]]['data_path']
        print(f"Reading image: {img_path}")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"cannot read {img_path}")
            continue

        # draw
        vis_img = draw_points_with_weights(img_bgr, u_, v_, w_, mask)
        out_name = os.path.basename(img_path)
        # ディレクトリが存在するかを事前にチェック
        if not os.path.exists(cam_out_path[cam_idx]):
            os.makedirs(cam_out_path[cam_idx], exist_ok=True)
            print(f"Directory created: {cam_out_path[cam_idx]}")
        else:
            print(f"Directory already exists: {cam_out_path[cam_idx]}")
        out_path = os.path.join(cam_out_path[cam_idx], out_name)
        cv2.imwrite(out_path, vis_img)

        print(f"Saved {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet を用いてモデルのテスト（および評価）を実行するためのスクリプト')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='畳み込み層（Conv）とバッチ正規化層（BN）を統合するかどうか。これにより推論速度がわずかに向上')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='評価を実行せず、結果を指定フォーマットに変換するためのオプション。 例: テストサーバーへの提出用にフォーマットを変換する際に使用')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='COCOデータセット用: "bbox", "segm", "proposal"　ASCAL VOCデータセット用: "mAP", "recall"')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    
    # 今回は　eval bbox

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # configは./projects/configs/stage2_e2e/base_e2e.py
    # cofigに書かれた内容で上書き
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                print('0')
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    # cudaの最適化を行うかどうか
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    # データセットが複数か単数かで処理を分ける
    if isinstance(cfg.data.test, dict):
        print('1')
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    # gpuの分散処理を行うかどうか
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        print('2')
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    '''
    dataloaderの出力
    Loading NuScenes tables for version v1.0-mini...
    23 category,
    8 attribute,
    4 visibility,
    911 instance,
    12 sensor,
    120 calibrated_sensor,
    31206 ego_pose,
    8 log,
    10 scene,
    404 sample,
    31206 sample_data,
    18538 sample_annotation,
    4 map,
    Done loading in 0.343 seconds.
    '''

    print('3')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # モデルの構築
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    def get_activation_hook(name):
        def hook(module, input, output):
            print(f"[HOOK CALLED] {name}")
            if name not in activations:
                activations[name] = [] 

            # module: フックを登録した層 (nn.Module)
            # input:  その層に入ってきた入力 (tuple of Tensor)
            # output: その層から出力される出力 (Tensor or tuple of Tensor)

            x = input[0] if isinstance(input, (list, tuple)) else input
            if isinstance(x, torch.Tensor):
                print(f"[{name}] input shape: {x.shape}")
            else:
                print(f"[{name}] input is a {type(x)}")

            # 出力に対して
            if isinstance(output, torch.Tensor):
                # 通常のTensorの場合
                print(f"[{name}] output shape: {output.shape}")
            elif isinstance(output, (list, tuple)):
                # タプルやリストなら要素ごとに
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        print(f"[{name}] output[{i}] shape: {out.shape}")
                    else:
                        print(f"[{name}] output[{i}] is a {type(out)}")
            else:
                print(f"[{name}] output is a {type(output)}")
            
            activations[name].append(output.detach().cpu())
            print(f'activation shape: {output.shape}')
            print(f'activation shape: {activations[name][-1].shape}')
        return hook
    # print('model:', model)
    # TemporalSelfAttentionとSpatialCrossAttentionを対象にフックを登録

    # !!!!!!!個々変更
    valid_names = [
        'img_backbone.layer4.2'
    ]

    #!!!!! bev attn変更９
    return_ref_cam_mask = False

    for name, module in model.named_modules():
        # print(f'Registering hook for {name}')
        if name in valid_names:
            print(f'Registering hook for {name}')
            module.register_forward_hook(get_activation_hook(name))

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # チェックポイントにクラス情報が含まれていない場合、データセットのクラス情報を使用する
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    print('4')

    # モデルをGPUに転送
    if not distributed:
        assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        print('5')
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        # !!!!!ここをmy_にするとheatmap取ってこれる
        # outputs = my_custom_multi_gpu_test(model, data_loader, args.tmpdir,
        #                                 args.gpu_collect, return_ref_cam_mask=return_ref_cam_mask)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)
        print('iiiiiiiiiiiiiii')

    rank, _ = get_dist_info()

    pkl_path = '/home/yoshi-22/FusionAD/UniAD/data/infos/nuscenes_infos_temporal_val.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    infos = data['infos']

    cam_out_path = [
        '/home/yoshi-22/UniAD/outputs/CAM_FRONT',
        '/home/yoshi-22/UniAD/outputs/CAM_FRONT_RIGHT',
        '/home/yoshi-22/UniAD/outputs/CAM_FRONT_LEFT',
        '/home/yoshi-22/UniAD/outputs/CAM_BACK',
        '/home/yoshi-22/UniAD/outputs/CAM_BACK_LEFT',
        '/home/yoshi-22/UniAD/outputs/CAM_BACK_RIGHT'
    ]

    cam_names = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]


    # !!!!!!!個々変更
    # name = 'img_backbone.layer4.2'
    # my_activations = outputs['my_activations']

    # for i in sorted(my_activations[name].keys()):
    #     if 79 <= i <= 119:
    #         feat_i = my_activations[name][i + 41]
    #         print(f"Processing frame {i, i + 41}")
    #     elif 120 <= i <= 160:
    #         feat_i = my_activations[name][i - 41]
    #         print(f"Processing frame {i, i -41}")
    #     else:
    #         feat_i = my_activations[name][i]
    #         print(f"Processing frame {i}")

    #     feat_i = my_activations[name][i]
    #     print(f"Processing frame {i}")
    #     info_i = infos[i]

        # visualize_bevformer_encoder_attention(
        #     outputs['bbox_results'][i]['ref_3d'],
        #     outputs['bbox_results'][i]['reference_points_cam'],
        #     outputs['bbox_results'][i]['bev_mask'],
        #     outputs['bbox_results'][i]['offsets'],
        #     outputs['bbox_results'][i]['weights'],
        #     outputs['bbox_results'][i]['pc_range'],
        #     outputs['bbox_results'][i]['img_metas2'],
        #     infos[i],
        #     cam_names=cam_names,
        #     cam_out_path=cam_out_path
        # )

        # for cam_idx in range(6):
        #     single_feat_map = feat_i[cam_idx]
        #     # カメラごとのファイル名に cam_names[cam_idx] を入れる
        #     heatmap_bgr = tensor_to_heatmap(single_feat_map)

        #     # 入力画像のパス
        #     img_path = info_i['cams'][cam_names[cam_idx]]['data_path']
        #     print(f"Reading image: {img_path}")
        #     image_bgr = cv2.imread(img_path)
        #     if image_bgr is None:
        #         print(f"Failed to read image: {img_path}")
        #         continue

        #     # feat[cam_idx] → (C, H, W)
        #     overlaid = overlay_heatmap_on_image(image_bgr, heatmap_bgr, alpha=0.5)

        #     # 出力ファイル名
        #     out_name = os.path.basename(img_path)
        #     # ディレクトリが存在するかを事前にチェック
        #     if not os.path.exists(cam_out_path[cam_idx]):
        #         os.makedirs(cam_out_path[cam_idx], exist_ok=True)
        #         print(f"Directory created: {cam_out_path[cam_idx]}")
        #     else:
        #         print(f"Directory already exists: {cam_out_path[cam_idx]}")
        #     out_path = os.path.join(cam_out_path[cam_idx], out_name)
        #     cv2.imwrite(out_path, overlaid)
        #     print(f"Heatmap saved to {out_path}")

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            #assert False
            mmcv.dump(outputs, args.out)
            #outputs = mmcv.load(args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)            

        if args.eval:
            print('5555555')
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            print('7777777')
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print('8888888')

            print(dataset.evaluate(outputs, **eval_kwargs))

# mmdetection3dのテストスクリプト
# 指定したデータ・セットで推論を行い、結果を保存、評価、表示する
if __name__ == '__main__':
    main()
