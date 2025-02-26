import mmcv
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def ensure_dir(path):
    """指定したパスが存在しない場合、ディレクトリを再帰的に作成する"""
    os.makedirs(path, exist_ok=True)

def tensor_to_heatmap(feature_map: torch.Tensor) -> np.ndarray:
    """
    feature_map: 形状が (C, H, W) のTensor (1枚のカメラ分)を想定
    戻り値: ヒートマップ（カラー）のBGR画像 (H, W, 3)
    """
    heatmap = feature_map.mean(dim=0)  # -> shape: (H, W)
    heatmap = heatmap.detach().cpu().numpy()
    # 以下、0-1正規化
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def compute_shannon_entropy_2d(array_2d: np.ndarray) -> float:
    """
    エントロピーを計算する
    1) 負を含む可能性があるので一旦 [0,1] スケーリングを施す
    2) flatten -> 合計値で割る -> p=確率 -> -∑ p log p
    """
    # まずコピー（元データを変えない）
    val = array_2d.copy().astype(float)

    # min/max で [0, 1] に正規化
    val -= val.min()
    vmax = val.max()
    if vmax > 0:
        val /= vmax
    
    # 確率分布化
    total = val.sum()
    if total == 0:
        return 0.0
    p = val.flatten() / total
    
    # p>0 の部分のみで計算
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    return float(entropy)

def compare_and_save_heatmaps(
    pkl_path1,
    pkl_path2,
    out_dir='compare_heatmaps',
    target_layer='img_backbone.layer4.2',
    make_diff_map=True,
    scale=4
):
    """
    2つのpklファイル (pkl_path1, pkl_path2) の 'my_activations' を読み込み、
    指定した層 (target_layer) の特徴マップをヒートマップ画像として出力する
    加えて、フレームごと6台カメラの平均エントロピーを計算し、最終的に図示する

    Args:
        pkl_path1 (str): 1つ目のpklファイル (例: 晴れ)
        pkl_path2 (str): 2つ目のpklファイル (例: 雨)
        out_dir (str): 出力先ディレクトリ
        target_layer (str): 比較対象の層
        make_diff_map (bool): Trueなら、2つの差分ヒートマップも作成する
        scale (int): ヒートマップ画像の拡大率 (1なら拡大なし)
    """
    data1 = mmcv.load(pkl_path1)
    data2 = mmcv.load(pkl_path2)

    my_acts1 = data1['my_activations']
    my_acts2 = data2['my_activations']

    if target_layer not in my_acts1:
        raise KeyError(f"'{target_layer}' is not found in {pkl_path1}")
    if target_layer not in my_acts2:
        raise KeyError(f"'{target_layer}' is not found in {pkl_path2}")

    keys1 = sorted(my_acts1[target_layer].keys())
    keys2 = sorted(my_acts2[target_layer].keys())
    common_keys = sorted(set(keys1).intersection(set(keys2)))

    if not common_keys:
        print("[Warning] 共通キーがありません")
        return

    ensure_dir(out_dir)
    dir1 = os.path.join(out_dir, 'pkl1_heatmaps')
    dir2 = os.path.join(out_dir, 'pkl2_heatmaps')
    dir_diff = os.path.join(out_dir, 'diff_heatmaps')

    ensure_dir(dir1)
    ensure_dir(dir2)
    if make_diff_map:
        ensure_dir(dir_diff)

    # フレームごとのエントロピー格納リスト
    # （pkl1用、pkl2用。それぞれフレームインデックス順に保存）
    pkl1_frame_entropy = []
    pkl2_frame_entropy = []
    frame_list = []

    for i in common_keys:
        feat_list1 = my_acts1[target_layer][i]
        feat_list2 = my_acts2[target_layer][i]

        if len(feat_list1) != len(feat_list2):
            print(f"[Warning] Frame {i} で、カメラ数が異なります")
            continue

        num_cams = len(feat_list1)
        
        # このフレームのカメラ別エントロピー
        entropies_cam_pkl1 = []
        entropies_cam_pkl2 = []

        for cam_idx in range(num_cams):
            ft1 = feat_list1[cam_idx]  # shape (C,H,W)
            ft2 = feat_list2[cam_idx]  # shape (C,H,W)

            # ---------- ヒートマップ画像の保存部分 ----------
            hm1 = tensor_to_heatmap(ft1)
            hm2 = tensor_to_heatmap(ft2)

            if scale != 1:
                hm1 = cv2.resize(hm1, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                hm2 = cv2.resize(hm2, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            out_path1 = os.path.join(dir1, f"frame_{i}_cam{cam_idx}.png")
            out_path2 = os.path.join(dir2, f"frame_{i}_cam{cam_idx}.png")

            cv2.imwrite(out_path1, hm1)
            cv2.imwrite(out_path2, hm2)

            if make_diff_map:
                diff_tensor = (ft1 - ft2).abs()
                diff_hm = tensor_to_heatmap(diff_tensor)

                if scale != 1:
                    diff_hm = cv2.resize(diff_hm, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

                out_path_diff = os.path.join(dir_diff, f"frame_{i}_cam{cam_idx}_diff.png")
                cv2.imwrite(out_path_diff, diff_hm)
            # ---------------------------------------------

            # ---------- エントロピー計算部分 ----------
            ft1_mean = ft1.mean(dim=0).cpu().numpy()  # shape: (H,W)
            entropy1 = compute_shannon_entropy_2d(ft1_mean)
            entropies_cam_pkl1.append(entropy1)

            ft2_mean = ft2.mean(dim=0).cpu().numpy()  # shape: (H,W)
            entropy2 = compute_shannon_entropy_2d(ft2_mean)
            entropies_cam_pkl2.append(entropy2)

        avg_entropy_pkl1 = float(np.mean(entropies_cam_pkl1))
        avg_entropy_pkl2 = float(np.mean(entropies_cam_pkl2))

        pkl1_frame_entropy.append(avg_entropy_pkl1)
        pkl2_frame_entropy.append(avg_entropy_pkl2)
        frame_list.append(i)

    print(f"Done! Heatmaps are saved in {out_dir}")

    # ------------------
    # エントロピーのプロット
    # ------------------
    if len(frame_list) == 0:
        print("共通フレームが無いため、エントロピーの図示をスキップします")
        return
    
    x_vals = np.arange(len(frame_list))

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, pkl1_frame_entropy, marker='', label='pkl1 Entropy')
    plt.plot(x_vals, pkl2_frame_entropy, marker='', label='pkl2 Entropy')

    # グリッド表示
    plt.grid(True)

    # 横軸を10刻みにしたい場合
    x_ticks = np.arange(0, len(frame_list), 10)
    # frame_listはインデックス順にソートしているので、10刻みのインデックスに対応するフレームキーを並べる
    plt.xticks(x_ticks, [frame_list[i] for i in x_ticks], rotation=45)

    plt.xlabel("Frame Key")
    plt.ylabel("Entropy")
    plt.title("Average Entropy per Frame (6 cameras)")
    plt.legend()
    plt.tight_layout()

    out_plot = os.path.join(out_dir, "entropy_plot.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"フレームごとの平均エントロピーを '{out_plot}' にプロットしました")

    # ------------------
    # エントロピー数値を csvなどで保存
    # ------------------
    out_csv = os.path.join(out_dir, "frame_entropy_values_uniad.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("frame_key,pkl1_entropy,pkl2_entropy\n")
        for i, frame_key in enumerate(frame_list):
            f.write(f"{frame_key},{pkl1_frame_entropy[i]},{pkl2_frame_entropy[i]}\n")
    print(f"エントロピー数値を '{out_csv}' に保存しました")


if __name__ == "__main__":
    pkl_normal   = './output/results_no.pkl'
    pkl_raindrop = './output/results.pkl'
    output_dir   = "./compare_heatmaps_output"
    layer_name   = "img_backbone.layer4.2"

    compare_and_save_heatmaps(
        pkl_path1=pkl_normal,
        pkl_path2=pkl_raindrop,
        out_dir=output_dir,
        target_layer=layer_name,
        make_diff_map=True,
        scale=4
    )
