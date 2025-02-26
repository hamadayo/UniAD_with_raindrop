import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_attention_entropy(attn_map):
    """
    Attention マップ (200,200) から Shannon エントロピーを計算する。
    1) Flatten
    2) 合計値で割って確率分布に
    3) -∑ p log p
    """
    attn_flat = attn_map.flatten().astype(float)  # shape=(40000,)

    total = attn_flat.sum()
    if total == 0:
        return 0.0

    p = attn_flat / total
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    return entropy

def load_planner_attn(result_pkl_path):
    """
    指定した result.pkl から Planner の attn_mask（cross_attn）をサンプルトークンごとに
    取り出して辞書として返す簡易例。
    
    戻り値:
        { sample_token(str): attn_mask(ndarray or None), ... }
    """
    outputs = mmcv.load(result_pkl_path)
    outputs = outputs['bbox_results']
    token_to_attn = dict()
    
    for entry in outputs:
        sample_token = entry['token']
        # Planner 用の cross_attn_list が存在しない場合は None とする
        if 'cross_attn_list' not in entry:
            token_to_attn[sample_token] = None
            continue
        
        cross_attn = entry['cross_attn_list']
        # cross_attn_list が空（[]）の場合もあるかもしれない
        if len(cross_attn) > 0:
            # shape (1, 1, 200, 200) などを想定して reshape
            attn_mask = cross_attn[0].squeeze().reshape(200, 200)
            # torch.Tensor の場合があるので、cpu().numpy() で numpy に変換
            attn_mask = attn_mask.cpu().numpy() if hasattr(attn_mask, 'cpu') else attn_mask
        else:
            attn_mask = None
        
        token_to_attn[sample_token] = attn_mask
    
    return token_to_attn

def plot_entropy_comparison(normal_entropies, raindrop_entropies, out_dir, file_name="entropy_plot.png"):
    """
    2つのリスト (normal_entropies, raindrop_entropies) を比較プロットする。
    インデックスを x 軸とし、10 ステップ刻みに目盛りを表示。
    """
    # x 軸は 0 ~ N-1
    x_vals = np.arange(len(normal_entropies))
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, normal_entropies, label="Normal Entropy")
    plt.plot(x_vals, raindrop_entropies, label="Raindrop Entropy")

    # グリッド
    plt.grid(True)

    # 10刻みでラベル表示したい場合
    x_ticks = np.arange(0, len(x_vals), 10)
    plt.xticks(x_ticks, x_ticks)

    plt.xlabel("Index")
    plt.ylabel("Shannon Entropy")
    plt.title("Comparison of Shannon Entropy (Normal vs. Raindrop)")
    plt.legend()
    plt.tight_layout()

    # 保存
    out_path = os.path.join(out_dir, file_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Shannon Entropy を比較したプロットを保存しました: {out_path}")


def compare_planner_attn_index(
    pkl_normal,
    pkl_raindrop,
    out_dir='comparison_results',
    vmax=0.2,
    thr=0.2,
    save_csv=True
):
    """
    2つの pkl について、同一サンプルトークンごとに attn_mask を比較可視化する。
    - サンプルトークンをソートし、インデックス (0 ~) で処理。
    - 画像は out_dir に「000_compare.png」などの形で保存。
    - 晴れ/雨のattentionマップで「しきい値以上の画素数の差分」をフレームごとにprint。
    - Shannonエントロピーを計算して出力し、グラフ化して比較(CSVへの出力も可)。

    Args:
        pkl_normal (str): ノーマル環境の result.pkl のパス
        pkl_raindrop (str): 雨滴環境の result.pkl のパス
        out_dir (str): 結果画像の保存先ディレクトリ
        vmax (float): imshow の最大値 (視覚的にクリップしたい場合は調整)
        thr (float): しきい値 (>= thr の画素数をカウント)
        save_csv (bool): True なら CSV (comparison_stats.csv) に結果を出力
    """
    print(f"Normal: {pkl_normal}")
    attn_normal = load_planner_attn(pkl_normal)
    print(f"Rain: {pkl_raindrop}")
    attn_raindrop = load_planner_attn(pkl_raindrop)
    
    # どちらにも含まれるサンプルトークンの共通集合を取得してソート
    common_tokens = sorted(attn_normal.keys() & attn_raindrop.keys())
    
    # 保存先フォルダを作成
    os.makedirs(out_dir, exist_ok=True)

    # CSV の準備
    csv_path = os.path.join(out_dir, "comparison_stats.csv")
    f_csv = None
    if save_csv:
        f_csv = open(csv_path, 'w', encoding='utf-8')
        # ヘッダ行
        f_csv.write("index,token,normal_entropy,rain_entropy,normal_pix,normal_ratio,rain_pix,rain_ratio,diff_pix,diff_ratio\n")

    # ピクセル総数（200 x 200）
    total_pixels = 200 * 200

    # エントロピーを図示するためのリスト
    normal_entropies = []
    raindrop_entropies = []
    
    for i, token in enumerate(common_tokens):
        mask_n = attn_normal[token]
        mask_r = attn_raindrop[token]
        
        # いずれかが None ならばスキップ
        if mask_n is None or mask_r is None:
            print(f"Index={i}, token={token}: attn_mask が None のためスキップ")
            continue
        
        # エントロピー計算
        entropy_n = compute_attention_entropy(mask_n)
        entropy_r = compute_attention_entropy(mask_r)
        
        normal_entropies.append(entropy_n)
        raindrop_entropies.append(entropy_r)

        # アテンションマップを最大値で正規化（可視化・しきい値用）
        mask_n_norm = mask_n / mask_n.max()
        mask_r_norm = mask_r / mask_r.max()
        
        # しきい値以上の画素数
        count_n = np.sum(mask_n_norm >= thr)
        count_r = np.sum(mask_r_norm >= thr)
        
        ratio_n = (count_n / total_pixels) * 100
        ratio_r = (count_r / total_pixels) * 100
        
        diff_count = count_r - count_n
        diff_ratio = ratio_r - ratio_n
        
        # ターミナル表示
        print(f"[Index={i:03d}] Token={token[:8]}... | Thr={thr:.2f}")
        print(f"  - Normal Entropy:   {entropy_n:.4f}")
        print(f"  - Raindrop Entropy: {entropy_r:.4f}")
        print(f"  - Normal >= thr:    {count_n} pixels ({ratio_n:.2f}%)")
        print(f"  - Raindrop >= thr:  {count_r} pixels ({ratio_r:.2f}%)")
        print(f"  - Diff (rain - normal) = {diff_count} pixels ({diff_ratio:.2f}%)\n")
        
        # CSV 書き込み
        if f_csv:
            f_csv.write(
                f"{i},{token},{entropy_n:.4f},{entropy_r:.4f},{count_n},{ratio_n:.4f},{count_r},{ratio_r:.4f},{diff_count},{diff_ratio:.4f}\n"
            )

        # 差分 (雨滴 - ノーマル)
        diff = mask_r_norm - mask_n_norm
        
        # 可視化用プロット
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Normal
        im_n = axes[0].imshow(
            mask_n_norm,
            extent=(-51.2, 51.2, -51.2, 51.2),
            vmin=0, vmax=vmax,
            cmap='plasma',
            origin='lower'
        )
        axes[0].set_title(f'Index={i}\nNormal\n{token[:8]}...')
        plt.colorbar(im_n, ax=axes[0], fraction=0.046, pad=0.04)

        # Raindrop
        im_r = axes[1].imshow(
            mask_r_norm,
            extent=(-51.2, 51.2, -51.2, 51.2),
            vmin=0, vmax=vmax,
            cmap='plasma',
            origin='lower'
        )
        axes[1].set_title(f'Raindrop\n{token[:8]}...')
        plt.colorbar(im_r, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Diff
        im_d = axes[2].imshow(
            diff,
            extent=(-51.2, 51.2, -51.2, 51.2),
            cmap='bwr',
            origin='lower',
            vmin=-1.0,
            vmax=1.0
        )
        axes[2].set_title('Diff: (Raindrop - Normal)')
        plt.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 画像を保存
        out_path = os.path.join(out_dir, f"{str(i).zfill(3)}_compare.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    
    # CSV を閉じる
    if f_csv:
        f_csv.close()

    # ============================
    # エントロピー比較用のグラフ描画
    # ============================
    if len(normal_entropies) > 0:
        plot_entropy_comparison(
            normal_entropies,
            raindrop_entropies,
            out_dir=out_dir,
            file_name="entropy_plot.png"
        )
    
    print(f"可視化＆集計完了しました。結果は '{out_dir}' に保存されています。")
    if save_csv:
        print(f"集計結果 CSV: {csv_path}")


# -------------------------------------------
# 使い方の例 (このファイルを直接実行する場合)
# -------------------------------------------
if __name__ == '__main__':
    pkl_normal    = '../output/results_no.pkl'   # 晴れ(あるいは雨滴なし)
    pkl_raindrop  = '../output/results.pkl'      # 雨
    out_dir       = 'attn_compare_output2'
    
    compare_planner_attn_index(
        pkl_normal,
        pkl_raindrop,
        out_dir=out_dir,
        vmax=0.2,
        thr=0.2,
        save_csv=True  # CSV にも残したい場合は True
    )
