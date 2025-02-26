
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon


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


def compute_distribution_metrics(p, q, eps=1e-12):
    """
    2つの2次元マップ p, q (shape=(H,W)) を、それぞれ [0,1] の「確率分布」とみなして
    - Pearson相関
    - Spearman相関
    - KLダイバージェンス(KL(p||q), KL(q||p) も計算)
    - Jensen-Shannonダイバージェンス
    をまとめて返す。
    """
    # この関数自体は使わなくてもよいが、元コードの名残として残しておきます
    # （不要なら削除可）
    p = p.flatten().astype(float)
    q = q.flatten().astype(float)

    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    sp = p.sum()
    sq = q.sum()
    if sp > 0:
        p /= sp
    if sq > 0:
        q /= sq

    def safe_pearsonr(a, b):
        if (np.std(a) < 1e-15) or (np.std(b) < 1e-15):
            return 0.0
        return pearsonr(a, b)[0]

    def safe_spearmanr(a, b):
        if (np.std(a) < 1e-15) or (np.std(b) < 1e-15):
            return 0.0
        return spearmanr(a, b).correlation

    pearson_val  = safe_pearsonr(p, q)
    spearman_val = safe_spearmanr(p, q)

    p_ = p + eps
    q_ = q + eps
    kl_pq = np.sum( p_ * np.log( p_/q_ ) )
    kl_qp = np.sum( q_ * np.log( q_/p_ ) )

    js_dist = jensenshannon(p_, q_)
    jsd = js_dist**2
    return {
        'pearson':   pearson_val,
        'spearman':  spearman_val,
        'kl_pq':     kl_pq,
        'kl_qp':     kl_qp,
        'jsd':       jsd
    }


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
        if 'cross_attn_list' not in entry:
            token_to_attn[sample_token] = None
            continue
        
        cross_attn = entry['cross_attn_list']
        if len(cross_attn) > 0:
            attn_mask = cross_attn[0].squeeze().reshape(200, 200)
            attn_mask = attn_mask.cpu().numpy() if hasattr(attn_mask, 'cpu') else attn_mask
        else:
            attn_mask = None
        
        token_to_attn[sample_token] = attn_mask
    
    return token_to_attn

def compare_planner_attn_index(
    pkl_normal,
    pkl_raindrop,
    out_dir='attn_compare_output',
    vmax=0.2,
    thr=0.2
):
    print(f"--- Loading normal pkl: {pkl_normal} ---")
    attn_normal = load_planner_attn(pkl_normal)
    print(f"--- Loading raindrop pkl: {pkl_raindrop} ---")
    attn_raindrop = load_planner_attn(pkl_raindrop)
    
    # ◆◆◆ 修正ポイント ◆◆◆
    # 「ソートせず」、attn_normal に入っている順序のままキーを取り出し、
    # かつ raindrop 側にも含まれるものだけを抽出
    #   ※ Python 3.7+ では dict の「挿入順序」が保持されます
    common_tokens = [k for k in attn_normal if k in attn_raindrop]

    os.makedirs(out_dir, exist_ok=True)
    total_pixels = 200 * 200

    result_list = []

    for i, token in enumerate(common_tokens):
        mask_n = attn_normal[token]
        mask_r = attn_raindrop[token]
        
        if mask_n is None or mask_r is None:
            print(f"frame_key={i}, token={token}: attn_mask が None のためスキップ")
            continue
        
        # (1) 正規化
        mask_n_norm = mask_n / mask_n.max() if mask_n.max() > 1e-12 else mask_n
        mask_r_norm = mask_r / mask_r.max() if mask_r.max() > 1e-12 else mask_r
        
        # (2) エントロピーを計算
        entropy_n = compute_attention_entropy(mask_n_norm)
        entropy_r = compute_attention_entropy(mask_r_norm)

        print(f"[frame_key={i:03d}] Token={token} -> Normal Entropy={entropy_n:.3f}, Rain Entropy={entropy_r:.3f}")

        # (画像出力が必要なら残す)
        diff = mask_r_norm - mask_n_norm
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        im_n = axes[0].imshow(
            mask_n_norm, vmin=0, vmax=vmax, cmap='plasma', origin='lower'
        )
        axes[0].set_title(f"Normal Ent={entropy_n:.2f}")
        plt.colorbar(im_n, ax=axes[0], fraction=0.046, pad=0.04)

        im_r = axes[1].imshow(
            mask_r_norm, vmin=0, vmax=vmax, cmap='plasma', origin='lower'
        )
        axes[1].set_title(f"Rain Ent={entropy_r:.2f}")
        plt.colorbar(im_r, ax=axes[1], fraction=0.046, pad=0.04)

        im_d = axes[2].imshow(diff, cmap='bwr', origin='lower')
        axes[2].set_title("Rain - Normal")
        plt.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_name = f"{str(i).zfill(3)}_{token}_compare.png"
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        
        # CSV用のデータ (最小限のみ)
        result_item = {
            'frame_key': i,   # 連番でOK; 必要なら token を frame_key にしてもよい
            'entropy_normal': entropy_n,
            'entropy_rain':   entropy_r
        }
        result_list.append(result_item)

    if result_list:
        import csv
        csv_path = os.path.join(out_dir, "attn_compare_metrics_minimal.csv")
        fieldnames = ['frame_key', 'entropy_normal', 'entropy_rain']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in result_list:
                writer.writerow(row)
        print(f"--> 集計結果をCSV出力しました: {csv_path}")
    else:
        print("result_list が空でした。共通トークンがない or None が多い可能性があります。")


# -------------------------------------------
# 使い方の例
# -------------------------------------------
if __name__ == '__main__':
    pkl_normal    = './output/results_no.pkl'
    pkl_raindrop  = './output/results.pkl'
    out_dir       = 'fusionad_gogo'
    
    compare_planner_attn_index(
        pkl_normal, 
        pkl_raindrop, 
        out_dir=out_dir, 
        vmax=0.2, 
        thr=0.2
    )
