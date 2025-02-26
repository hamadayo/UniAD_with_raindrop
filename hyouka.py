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
    をまとめて返す。辞書型で { 'pearson':..., 'spearman':..., 'kl_pq':..., 'kl_qp':..., 'jsd':... }
    を返却する。

    注意:
      KL や JS は p,q を 1Dに flatten して (sum=1) の確率分布として計算する。
      0割り回避に eps を加える。
    """
    # 1) flatten + float変換
    p = p.flatten().astype(float)
    q = q.flatten().astype(float)

    # 2) それぞれ min/max などで負値を排除し、さらに正規化
    #    ただし既に 0~1 に正規化されていればスキップ可。念のため安全策として絶対値クリップしておく。
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)

    sp = p.sum()
    sq = q.sum()
    if sp > 0:
        p /= sp
    if sq > 0:
        q /= sq

    # -- Pearson / Spearman ---
    #   p, q は [0, 1] の分布だが、相関用にそのまま使ってOK
    #   ただし定数ベクトルの場合はエラーになるのでチェック
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

    # -- KLダイバージェンス --
    # KL(p||q) = ∑ p_i * log( p_i / q_i )
    # 0割り回避に eps を足す
    p_ = p + eps
    q_ = q + eps
    kl_pq = np.sum( p_ * np.log( p_/q_ ) )
    kl_qp = np.sum( q_ * np.log( q_/p_ ) )

    # -- Jensen-Shannonダイバージェンス --
    # JS(p||q) = 0.5 KL(p||m) + 0.5 KL(q||m), m=(p+q)/2
    # scipy の jensenshannon() は 距離。JSD = JSdistance^2
    # ただし jensenshannon は sqrt( JS divergence ) が返る仕様なので注意。
    js_dist = jensenshannon(p_, q_)/1.0
    jsd = js_dist**2  # divergeceにするには2乗

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

    ※ UniAD / ST-P3 / FusionAD 等のモデル実装によって
      pkl内部のキー構造が異なる可能性に注意。
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


def compare_planner_attn_index(
    pkl_normal,
    pkl_raindrop,
    out_dir='attn_compare_output',
    vmax=0.2,
    thr=0.2
):
    """
    2つの pkl について、同一サンプルトークンごとに attn_mask (200,200) を比較可視化する。
    - 雨なし vs 雨あり
    - しきい値 (thr) 以上の画素数の差分をコンソール表示
    - Pearson/Spearman相関、KL, JSD などの分布指標を計算
    - 画像は out_dir に「00_compare.png」などの形で保存。

    Args:
        pkl_normal (str): ノーマル環境(雨無し)の result.pkl のパス
        pkl_raindrop (str): 雨滴環境の result.pkl のパス
        out_dir (str): 結果画像の保存先ディレクトリ
        vmax (float): imshow の最大値 (視覚的にクリップしたい場合は調整)
        thr (float): しきい値
    """
    # PlannerのAttentionを読み込み
    print(f"--- Loading normal pkl: {pkl_normal} ---")
    attn_normal = load_planner_attn(pkl_normal)
    print(f"--- Loading raindrop pkl: {pkl_raindrop} ---")
    attn_raindrop = load_planner_attn(pkl_raindrop)
    
    # 両方に含まれるサンプルトークンの共通集合を取り出してソート
    common_tokens = sorted(attn_normal.keys() & attn_raindrop.keys())
    
    os.makedirs(out_dir, exist_ok=True)
    total_pixels = 200 * 200

    # 結果集計用のリスト
    result_list = []

    for i, token in enumerate(common_tokens):
        mask_n = attn_normal[token]
        mask_r = attn_raindrop[token]
        
        # いずれかが None ならばスキップ
        if mask_n is None or mask_r is None:
            print(f"Index={i}, token={token}: attn_mask が None のためスキップ")
            continue
        
        # (1) 正規化 (例: maxで割る)
        mask_n_norm = mask_n / mask_n.max() if mask_n.max() > 1e-12 else mask_n
        mask_r_norm = mask_r / mask_r.max() if mask_r.max() > 1e-12 else mask_r
        
        # (2) エントロピーを計算
        entropy_n = compute_attention_entropy(mask_n_norm)
        entropy_r = compute_attention_entropy(mask_r_norm)

        # (3) しきい値以上の画素数
        count_n = np.sum(mask_n_norm >= thr)
        count_r = np.sum(mask_r_norm >= thr)
        ratio_n = (count_n / total_pixels) * 100
        ratio_r = (count_r / total_pixels) * 100
        diff_count = count_r - count_n
        diff_ratio = ratio_r - ratio_n

        # (4) 分布の相関・ダイバージェンス
        dist_metrics = compute_distribution_metrics(mask_n_norm, mask_r_norm)
        # 例: dist_metrics = {
        #     'pearson': 0.99,
        #     'spearman': 0.98,
        #     'kl_pq': 0.10,
        #     'kl_qp': 0.12,
        #     'jsd': 0.05
        # }

        # コンソールに出力
        print(f"[Idx={i:03d}] Token={token[:8]}...  Thr={thr:.2f}")
        print(f"  - Entropy(normal)={entropy_n:.3f}, Entropy(rain)={entropy_r:.3f}")
        print(f"  - >=thr: normal={count_n} px({ratio_n:.2f}%), rain={count_r} px({ratio_r:.2f}%), diff={diff_count} px({diff_ratio:.2f}%)")
        print("  - DistMetrics: " +
              f"pearson={dist_metrics['pearson']:.3f}, spearman={dist_metrics['spearman']:.3f}, "
              f"KL(p||q)={dist_metrics['kl_pq']:.3f}, KL(q||p)={dist_metrics['kl_qp']:.3f}, JSD={dist_metrics['jsd']:.3f}\n")

        # (5) 可視化
        diff = mask_r_norm - mask_n_norm
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        im_n = axes[0].imshow(
            mask_n_norm, 
            extent=(-51.2, 51.2, -51.2, 51.2),
            vmin=0, vmax=vmax, 
            cmap='plasma',
            origin='lower'
        )
        axes[0].set_title(f"[{i}] Normal\nEntropy={entropy_n:.2f}")
        plt.colorbar(im_n, ax=axes[0], fraction=0.046, pad=0.04)

        im_r = axes[1].imshow(
            mask_r_norm, 
            extent=(-51.2, 51.2, -51.2, 51.2),
            vmin=0, vmax=vmax, 
            cmap='plasma',
            origin='lower'
        )
        axes[1].set_title(f"Rain\nEntropy={entropy_r:.2f}")
        plt.colorbar(im_r, ax=axes[1], fraction=0.046, pad=0.04)

        im_d = axes[2].imshow(
            diff,
            extent=(-51.2, 51.2, -51.2, 51.2),
            cmap='bwr',
            origin='lower'
        )
        axes[2].set_title("Diff(Rain - Normal)")
        plt.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_name = f"{str(i).zfill(3)}_{token}_compare.png"
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        # CSV等にまとめられるよう、リストに格納
        result_item = {
            'index': i,
            'token': token,
            'entropy_normal': entropy_n,
            'entropy_rain':   entropy_r,
            'count_normal':   count_n,
            'count_rain':     count_r,
            'diff_count':     diff_count,
            'pearson':        dist_metrics['pearson'],
            'spearman':       dist_metrics['spearman'],
            'kl_pq':          dist_metrics['kl_pq'],
            'kl_qp':          dist_metrics['kl_qp'],
            'jsd':            dist_metrics['jsd']
        }
        result_list.append(result_item)

    # まとめて CSV に保存
    if result_list:
        import csv
        csv_path = os.path.join(out_dir, "attn_compare_metrics.csv")
        fieldnames = list(result_list[0].keys())
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
    pkl_normal    = '../output/results_no.pkl'   # 雨無し
    pkl_raindrop  = '../output/results.pkl'   # 雨滴あり
    out_dir       = 'attn_compare_output2'
    
    compare_planner_attn_index(pkl_normal, pkl_raindrop, out_dir=out_dir, vmax=0.2, thr=0.2)
