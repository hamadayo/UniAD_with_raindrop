import csv
import numpy as np

def compute_camera_dependency(csv_path, weight_jsd=0.5, weight_pearson=0.5):
    """
    複数指標 (JSD, Pearson相関) を組み合わせ、カメラ依存度 (0~1) を算出。
    ここでは簡易に「平均 JSD」と「1 - 平均Pearson」を重み付き合算する例。

    CSV想定カラム: 'jsd', 'pearson' など
    
    Parameters
    ----------
    csv_path : str
        読み込むCSVファイルのパス
    weight_jsd : float
        JSD由来の依存度に対する重み (0 ~ 1)
    weight_pearson : float
        Pearson由来の依存度に対する重み (0 ~ 1)

    Returns
    -------
    float
        0~1 の数値で表されるカメラ依存度 (1に近いほど依存度が高い)
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            rows.append(line)

    # CSVが空の場合は暫定値(0.5)を返す
    if not rows:
        print("[Warning] CSV is empty, returning 0.5")
        return 0.5

    jsd_vals = []
    pearson_vals = []
    for r in rows:
        jsd_vals.append(float(r['jsd']))
        pearson_vals.append(float(r['pearson']))
    
    # JSDとPearsonの平均値を計算
    mean_jsd     = np.mean(jsd_vals)
    mean_pearson = np.mean(pearson_vals)

    # 例: mean_jsdが大きいほど依存度大 → 0~1の変換として「1 - exp(-3*jsd)」を採用
    alpha_jsd = 1.0 - np.exp(-3.0 * mean_jsd)
    # クリップ (0~1)
    alpha_jsd = min(max(alpha_jsd, 0.0), 1.0)

    # 例: Pearsonが大きいほど依存度小 → 「1 - mean_pearson」を採用
    alpha_p = 1.0 - mean_pearson
    # クリップ (0~1)
    alpha_p = min(max(alpha_p, 0.0), 1.0)

    # 重み付き合算して最終的なカメラ依存度を計算
    camera_dep = weight_jsd * alpha_jsd + weight_pearson * alpha_p
    # クリップ (0~1)
    camera_dep = min(max(camera_dep, 0.0), 1.0)

    return camera_dep

if __name__ == "__main__":
    # 使用例: 自分のCSVファイルへのパスを指定してください
    # csv_file_path = "/home/yoshi-22/FusionAD/UniAD/fusionad_graph/attn_compare_metrics.csv"
    csv_file_path = "/home/yoshi-22/FusionAD/UniAD/uniad_graph/attn_compare_metrics_uniad.csv"

    
    # 関数を呼び出して結果を表示
    camera_dependency_score = compute_camera_dependency(
        csv_path=csv_file_path,
        weight_jsd=0.5,
        weight_pearson=0.5
    )

    print(f"Camera Dependency: {camera_dependency_score:.4f}")
