import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import matplotlib

def analyze_backbone_planner(
    backbone_csv,
    planner_csv,
    out_dir='results_correlation'
):
    """
    (1) BackboneのCSV (frame_key, entropy_normal, entropy_rain, ...)
    (2) PlannerのCSV  (index,      entropy_normal, entropy_rain, ...)
    を読み込み、frame_key=index で内部結合。
    その後、雨なし・雨ありそれぞれについて、
     - バックボーン vs. プランナーのエントロピー相関(Pearson, Spearman)
     - 散布図の保存
    を行う。
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------
    #  CSV読み込み
    # ---------------------------
    df_bb = pd.read_csv(backbone_csv)
    df_pl = pd.read_csv(planner_csv)

    # バックボーンの CSV 列名を変更 (区別するため)
    df_bb.rename(columns={
        'entropy_normal': 'bb_entropy_normal',
        'entropy_rain':   'bb_entropy_rain'
    }, inplace=True)

    # プランナーの CSV 列名を変更
    # 'index' を 'frame_key' に合わせる
    df_pl.rename(columns={
        'index':          'frame_key',
        'entropy_normal': 'pl_entropy_normal',
        'entropy_rain':   'pl_entropy_rain'
    }, inplace=True)

    # ---------------------------
    #  内部結合
    # ---------------------------
    # frame_key (バックボーン側) と frame_key (プランナー側) で join
    df_merged = pd.merge(df_bb, df_pl, on='frame_key', how='inner')

    # ちゃんとマージできたかサイズ確認
    print(f"Backbone CSV size: {len(df_bb)}  Planner CSV size: {len(df_pl)}")
    print(f"--> Merged size: {len(df_merged)} (inner join)")

    # ---------------------------
    #  雨なし vs 雨なし の相関
    # ---------------------------
    x_normal = df_merged['bb_entropy_normal']
    y_normal = df_merged['pl_entropy_normal']

    pearson_val_n, p_val_n   = pearsonr(x_normal, y_normal)
    spearman_val_n, sp_val_n = spearmanr(x_normal, y_normal)

    print("\n=== (no rain) Backbone vs. Planner ===")
    print(f"  - Pearson:  r={pearson_val_n:.4f}, p={p_val_n:.2e}")
    print(f"  - Spearman: r={spearman_val_n:.4f}, p={sp_val_n:.2e}")

    # 散布図 (雨なし)
    plt.figure(figsize=(6,6))
    plt.scatter(x_normal, y_normal, alpha=0.5, label="Data points")

    # 近似直線 (y = ax + b)
    coef_n = np.polyfit(x_normal, y_normal, 1)
    poly_n = np.poly1d(coef_n)
    x_linspace = np.linspace(x_normal.min(), x_normal.max(), 100)
    plt.plot(x_linspace, poly_n(x_linspace), 'r--', label='Linear Fit')

    plt.xlabel("Backbone Entropy (Normal)")
    plt.ylabel("Planner Entropy (Normal)")
    plt.title(f"No Rain:\nPearson={pearson_val_n:.3f}, Spearman={spearman_val_n:.3f}")
    plt.grid(True)
    plt.legend()

    out_path_normal = os.path.join(out_dir, "scatter_normal.png")
    plt.savefig(out_path_normal, dpi=150)
    plt.close()
    print(f"  --> 雨なし散布図を保存しました: {out_path_normal}")

    # ---------------------------
    #  雨あり vs 雨あり の相関
    # ---------------------------
    x_rain = df_merged['bb_entropy_rain']
    y_rain = df_merged['pl_entropy_rain']

    pearson_val_r, p_val_r   = pearsonr(x_rain, y_rain)
    spearman_val_r, sp_val_r = spearmanr(x_rain, y_rain)

    print("\n=== (rain) Backbone vs. Planner ===")
    print(f"  - Pearson:  r={pearson_val_r:.4f}, p={p_val_r:.2e}")
    print(f"  - Spearman: r={spearman_val_r:.4f}, p={sp_val_r:.2e}")

    # 散布図 (雨あり)
    plt.figure(figsize=(6,6))
    plt.scatter(x_rain, y_rain, alpha=0.5, label="Data points")

    # 近似直線 (y = ax + b)
    coef_r = np.polyfit(x_rain, y_rain, 1)
    poly_r = np.poly1d(coef_r)
    x_linspace = np.linspace(x_rain.min(), x_rain.max(), 100)
    plt.plot(x_linspace, poly_r(x_linspace), 'r--', label='Linear Fit')

    plt.xlabel("Backbone Entropy (Rain)")
    plt.ylabel("Planner Entropy (Rain)")
    plt.title(f"Rain:\nPearson={pearson_val_r:.3f}, Spearman={spearman_val_r:.3f}")
    plt.grid(True)
    plt.legend()

    out_path_rain = os.path.join(out_dir, "scatter_rain.png")
    plt.savefig(out_path_rain, dpi=150)
    plt.close()
    print(f"  --> 雨あり散布図を保存しました: {out_path_rain}\n")


if __name__ == '__main__':
    # 例: 2つのCSVを指定して実行
    backbone_csv = '/home/yoshi-22/FusionAD/UniAD/fusionad_graph/frame_entropy_values_fusionad.csv'  # frame_key, entropy_normal, entropy_rain ...
    planner_csv  = '/home/yoshi-22/FusionAD/UniAD/fusionad_graph/planner_entropy.csv'        # index, token, entropy_normal, entropy_rain ...
    # backbone_csv = '/home/yoshi-22/FusionAD/UniAD/uniad_graph/frame_entropy_values_uniad.csv'  # frame_key, entropy_normal, entropy_rain ...
    # planner_csv  = '/home/yoshi-22/FusionAD/UniAD/uniad_graph/planner_entropy_uni.csv'        # index, token, entropy_normal, entropy_rain ...
    


    analyze_backbone_planner(backbone_csv, planner_csv, out_dir='results_correlation')
