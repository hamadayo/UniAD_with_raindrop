import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon


def main(csv_path, out_dir="entropy_compare_output"):
    """
    CSVを読み込み、entropy_normal と entropy_rain の相関・JSDを計算。
    また、それらの値を可視化するグラフを出力するサンプルコード。
    """

    # 出力先ディレクトリの作成
    os.makedirs(out_dir, exist_ok=True)

    # CSVの読み込み
    frame_keys = []
    normal_list = []
    rain_list   = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_keys.append(int(row['frame_key']))
            normal_list.append(float(row['entropy_normal']))
            rain_list.append(float(row['entropy_rain']))

    normal_array = np.array(normal_list, dtype=np.float64)
    rain_array   = np.array(rain_list,   dtype=np.float64)

    # ----------------------------------
    # 1) ピアソン / スピアマン相関の計算
    # ----------------------------------
    # scipy.stats.pearsonr/spearmanr は (相関値, p値) を返す
    pearson_val,  pearson_p  = pearsonr(normal_array, rain_array)
    spearman_val, spearman_p = spearmanr(normal_array, rain_array)

    # ----------------------------------
    # 2) JSD の計算
    #     ・Jensen-Shannon distance = jensenshannon(p, q)
    #     ・Jensen-Shannon divergence = distance^2 がしばしば用いられる
    # ----------------------------------
    # entropies は基本的に正。念のため負やゼロが入っていてもクラッシュしないようclip
    # （ただしエントロピーが負になることは通常はないため、そのままでもOKな場合あり）
    normal_clipped = np.clip(normal_array, 0, None)
    rain_clipped   = np.clip(rain_array,   0, None)

    # 分布として正規化 (合計1に)
    sum_n = normal_clipped.sum()
    sum_r = rain_clipped.sum()
    if sum_n > 0:
        normal_dist = normal_clipped / sum_n
    else:
        normal_dist = np.zeros_like(normal_clipped)

    if sum_r > 0:
        rain_dist = rain_clipped / sum_r
    else:
        rain_dist = np.zeros_like(rain_clipped)

    js_distance = jensenshannon(normal_dist, rain_dist)
    js_divergence = js_distance**2

    # ----------------------------------
    # 結果の表示
    # ----------------------------------
    print("=== 結果 ===")
    print(f"Pearson相関:  {pearson_val:.4f} (p={pearson_p:.4e})")
    print(f"Spearman相関: {spearman_val:.4f} (p={spearman_p:.4e})")
    print(f"Jensen-Shannonダイバージェンス: {js_divergence:.6f}")
    print("============================\n")

    # ----------------------------------
    # 可視化 (元コードの 3-pane 表示を1次元折れ線プロット版に)
    # ----------------------------------
    #   左: normal の折れ線
    #   中: rain の折れ線
    #   右: (rain - normal) の差分折れ線
    diff_array = rain_array - normal_array

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)

    # 左: Normal
    axes[0].plot(frame_keys, normal_array, marker='o', color='tab:blue')
    axes[0].set_title("Normal Entropy")
    axes[0].set_xlabel("frame_key")
    axes[0].set_ylabel("entropy_normal")

    # 中: Rain
    axes[1].plot(frame_keys, rain_array, marker='o', color='tab:orange')
    axes[1].set_title("Rain Entropy")
    axes[1].set_xlabel("frame_key")
    axes[1].set_ylabel("entropy_rain")

    # 右: 差分 (Rain - Normal)
    axes[2].plot(frame_keys, diff_array, marker='o', color='tab:green')
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    axes[2].set_title("Rain - Normal")
    axes[2].set_xlabel("frame_key")
    axes[2].set_ylabel("diff")

    plt.tight_layout()
    out_fig_path = os.path.join(out_dir, "entropy_compare_3pane.png")
    plt.savefig(out_fig_path)
    plt.close(fig)
    print(f"--> 3Paneグラフを保存しました: {out_fig_path}")

    # ----------------------------------
    # オプション: 散布図で相関可視化
    # ----------------------------------
    fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=120)
    ax2.scatter(normal_array, rain_array, color='tab:purple', alpha=0.7, edgecolors='k')
    ax2.set_xlabel("entropy_normal")
    ax2.set_ylabel("entropy_rain")
    ax2.set_title("Normal vs Rain (Scatter)")

    # テキストで相関値を表示
    text_str = (
        f"Pearson: {pearson_val:.3f}\n"
        f"Spearman: {spearman_val:.3f}\n"
        f"JSD: {js_divergence:.4f}"
    )
    # 右上あたりに配置
    ax2.text(0.95, 0.05, text_str, transform=ax2.transAxes,
             fontsize=10, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    out_scatter_path = os.path.join(out_dir, "entropy_compare_scatter.png")
    plt.savefig(out_scatter_path)
    plt.close(fig2)
    print(f"--> 散布図を保存しました: {out_scatter_path}")


if __name__ == "__main__":
    """
    使い方イメージ:
      python this_script.py

    実際には以下のように適宜書き換えて使って下さい。
    """
    # テスト用のCSVパス（例）
    # csv_path_example = "/home/yoshi-22/FusionAD/UniAD/fusionad_graph/frame_entropy_values_fusionad.csv"
    csv_path_example = "/home/yoshi-22/FusionAD/UniAD/uniad_graph/frame_entropy_values_uniad.csv"

    # 出力先ディレクトリ
    out_dir_example  = "entropy_compare_output"

    # 実行
    main(csv_path_example, out_dir_example)
