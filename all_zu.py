import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics_from_csv(csv_path, out_dir="metrics_plots"):
    """
    指定した CSV ファイル (attn_compare_metrics.csv) を読み込み、
    様々な分布指標 (pearson, spearman, kl_pq, kl_qp, jsd) を
    frame_index(=row order) を x 軸にして可視化する。
    
    Args:
        csv_path (str): CSVファイルへのパス
        out_dir (str): グラフ画像の出力先
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # CSV を読み込み
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            rows.append(line)
    
    if not rows:
        print("CSVが空です。")
        return
    
    # CSV の各行は文字列なので、必要な列を float に変換
    # x 軸に使う index (int化) とか
    index_list = []
    
    # metric ごとにリストを作る
    pearson_vals  = []
    spearman_vals = []
    kl_pq_vals    = []
    kl_qp_vals    = []
    jsd_vals      = []
    
    # CSV にエントリがあれば、
    # 'index','token','entropy_normal','entropy_rain',...,'pearson','spearman','kl_pq','kl_qp','jsd'
    # などが含まれている想定。
    for row in rows:
        idx = int(row['index'])
        index_list.append(idx)
        
        pearson_vals.append(  float(row['pearson']) )
        spearman_vals.append( float(row['spearman']) )
        kl_pq_vals.append(    float(row['kl_pq']) )
        kl_qp_vals.append(    float(row['kl_qp']) )
        jsd_vals.append(      float(row['jsd']) )
    
    # ------------------------------------------------
    # x 軸としては index_list を使う（フレームindex: 0,1,2,3,...）
    # ------------------------------------------------
    x_vals = index_list
    
    # 各指標をまとめて管理
    # ここで "title" や "y_label" も分かりやすく補足説明
    # 例えば「(1に近いほど強い相関)」など
    metrics_info = [
        {
            'name': 'pearson',
            'values': pearson_vals,
            'title': "Pearson Correlation vs. Frame Index\n(1 => perfect linear correlation, 0 => no correlation)",
            'ylim': (-0.1, 1.05)  # 例: 相関係数なので -1～1 だがここでは -0.1～1.05 くらいに
        },
        {
            'name': 'spearman',
            'values': spearman_vals,
            'title': "Spearman Correlation vs. Frame Index\n(1 => perfect rank correlation, 0 => no rank correlation)",
            'ylim': (-0.1, 1.05)
        },
        {
            'name': 'kl_pq',
            'values': kl_pq_vals,
            'title': "KL(p||q) vs. Frame Index\n(0 => same distribution, larger => more difference)",
            'ylim': None  # 自動スケール
        },
        {
            'name': 'kl_qp',
            'values': kl_qp_vals,
            'title': "KL(q||p) vs. Frame Index\n(0 => same distribution, larger => more difference)",
            'ylim': None
        },
        {
            'name': 'jsd',
            'values': jsd_vals,
            'title': "Jensen-Shannon Divergence vs. Frame Index\n(0 => same distribution, higher => more difference)",
            'ylim': None
        },
    ]
    
    for m in metrics_info:
        metric_name = m['name']
        metric_data = m['values']
        plot_title  = m['title']
        ylim_range  = m['ylim']
        
        plt.figure(figsize=(6,4))
        
        # 折れ線に点を付けず、線だけを表示
        plt.plot(x_vals, metric_data, label=metric_name, linewidth=2)
        
        plt.xlabel("Frame Index")
        plt.ylabel(metric_name)
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        
        if ylim_range is not None:
            plt.ylim(ylim_range)
        
        out_path = os.path.join(out_dir, f"{metric_name}_vs_index.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    csv_path = 'attn_compare_metrics.csv'  # これを差し替えて
    out_dir  = 'metrics_plots_improved'
    plot_metrics_from_csv(csv_path, out_dir=out_dir)
