import mmcv
from nuscenes import NuScenes


# prediction.pkl を読み込む
pred_data = mmcv.load('/home/yoshi-22/UniAD/output/results_tr.pkl')
prediction_tokens = {entry['token'] for entry in pred_data['bbox_results']}

print(f"Number of tokens in prediction.pkl: {len(prediction_tokens)}")
missing_tokens = dataset_tokens - prediction_tokens
print(f"Number of missing tokens: {len(missing_tokens)}")
print(f"Sample missing tokens: {list(missing_tokens)[:10]}")  # 欠損トークンの一部を表示
