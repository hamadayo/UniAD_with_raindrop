import pickle

# ファイルをバイナリ読み込みモードで開く
with open('/home/yoshi-22/UniAD/output/results.pkl', 'rb') as f:
    data = pickle.load(f)

# dataが辞書の場合、キー一覧を表示
if isinstance(data, dict):
    print("キー一覧:")
    for key in data.keys():
        print(key)
    for key2 in data['bbox_results'][0].keys():
        print(key2)

    ma = data['my_activations']
    print("Type of my_activations:", type(ma))

        
    print(len(data['bbox_results']))
    print(data['my_activations'][0]['my_activations'].keys())
    print(data['my_activations']['img_backbone.layer4.2'][0][0].shape)
    print(data['bbox_results'][0]['cross_attn_list'][0].shape)
    # print("\nキーと値の組み合わせ:")
    # for key, value in data.items():
    #     print(f"{key}: {value}")
else:
    print("読み込んだデータは辞書ではありません。")
    print("データの型:", type(data))
    print("データ内容:", data)
