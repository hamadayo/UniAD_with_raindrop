def extract_attention_weights_lines(file_path):
    attention_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # "attention_weights"という文字列が含まれている行のみを抽出
            if "" in line:
                # 末尾の改行や空白を取り除き、リストに追加
                attention_lines.append(line.strip())
    return attention_lines

if __name__ == "__main__":
    file_path = "/home/yoshi-22/UniAD/gethook.log"
    results = extract_attention_weights_lines(file_path)
    # 抽出結果を出力
    for line in results:
        print(line)
