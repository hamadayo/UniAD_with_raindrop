import argparse
from os import path as osp
import sys
from data_converter import uniad_nuscenes_converter as nuscenes_converter
sys.path.append('.')


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """
    nuScenesデータセットに関連するデータを準備します。

    関連データには、基本情報、2Dアノテーション、およびグラウンドトゥルースデータベースを記録した「.pkl」ファイルが含まれます。

    Args:
        root_path (str): データセットのルートパス。（./data/nuscenes)
        info_prefix (str): 情報ファイル名のプレフィックス。(nuscenes)
        version (str): データセットのバージョン。(v1.0-mini)
        dataset_name (str): データセットクラスの名前。(NuScenesDataset)
        out_dir (str): グラウンドトゥルースデータベース情報の出力ディレクトリ。(./data/infos)
        max_sweeps (int): 入力として使用する連続フレーム数。デフォルト: 10
    """
    # データセットの情報をpklファイルに保存
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        # train_version = f'{args.version}-trainval'
        # nuscenes_data_prep(
        #     root_path=args.root_path,
        #     can_bus_root_path=args.canbus,
        #     info_prefix=args.extra_tag,
        #     version=train_version,
        #     dataset_name='NuScenesDataset',
        #     out_dir=args.out_dir,
        #     max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)