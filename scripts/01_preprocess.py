"""Step 1: preprocess the official HAI 23.05 files into numpy arrays.

Reads <CLARO_RAW>/hai-23.05/{hai-train1-4,hai-test1-2,label-test1-2}.csv and writes
X_train.npy, X_test.npy, y_test.npy and sensor_cols.txt to <CLARO_DATA>.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claro import config, data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=config.RAW_DIR, help='clone of the HAI repository')
    parser.add_argument('--out', type=Path, default=config.DATA_DIR, help='output directory')
    parser.add_argument('--suffix', default='.csv', help='file extension of the HAI files')
    args = parser.parse_args()
    x_train, x_test, y, columns = data.preprocess(args.raw, args.out, args.suffix)
    print(f'X_train {x_train.shape}, X_test {x_test.shape}, attack fraction {y.mean():.4f}, channels {len(columns)}')
    if columns != data.channel_names():
        print(f'warning: channel list differs from {config.SENSOR_COLS}; the shipped graphs do not apply')


if __name__ == '__main__':
    main()
