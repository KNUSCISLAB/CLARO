"""HAI 23.05 preprocessing and loading of the preprocessed arrays."""
import numpy as np

from . import config


def preprocess(raw_dir=None, out_dir=None, suffix='.csv'):
    """Build the arrays used by every later step from the official HAI 23.05 files.

    Training files 1-4 and test files 1-2 are concatenated in order. Channels that
    are constant on the training data are removed, and missing values are replaced by
    the training mean. Writes ``X_train.npy`` and ``X_test.npy`` (float32),
    ``y_test.npy`` (int8) and ``sensor_cols.txt`` to ``out_dir``.
    """
    import pandas as pd
    raw_dir = (config.RAW_DIR if raw_dir is None else raw_dir) / 'hai-23.05'
    out_dir = config.DATA_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def read(stem, ids):
        return pd.concat([pd.read_csv(raw_dir / f'{stem}{i}{suffix}') for i in ids], ignore_index=True)

    train, test, label = read('hai-train', range(1, 5)), read('hai-test', range(1, 3)), read('label-test', range(1, 3))
    columns = [c for c in train.columns if c != 'timestamp']
    x_train = train[columns].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    x_test = test[columns].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    y = label['label'].values.astype(np.int8)

    keep = x_train.std(axis=0) > 1e-6
    columns = [c for c, k in zip(columns, keep) if k]
    x_train, x_test = x_train[:, keep], x_test[:, keep]
    means = np.nanmean(x_train, axis=0)
    for array in (x_train, x_test):
        for j in range(array.shape[1]):
            array[np.isnan(array[:, j]), j] = means[j]

    np.save(out_dir / 'X_train.npy', x_train)
    np.save(out_dir / 'X_test.npy', x_test)
    np.save(out_dir / 'y_test.npy', y)
    (out_dir / 'sensor_cols.txt').write_text('\n'.join(columns))
    return x_train, x_test, y, columns


def channel_names():
    """Channel order shared by the data arrays and every graph."""
    return [n.split('\\')[-1] for n in config.SENSOR_COLS.read_text().splitlines() if n.strip()]


def _check_schema():
    processed = config.DATA_DIR / 'sensor_cols.txt'
    if processed.exists():
        found = [n for n in processed.read_text().splitlines() if n.strip()]
        if found != channel_names():
            raise ValueError(f'{processed} does not match {config.SENSOR_COLS}')


def load_train():
    """Normal training data (memory-mapped float32, N x F)."""
    _check_schema()
    return np.load(config.DATA_DIR / 'X_train.npy', mmap_mode='r')


def load_test():
    """Test data as float64 (T x F)."""
    _check_schema()
    return np.asarray(np.load(config.DATA_DIR / 'X_test.npy', mmap_mode='r'), dtype=float)


def load_labels():
    """Point labels of the test data (used for calibration and scoring only)."""
    return np.load(config.DATA_DIR / 'y_test.npy').ravel().astype(int)


def valid_mask(n):
    """Window rows that are usable online.

    Windows that span the boundary between the two test files are invalid; the
    alarm state is reset there.
    """
    valid = np.ones(n, dtype=bool)
    valid[config.TEST_FILE_BOUNDARY - config.OFFSET:config.TEST_FILE_BOUNDARY] = False
    return valid
