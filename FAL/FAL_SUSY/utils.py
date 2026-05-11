# utils.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import ssl
import bz2, gzip, shutil
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sampling import susy_iid

# ---------------------- 公共：联邦平均 / 实验信息 ----------------------
def average_weights(w_list):
    w_avg = copy.deepcopy(w_list[0])
    for k in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[k] += w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print('    IID' if args.iid else '    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

# ---------------------- SUSY：数据集载入与自动下载 ----------------------
class TabularDataset(Dataset):
    """X: float32 [N,D] (numpy), y: int64 [N]"""
    def __init__(self, X, y):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def _is_csv_like(path: str) -> bool:
    p = path.lower()
    return p.endswith(('.csv', '.csv.bz2', '.csv.gz', '.bz2', '.gz'))

def _download_requests(url: str, dst: str, insecure: bool):
    """优先用 requests + certifi 下载（带进度）"""
    import requests, certifi
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[SUSY] Downloading via requests:\n  {url}\n-> {dst}")
    with requests.get(url, stream=True, timeout=60,
                      verify=(False if insecure else certifi.where())) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = int(done * 100 / total)
                        print(f"\r  progress: {pct:3d}%", end="")
    print("\n  done.")

def _download_urllib(url: str, dst: str, insecure: bool):
    """回退 urllib + certifi CA（或可选跳过校验）"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[SUSY] Downloading via urllib:\n  {url}\n-> {dst}")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)

    def _report(count, block_size, total_size):
        if total_size > 0:
            pct = int(count * block_size * 100 / (total_size + 1e-6))
            pct = max(0, min(100, pct))
            print(f"\r  progress: {pct:3d}%", end="")

    urllib.request.urlretrieve(url, dst, _report)
    print("\n  done.")

def _download_with_progress(urls, dst_dir, insecure=False):
    """
    尝试多镜像，成功后返回 (本地文件路径, 成功的URL)。
    本地文件名保持与 URL 基名一致（从而保留 .gz/.bz2 后缀，便于后续解压识别）。
    """
    if isinstance(urls, str):
        urls = [urls]
    os.makedirs(dst_dir, exist_ok=True)
    last_err = None
    for url in urls:
        fname = os.path.basename(url)
        dst = os.path.join(dst_dir, fname)
        try:
            try:
                import requests, certifi   # 仅用于检查可用性
                _download_requests(url, dst, insecure)
            except Exception:
                _download_urllib(url, dst, insecure)
            return dst, url
        except Exception as e:
            last_err = e
            print(f"  failed: {url} ({e})")
            continue
    raise last_err

def _detect_compression(path: str):
    """基于扩展名 + 文件头字节双保险，返回 {'gz','bz2',None}"""
    p = path.lower()
    if p.endswith('.gz') or p.endswith('.csv.gz'):
        return 'gz'
    if p.endswith('.bz2') or p.endswith('.csv.bz2'):
        return 'bz2'
    # 读魔数
    try:
        with open(path, 'rb') as f:
            head = f.read(4)
        if len(head) >= 2 and head[0:2] == b'\x1f\x8b':
            return 'gz'
        if len(head) >= 3 and head[0:3] == b'BZh':
            return 'bz2'
    except Exception:
        pass
    return None

def _decompress_auto(src: str, dst_csv: str):
    """
    自动识别并解压到 dst_csv；若 src 非压缩，则直接复制。
    """
    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
    kind = _detect_compression(src)
    if kind == 'gz':
        print(f"[SUSY] Decompressing (gz):\n  {src}\n-> {dst_csv}")
        with gzip.open(src, 'rb') as f_in, open(dst_csv, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    elif kind == 'bz2':
        print(f"[SUSY] Decompressing (bz2):\n  {src}\n-> {dst_csv}")
        with bz2.open(src, 'rb') as f_in, open(dst_csv, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        # 不是压缩文件，直接复制
        print(f"[SUSY] Copying (no compression):\n  {src}\n-> {dst_csv}")
        shutil.copyfile(src, dst_csv)
    return dst_csv

def ensure_susy(susy_root: str, auto_download: bool, url: str, insecure: bool = False) -> str:
    """
    确保 SUSY 数据在本地可用；若缺失且允许，则自动下载。
    返回用于读取的实际路径（csv 或压缩 csv）。
    - 多镜像：UCI 官方(.gz) → UCI 物理镜像(.gz) → NTU/LIBSVM(.bz2)
    - 成功下载后：保持下载文件扩展名；解压到目标 csv（当目标是目录或 .csv）。
    """
    susy_root = os.path.expanduser(susy_root)
    abs_hint  = os.path.abspath(susy_root)

    MIRRORS = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
        "https://mlphysics.ics.uci.edu/data/susy/SUSY.csv.gz",
        url,  # 原始 NTU/LIBSVM（.bz2）
    ]

    # 情况 A：susy_root 是目录 -> <dir>/SUSY.csv
    if os.path.splitext(susy_root)[1] == "":
        os.makedirs(susy_root, exist_ok=True)
        csv_path = os.path.join(susy_root, "SUSY.csv")
        if os.path.isfile(csv_path):
            print(f"[SUSY] Found existing CSV: {os.path.abspath(csv_path)}")
            return csv_path
        if not auto_download:
            raise FileNotFoundError(f"[SUSY] Missing: {csv_path} (auto_download=0)")
        # 下到目录，保留扩展名
        downloaded, ok_url = _download_with_progress(MIRRORS, susy_root, insecure=insecure)
        _decompress_auto(downloaded, csv_path)
        return csv_path

    # 情况 B：susy_root 指向已存在的文件（csv 或压缩）
    if os.path.isfile(susy_root):
        print(f"[SUSY] Found dataset file: {abs_hint}")
        return susy_root

    # 情况 C：文件不存在，需要下载
    if not auto_download:
        raise FileNotFoundError(f"[SUSY] File not found: {abs_hint} (auto_download=0)")

    parent = os.path.dirname(susy_root) or "."
    os.makedirs(parent, exist_ok=True)

    # C1) 目标是 .csv ：下载到同目录（保留扩展名）→ 解压到该 csv
    if susy_root.lower().endswith('.csv'):
        downloaded, ok_url = _download_with_progress(MIRRORS, parent, insecure=insecure)
        _decompress_auto(downloaded, susy_root)
        return susy_root

    # C2) 目标是压缩包（.bz2/.gz）：直接下载到该路径
    if _is_csv_like(susy_root):
        _download_with_progress(MIRRORS, parent, insecure=insecure)
        # 如果下载文件名与目标不一致，复制过去（保持你传入的目标名）
        fname = os.path.basename(MIRRORS[-1])  # 不一定是最后一个成功，但一般保持一致
        # 更稳妥：直接用刚刚下载的文件路径
        downloaded, ok_url = _download_with_progress(MIRRORS, parent, insecure=insecure)
        if os.path.abspath(os.path.join(parent, os.path.basename(downloaded))) != os.path.abspath(susy_root):
            shutil.copyfile(downloaded, susy_root)
        return susy_root

    # 兜底：当作目录
    os.makedirs(susy_root, exist_ok=True)
    csv_path = os.path.join(susy_root, "SUSY.csv")
    downloaded, ok_url = _download_with_progress(MIRRORS, susy_root, insecure=insecure)
    _decompress_auto(downloaded, csv_path)
    return csv_path

def _load_susy_csv(path: str):
    """读取 CSV 或压缩 CSV；非 CSV 则尝试按 svmlight/libsvm 读取。"""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[SUSY] Not found: {abs_path}")

    if _is_csv_like(path):
        try:
            df = pd.read_csv(path, header=None, dtype=np.float32, compression='infer')
        except Exception:
            df = pd.read_csv(path, dtype=np.float32, compression='infer')
        arr = df.to_numpy()
        if arr.shape[1] < 2:
            raise ValueError(f"[SUSY] CSV has <2 cols: {abs_path}, shape={arr.shape}")
        y = arr[:, 0].astype(np.int64)
        X = arr[:, 1:].astype(np.float32)
    else:
        from sklearn.datasets import load_svmlight_file
        Xsp, y = load_svmlight_file(path)
        X = Xsp.toarray().astype(np.float32)
        y = np.asarray(y)
        y = (y > 0).astype(np.int64) if y.min() < 0 else y.astype(np.int64)

    uniq = set(np.unique(y).tolist())
    if uniq - {0, 1}:
        y = (y == max(uniq)).astype(np.int64)

    print(f"[SUSY] Loaded: {abs_path} | X={X.shape}, y={y.shape}, labels={sorted(set(y.tolist()))}")
    return X, y

def _minmax_fit_transform(X_train, X_test):
    xmin = X_train.min(axis=0, keepdims=True)
    xmax = X_train.max(axis=0, keepdims=True)
    denom = np.where(xmax > xmin, xmax - xmin, 1.0)
    X_train = np.clip((X_train - xmin) / denom, 0.0, 1.0)
    X_test  = np.clip((X_test  - xmin) / denom, 0.0, 1.0)
    return X_train.astype(np.float32), X_test.astype(np.float32)

def get_dataset(args):
    # 1) 确保数据存在（必要时自动下载）
    susy_path = getattr(args, "susy_root", "../data/SUSY.csv")
    susy_url  = getattr(args, "susy_url",
                        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.csv.bz2")
    auto_dl   = bool(getattr(args, "auto_download", 1))
    insecure  = bool(getattr(args, "insecure", 0))

    print(f"[SUSY] Target: {os.path.abspath(susy_path)} (auto_download={int(auto_dl)}, insecure={int(insecure)})")
    susy_path = ensure_susy(susy_path, auto_download=auto_dl, url=susy_url, insecure=insecure)

    # 2) 读取
    X, y = _load_susy_csv(susy_path)

    # 3) 全局 8:2 切分
    N = X.shape[0]
    rng = np.random.RandomState(getattr(args, "seed", 1))
    idx = rng.permutation(N)
    split = int(0.8 * N)
    tr_idx, te_idx = idx[:split], idx[split:]
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    # 4) Min-Max 到 [0,1]（与 PGD clip 一致）
    X_train, X_test = _minmax_fit_transform(X_train, X_test)

    # 5) Dataset & IID 客户端划分
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset  = TabularDataset(X_test,  y_test)
    user_groups   = susy_iid(len(train_dataset), args.num_users, seed=getattr(args, "seed", 1))
    return train_dataset, test_dataset, user_groups
