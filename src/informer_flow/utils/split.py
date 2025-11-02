from typing import Tuple, Dict
import pandas as pd

def time_splits(df: pd.DataFrame, split_cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按配置的 train/val/test 起止时间在 DatetimeIndex 上切分，闭区间。
    必须保证 df.index 为升序 DatetimeIndex（UTC）。
    """
    def _slice(span):
        s = pd.to_datetime(span["start"], utc=True)
        e = pd.to_datetime(span["end"],   utc=True)
        return df.loc[(df.index >= s) & (df.index <= e)].copy()
    tr = _slice(split_cfg["train"])
    va = _slice(split_cfg["val"])
    te = _slice(split_cfg["test"])
    if len(tr)==0 or len(va)==0 or len(te)==0:
        raise ValueError("某个切分区间为空，请检查 split 配置与数据时间范围。")
    return tr, va, te
