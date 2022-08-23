from functools import partial
from pathlib import Path
import numpy as np
import tqdm
import pandas as pd

import t4c22
from t4c22.t4c22_config import load_basedir
from t4c22.t4c22_config import load_road_graph

import pyarrow.parquet as pq


def unique_ctr_list(basedir: Path, city, split="train"):

    path_to_ctr = basedir / "train" / city / "input"
    ctr_list = []
    for f in path_to_ctr.glob(f"counters*.parquet"):
        df = pq.read_table(f).to_pandas()
        ctr_list.append(df["node_id"].unique().copy(deep=True))
        break

    return np.unique(np.array(ctr_list))


def bfs(start, graph, hotlist, return_path=False):
    visited = set()
    queue = []
    ctr = []

    queue.append(start)
    while queue:
        vertex = queue.pop(0)
        vertex = int(vertex)
        if vertex in hotlist:
            visited.add(vertex)
            ctr.append(vertex)
            if len(ctr) == 5:
                return ctr
            continue

        if vertex not in visited:
            visited.add(vertex)
            children = graph[graph["u"] == vertex]["children"].values
            if len(children) == 1:
                # return value is [list(children)]
                children = set(children[0])
            else:
                # no children
                continue

            for nxt in children - set(visited):
                if nxt in hotlist:
                    visited.add(vertex)
                    ctr.append(vertex)
                    if len(ctr) == 5:
                        return ctr
                    continue

                queue.append(nxt)
    return ctr


if __name__ == "__main__":
    BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)

    city = "madrid"
    root = BASEDIR

    df_edges, df_nodes = load_road_graph(root, city)

    road_graph = df_edges.groupby("u")["v"].apply(list).reset_index(name="children").copy(deep=True)

    ctr_list = unique_ctr_list(root, city=city)
    nearest_ctr_lst = []
    pbar = tqdm.tqdm(road_graph["u"])
    for u in pbar:
        pbar.set_postfix({"node: ": str(u)})
        nearest_ctr_lst += [[u, v] for v in bfs(u, graph=road_graph, hotlist=ctr_list) if v != -1]

    nearest_ctr = pd.DataFrame(nearest_ctr_lst, columns=["node_id", "nearest_ctr"])
    nearest_ctr.to_parquet(BASEDIR / "road_graph" / city / "nearest_ctr.parquet")
