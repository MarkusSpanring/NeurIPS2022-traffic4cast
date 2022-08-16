from functools import partial
from pathlib import Path
import numpy as np

import t4c22
from t4c22.t4c22_config import load_basedir
from t4c22.t4c22_config import load_road_graph

import pyarrow.parquet as pq


def unique_ctr_list(basedir: Path, city, split="train"):

    path_to_ctr = basedir / "train" / city / "input"
    ctr_list = []
    for f in path_to_ctr.glob(f"counters*.parquet"):
        df = pq.read_table(f).to_pandas()
        ctr_list.append(df["node_id"].unique().copy())
        break

    return np.unique(np.array(ctr_list))


def bfs(start, graph, hotlist, return_path=False):
    visited, queue = [], [(int(start), [int(start)])]
    while queue:
        (vertex, path) = queue.pop(0)
        vertex = int(vertex)

        if vertex not in visited:
            visited.append(vertex)
            children = graph[graph["u"] == vertex]["children"].values
            if len(children) == 1:
                children = set(children[0])
            else:
                continue

            for nxt in children - set(path):
                if nxt in hotlist:
                    if return_path:
                        return path + [nxt]
                    else:
                        return nxt
                else:
                    queue.append((nxt, path + [nxt]))
    return -1


if __name__ == "__main__":
    BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)

    city = "london"
    root = BASEDIR

    df_edges, df_nodes = load_road_graph(root, city)

    road_graph = df_edges.groupby("u")["v"].apply(list).reset_index(name="children").copy(deep=True)

    ctr_list = unique_ctr_list(root, city=city)
    fn = partial(bfs, graph=road_graph, hotlist=ctr_list)
    road_graph["nearest_ctr"] = road_graph["u"].apply(fn)

    nearest_ctr = road_graph[["u", "nearest_ctr"]].copy(deep=True)
    nearest_ctr = nearest_ctr.rename(columns={"u": "node_id"})
    nearest_ctr.to_parquet(BASEDIR / "road_graph" / city / "nearest_ctr.parquet")
