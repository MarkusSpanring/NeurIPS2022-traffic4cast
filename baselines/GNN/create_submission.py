import os
import pandas as pd
import torch
import tqdm
from pathlib import Path
import numpy as np

import t4c22
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config

from t4c22.t4c22_config import load_basedir
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.misc.notebook_helpers import restartkernel  # noqa:F401

from GNN_model import CongestioNN
from GNN_model import LinkPredictor
import json
import zipfile
from t4c22.t4c22_config import load_road_graph


@torch.no_grad()
def create_prediction(dataset, model, predictor, day, city, device):
    model.to(device)
    predictor.to(device)
    dfs = []

    pbar = tqdm.tqdm(enumerate(dataset), total=len(dataset))
    pbar.set_description(city)
    for idx, data in pbar:

        data.x = data.x.nan_to_num(-1)

        data = data.to(device)
        h = model(data)

        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])

        # logits
        y_hat = predictor(x_i, x_j)
        df = pd.DataFrame(
            torch.nn.functional.softmax(y_hat, dim=1).cpu().numpy(),
            columns=["0", "1", "2"]
        )
        df = pd.concat(
            (
                df,
                pd.DataFrame(
                    y_hat.cpu().numpy(),
                    columns=["logit_green", "logit_yellow", "logit_red"])
            ),
            axis=1
        )
        df_data = dataset.torch_road_graph_mapping._torch_to_df_cc(
            data=y_hat, day=day, t=idx
        )
        df["u"] = df_data["u"].copy(deep=True)
        df["v"] = df_data["v"].copy(deep=True)
        df["day"] = df_data["day"].copy(deep=True)
        df["test_idx"] = df_data["t"].copy(deep=True)

        if data.y is not None:
            data.y = data.y.nan_to_num(3)
            df["y"] = data.y.cpu().numpy()
        else:
            df["y"] = -1

        df["y_hat"] = df[["0", "1", "2"]].idxmax(axis=1).astype("int64")
        dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def load_data(root, city, split, day):

    t4c_apply_basic_logging_config(loglevel="INFO")
    dataset = T4c22GeometricDataset(
        root=root,
        city=city,
        split=split,
        cachedir=Path("tmp/processed"),
        add_nearest_ctr_edge=True,
        day_t_filter=lambda _day, t: _day == day
    )
    return dataset


def load_model(city, epoch, device):

    with open("model_parameters.json", "r") as f:
        model_parameters = json.load(f)
    model = CongestioNN(**model_parameters["GNN"])
    model.load_state_dict(
        torch.load(f"{city}_model_{epoch:03d}.pt", map_location=device)
    )

    predictor = LinkPredictor(**model_parameters["Predictor"])
    predictor.load_state_dict(
        torch.load(f"{city}_predictor_{epoch:03d}.pt", map_location=device)
    )

    return model, predictor


def create_submission(root, city, submission_name, df):
    outdir = Path("submission") / city
    outdir.mkdir(exist_ok=True, parents=True)

    columns = ["u", "v", "logit_green", "logit_yellow", "logit_red", "test_idx"]
    submission_df = df[columns].copy(deep=True)

    df_edges, df_nodes = load_road_graph(root, city)
    submission_df = submission_df.merge(
        df_edges, left_on=["u", "v"], right_on=["u", "v"], suffixes=["_pred", ""]
    )
    assert len(submission_df) == 100 * len(df_edges), f"bad merge"

    submission_df.to_parquet(outdir / f"{submission_name}.parquet")
    return submission_df


def zip_submission(cities, submission_name):

    submission_zip = Path("submission") / f"{submission_name}.zip"
    with zipfile.ZipFile(submission_zip, "w") as z:
        for city in cities:
            z.write(
                filename=Path("submission") / city / f"{submission_name}.parquet",
                arcname=os.path.join(city, "labels", f"cc_labels_test.parquet")
            )
    print(submission_zip)


def main():

    epochs = 2
    submission_name = f"t4c_gnn_baseline_{epochs}"

    cities = ["london", "madrid", "melbourne"]

    day = "2019-07-01"
    split = "train"

    device = 0
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    root = load_basedir(fn="t4c22_config.json", pkg=t4c22)

    for city in cities:
        dataset = load_data(root, city, split, day)
        model, predictor = load_model(city, epochs, device)
        pred_df = create_prediction(dataset, model, predictor, day, city, device)
        submission_df = create_submission(root, city, submission_name, pred_df)

        tmp_values = submission_df["logit_green"].values
        assert np.isnan(tmp_values).sum() == 0, np.isnan(tmp_values).sum()

        tmp_values = submission_df["logit_yellow"].values
        assert np.isnan(tmp_values).sum() == 0, np.isnan(tmp_values).sum()

        tmp_values = submission_df["logit_red"].values
        assert np.isnan(tmp_values).sum() == 0, np.isnan(tmp_values).sum()

    zip_submission(cities, submission_name)


if __name__ == '__main__':
    main()
