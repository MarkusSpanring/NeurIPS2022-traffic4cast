import statistics
from collections import defaultdict

import torch
import torch_geometric
import tqdm
from pathlib import Path

import t4c22
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import class_fractions
from t4c22.t4c22_config import load_basedir
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset

from GNN_model import CongestioNN
from GNN_model import LinkPredictor


def train(model, predictor, dataset, optimizer, batch_size, device, city_class_weights):
    model.train()

    losses = []
    optimizer.zero_grad()
    pbar = tqdm.tqdm(
        torch_geometric.loader.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16
        ),
        "train",
        total=len(dataset) // batch_size,
    )
    for data in pbar:

        data = data.to(device)

        data.x = data.x.nan_to_num(-1)
        data.y = data.y.nan_to_num(3)

        h = model(data)
        assert (h.isnan()).sum() == 0, h
        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])

        y_hat = predictor(x_i, x_j)

        y = data.y
        y = y.long()
        loss_f = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
        loss = loss_f(y_hat, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.cpu().item())
    return losses


@torch.no_grad()
def test(model, predictor, validation_dataset, batch_size, device, city_class_weights):
    model.eval()
    total_loss = 0.0

    y_hat_list = []
    y_list = []

    pbar = tqdm.tqdm(
        torch_geometric.loader.dataloader.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16
        ),
        "test",
        total=len(validation_dataset),
    )
    for data in pbar:
        data = data.to(device)

        data.x = data.x.nan_to_num(-1)
        data.y = data.y.nan_to_num(3)
        h = model(data)

        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])

        y_hat = predictor(x_i, x_j)

        y_hat_list.append(y_hat)
        y_list.append(data.y)

    y_hat = torch.cat(y_hat_list, 0)
    y = torch.cat(y_list, 0)
    y = y.long()
    loss = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
    total_loss = loss(y_hat, y)

    return total_loss


if __name__ == "__main__":

    t4c_apply_basic_logging_config(loglevel="DEBUG")
    # load BASEDIR from file, change to your data root
    BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
    city = "london"
    # city = "melbourne"
    # city = "madrid"
    dataset = T4c22GeometricDataset(
        root=BASEDIR,
        city=city,
        split="train",
        cachedir=Path("tmp/processed"),
        add_nearest_ctr_edge=True
    )
    spl = int(((0.8 * len(dataset)) // 2) * 2)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [spl, len(dataset) - spl])

    city_class_fractions = class_fractions[city]
    city_class_fractions

    city_class_weights = get_weights_from_class_fractions([city_class_fractions[c] for c in ["green", "yellow", "red"]])
    city_class_weights.append(0.1) # weight for no data
    city_class_weights = torch.tensor(city_class_weights).float()
    city_class_weights

    batch_size = 1
    eval_steps = 1
    epochs = 20

    model_parameters = {
        "GNN": {
            "hidden_layer": 3,
            "in_features": 6,
            "hidden_features": 128,
            "out_features": 128
        },
        "Predictor": {
            "in_channels": 128,
            "hidden_channels": 256,
            "out_channels": 4,
            "num_layers": 3,
            "dropout": 0.0
        }
    }

    device = 0
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    city_class_weights = city_class_weights.to(device)

    model = CongestioNN(**model_parameters["GNN"]).to(device)

    predictor = LinkPredictor(**model_parameters["Predictor"]).to(device)

    train_losses = defaultdict(lambda: [])
    val_losses = defaultdict(lambda: -1)
    val_loss = ""

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    pbar = tqdm.tqdm(range(1, 1 + epochs), "epochs", total=epochs)
    for epoch in pbar:
        losses = train(
            model,
            predictor,
            dataset=train_dataset,
            optimizer=optimizer,
            batch_size=batch_size,
            device=device,
            city_class_weights=city_class_weights
        )
        train_losses[epoch] = losses

        if epoch % eval_steps == 0:

            val_loss = test(
                model,
                predictor,
                validation_dataset=val_dataset,
                batch_size=batch_size,
                device=device,
                city_class_weights=city_class_weights
            )
            val_losses[epoch] = val_loss

            torch.save(model.state_dict(), f"GNN_model_{epoch:03d}.pt")
            torch.save(predictor.state_dict(), f"GNN_predictor_{epoch:03d}.pt")
        pbar.set_postfix({"train loss": statistics.mean(losses), "val loss": float(val_loss)})
