import argparse
import sys
import json
import torch
import torch_geometric
import pytorch_lightning as pl
from pathlib import Path

import t4c22
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import class_fractions
from t4c22.t4c22_config import load_basedir
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset

from GNN_model import CongestioNN
from GNN_model import LinkPredictor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


def create_parser(args):
    parser = argparse.ArgumentParser()

    help_msg = "City"
    parser.add_argument(
        "-c", "--city", type=str, default="london", help=help_msg
    )
    help_msg = "Device"
    parser.add_argument(
        "-d", "--device", type=int, default=0, help=help_msg
    )

    args = parser.parse_args(args)

    return parser, args


class CongestionSystem(pl.LightningModule):
    def __init__(self, city):
        super().__init__()

        self.city = city

        with open("model_parameters.json", "r") as f:
            model_parameters = json.load(f)

        city_class_fractions = class_fractions[city]

        city_class_weights = get_weights_from_class_fractions(
            [city_class_fractions[c] for c in ["green", "yellow", "red"]]
        )
        if model_parameters["Predictor"]["out_channels"] == 4:
            city_class_weights = city_class_weights + [0.07]  # weight for no data

        city_class_weights = torch.tensor(city_class_weights).float()

        self.loss = torch.nn.CrossEntropyLoss(
            weight=city_class_weights, ignore_index=3
        )

        self.model = CongestioNN(**model_parameters["GNN"])

        self.predictor = LinkPredictor(**model_parameters["Predictor"])

    def forward(self, data):

        data.x = data.x.nan_to_num(-1)
        data.x[:, 4] = (data.x[:, 4] + 0.14059608) / 0.11106573
        data.x[:, 5] = (data.x[:, 5] - 51.493862) / 0.096448936
        data.y = data.y.nan_to_num(3)

        h = self.model(data)
        assert (h.isnan()).sum() == 0, h
        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])

        y_hat = self.predictor(x_i, x_j)

        return y_hat

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch.y.long()
        loss = self.loss(y_hat, y)

        self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        torch.save(
            self.model.state_dict(),
            f"{self.city}_model_{self.current_epoch:03d}.pt"
        )
        torch.save(
            self.predictor.state_dict(),
            f"{self.city}_predictor_{self.current_epoch:03d}.pt"
        )

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch.y.long()
        loss = self.loss(y_hat, y)
        self.log("loss/val", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters()},
                {"params": self.predictor.parameters()}
            ],
            lr=1e-4,
            weight_decay=0.00001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.1
        )
        return [optimizer], [scheduler]


if __name__ == '__main__':

    parser, args = create_parser(sys.argv[1:])

    t4c_apply_basic_logging_config(loglevel="INFO")
    # load BASEDIR from file, change to your data root
    BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
    city = args.city
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
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [spl, len(dataset) - spl]
    )

    train_loader = torch_geometric.loader.dataloader.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8
    )

    val_loader = torch_geometric.loader.dataloader.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    system = CongestionSystem(city)
    trainer = pl.Trainer(
        devices=[args.device],
        accelerator="gpu",
        max_epochs=3,
        logger=TensorBoardLogger(""),
        callbacks=[TQDMProgressBar(refresh_rate=100)]
    )
    trainer.fit(system, train_loader, val_loader)
