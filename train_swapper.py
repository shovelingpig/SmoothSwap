import os
import sys
import math
from glob import glob

__file__path = os.path.abspath("")
print(__file__path)
sys.path.append(__file__path)
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

# Configuration
from omegaconf import DictConfig, OmegaConf
import hydra

config_path = "./configs"
config_name = "config.yaml"
try:
    hydra.initialize(version_base=None, config_path=config_path)
except:
    pass
cfg = hydra.compose(config_name=config_name)
print(OmegaConf.to_yaml(cfg))
import wandb

from models.id_emb import IdEmbedder
from models.generator import Generator
from models.discriminator import Discriminator


class FaceSwapModel(nn.Module):
    def __init__(self, G, E, D):
        super(FaceSwapModel, self).__init__()
        self.G = G
        self.D = D
        self.E = E

    def forward(self, x_src, x_tar):
        z_src = self.E(x_src)
        swap = self.G(x_tar, z_src)
        disc = self.D(swap)
        z_swap = self.E(swap)
        return z_src, swap, z_swap, disc

    def get_discriminant(self, x):
        return self.D(x)

    def get_id_embedding(self, x):
        return self.E(x)

    def get_swap_face(self, x_src, x_tar):
        temb = self.E(x_src)
        return self.G(x_tar, temb)


def cosine_metric(x1, x2):
    return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))


class CombineLoss(nn.Module):
    """
    SmoothSwap Combined loss
    """

    def __init__(self, lambdas=[4.0, 1.0, 1.0]):
        super(CombineLoss, self).__init__()
        self.lambdas = lambdas
        cos = nn.CosineSimilarity(dim=1, eps=1e-7)

    def forward(self, x_tar, z_src, swap, z_swap, disc):
        lambdas = [4.0, 1.0, 1.0]
        L_id = torch.mean(1 - cosine_metric(z_src, z_swap))
        L_chg = torch.mean((swap - x_tar).pow(2))
        #         L_adv = torch.mean(-1. * torch.log(disc))
        L_adv = 0.0
        combined_loss = lambdas[0] * L_id + lambdas[1] * L_chg + lambdas[2] * L_adv

        return combined_loss, L_id, L_chg, L_adv


# dataloader
class FFHQDataset(Dataset):
    def __init__(self, image_pool, transform=None):
        self.image_pool = image_pool
        self.transform = transform

    def __len__(self):
        return len(self.image_pool)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_src = Image.open(self.image_pool[idx])
        img_tar = Image.open(np.random.choice(self.image_pool))

        if self.transform:
            img_src = self.transform(img_src)
            img_tar = self.transform(img_tar)

        return img_src, img_tar


def test_FFHQDataset(image_pool):
    import matplotlib.pyplot as plt

    ffhq = FFHQDataset(image_pool)

    for i in np.random.randint(len(ffhq), size=3):
        sample = ffhq[i]

        plt.subplot(1, 2, 1)
        plt.imshow(sample[0])
        plt.subplot(1, 2, 2)
        plt.imshow(sample[1])
        plt.tight_layout()
        plt.show()


# train_on_batch
def train_epoch(
    epoch,
    train_loader,
    model,
    loss_fn,
    optimizer,
    optimizer_D,
    cuda,
    log_interval,
    metrics,
):

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, sample in enumerate(train_loader):
        img_src, img_tar = sample
        # add target target sample
        img_src[-1] = img_tar[-1]
        if cuda:
            img_src = img_src.cuda()
            img_tar = img_tar.cuda()

        # Discriminator
        # with torch.no_grad():
        img_fake = model.get_swap_face(img_src, img_tar)

        gen_logit = model.D(img_fake)
        loss_Dgen = (F.relu(torch.ones_like(gen_logit) + gen_logit)).mean()

        real_logit = model.D(img_tar)
        loss_Dreal = (F.relu(torch.ones_like(real_logit) - real_logit)).mean()
        loss_d = loss_Dgen + loss_Dreal
        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()

        # Swapper
        z_src, swap, z_swap, disc = model(img_src, img_tar)
        loss_outputs = loss_fn(img_tar, z_src, swap, z_swap, disc)
        loss = torch.mean(loss_outputs[0])
        losses.append(loss.item())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for metric in metrics:
        #     metric(outputs, loss_outputs)

        if batch_idx % log_interval == 0:
            message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                batch_idx * len(img_src[0]),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                np.mean(losses),
            )

            train_log = {
                "epoch": epoch,
                "batch": batch_idx,
                "total_l": loss_outputs[0],
                "id_l": loss_outputs[1],
                "chg_l": loss_outputs[2],
                "gan_l": loss_outputs[3],
            }
            wandb.log(train_log)

            message += "\t id_loss:{:.3f}, chg_loss:{:.3f}, gan_loss:{:.3f}".format(
                *loss_outputs[1:]
            )

            for metric in metrics:
                message += "\t{}: {}".format(metric.name(), metric.value())

            print(message, end="\r")
            losses = []

    total_loss /= batch_idx + 1
    return total_loss, metrics


def detransform_output(x):
    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    x = x * imagenet_std
    x = x + imagenet_mean
    return x.numpy()


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, sample in enumerate(val_loader):
            img_src, img_tar = sample
            if cuda:
                img_src = img_src.cuda()
                img_tar = img_tar.cuda()

            z_src, swap, z_swap, disc = model(img_src, img_tar)
            loss_outputs = loss_fn(img_tar, z_src, swap, z_swap, disc)

            loss = (
                loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            )
            val_loss += loss.item()
            # for metric in metrics:
            #     metric(outputs, img_tar, loss_outputs)

        img_src = detransform_output(img_src.to("cpu")).transpose(0, 2, 3, 1)
        img_tar = detransform_output(img_tar.to("cpu")).transpose(0, 2, 3, 1)
        swap = detransform_output(swap.to("cpu")).transpose(0, 2, 3, 1)

        table = wandb.Table(columns=["src", "tar", "swap"])
        for src, tar, swa in zip(img_src, img_tar, swap):
            table.add_data(
                wandb.Image(src * 255), wandb.Image(tar * 255), wandb.Image(swa * 255)
            )
        wandb.log({"swap_sample": table}, commit=False)

    return val_loss, metrics


def fit(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    optimizer_D,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    checkpoint_dir="./checkpoints/smoothswap",
    metrics=[],
    start_epoch=0,
):

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss, metrics = train_epoch(
            epoch,
            train_loader,
            model,
            loss_fn,
            optimizer,
            optimizer_D,
            cuda,
            log_interval,
            metrics,
        )
        scheduler.step()

        message = "Epoch: {}/{}. Train set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, train_loss
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += "\nEpoch: {}/{}. Validation set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, val_loss
        )
        val_log = {
            "epoch": epoch,
            "val_loss": val_loss,
        }
        wandb.log(val_log)

        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())
        print(message)
        state = {"epoch": epoch, "model": model, "optimizer": optimizer}
        filename = os.path.join(
            checkpoint_dir,
            "cpt_" + str(epoch) + "_" + "{:.4f}".format(val_loss) + ".tar",
        )
        torch.save(state, filename)
        print("{} is saved".format(filename))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data/ffhq/images1024x1024/")
    parser.add_argument("--n_epochs", default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--log_interval", default=5)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--lr_d", default=0.004)
    parser.add_argument("--margin", default=1.0)
    parser.add_argument("--id_emb_checkpoint_path", default="")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    log_interval = args.log_interval
    margin = args.margin
    lr = args.lr
    lr_d = args.lr_d
    id_emb_checkpoint_path = args.id_emb_checkpoint_path
    checkpoint_dir = args.checkpoint_dir
    cuda = torch.cuda.is_available()

    image_pool = glob(args.data_dir + "/*/*.png")
    test_FFHQDataset(image_pool)

    wandb.init(
        project="FaceSwap-SmoothSwap",
        config={
            "epochs": n_epochs,
            "batch_size": batch_size,
            "log_interval": log_interval,
            "margin": margin,
            "lr": lr,
            "lr_d": lr_d,
        },
    )

    G = Generator(cfg)
    D = Discriminator(cfg.discriminator.image_size)
    E = IdEmbedder(cfg)
    if id_emb_checkpoint_path:
        if cuda:
            E = torch.load(id_emb_checkpoint_path)["model"].embedding_net
        else:
            E = torch.load(id_emb_checkpoint_path, map_location=torch.device("cpu"))[
                "model"
            ].embedding_net
    E.requires_grad_(False)
    E.eval()

    if cuda:
        G.cuda()
        D.cuda()
        E.cuda()

    fsm = FaceSwapModel(G, E, D)
    if cuda:
        fsm.cuda()
    loss_fn = CombineLoss()

    # Dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_set = np.random.choice(image_pool, int(len(image_pool) * 0.9))
    test_set = np.array(list(set(image_pool) - set(train_set)))

    train_dataset = FFHQDataset(train_set, transform=transform)
    test_dataset = FFHQDataset(test_set, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    optimizer = optim.Adam(fsm.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, batch_size, gamma=0.1, last_epoch=-1
    )
    optimizer_D = optim.Adam(fsm.D.parameters(), lr=lr_d)

    fit(
        train_loader,
        test_loader,
        fsm,
        loss_fn,
        optimizer,
        optimizer_D,
        scheduler,
        n_epochs,
        cuda,
        log_interval,
        checkpoint_dir,
    )

    wandb.finish()
