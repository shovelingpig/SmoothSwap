"""
On VGG Face2
"""
import os
import sys
from glob import glob

# 이 파일의 Directory의 절대 경로
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

######### Configuration ##########
from omegaconf import DictConfig, OmegaConf
import hydra

config_path = "./configs"
config_name = "config.yaml"
try:
    hydra.initialize(version_base=None, config_path=config_path)
except:
    pass
cfg = hydra.compose(config_name=config_name)

# torch random 값 생성
random_seed = 777
torch.manual_seed(random_seed)
np.random.seed(random_seed)
import random

random.seed(random_seed)


def get_train_test_set_vggface(data_dir, ratio_of_test=0.3):
    """
    example
        data_dir = "../../data/vggface2_crop_arcfacealign_224/"
    """
    all_pictures = glob(data_dir + "/*/*.jpg")

    test_size = int(len(all_pictures) * ratio_of_test)

    test_set = np.random.choice(all_pictures, size=test_size)
    train_set = list(set(all_pictures) - set(test_set))

    return np.array(train_set), test_set


def get_label_dict_vggface(data_dir):
    all_label = glob(data_dir + "/*")
    return {os.path.basename(name): label for label, name in enumerate(all_label)}


class VGGFace2Dataset(Dataset):
    def __init__(self, data_set, label_dict, transform=None):
        self.data_set = data_set
        self.label_dict = label_dict
        self.label_dict_inv = {v: k for k, v in label_dict.items()}
        self.label_keys = list(label_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_set[idx]

        img_dir = os.path.dirname(img_path)
        id_idx = os.path.basename(img_dir)
        label = self.label_dict[id_idx]

        # random pick positive
        pos_path = np.random.choice(glob(img_dir + "/*.jpg"))
        # random pick negative
        neg_idx = np.random.choice([x for x in self.label_keys if x != id_idx])
        neg_path = np.random.choice(
            glob(os.path.dirname(img_dir) + "/" + str(neg_idx) + "/*.jpg")
        )

        image = Image.open(img_path)
        posit = Image.open(pos_path)
        negat = Image.open(neg_path)

        img1 = image
        img2 = posit
        img3 = negat

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), label


def test_VGGFace2Dataset(data_set, label_dict):
    import matplotlib.pyplot as plt

    vgg_dataset = VGGFace2Dataset(data_set, label_dict)

    for i in np.random.randint(len(vgg_dataset), size=3):
        sample, label = vgg_dataset[i]

        plt.subplot(1, 3, 1)
        plt.title(vgg_dataset.label_dict_inv[label])
        plt.imshow(sample[0])
        plt.subplot(1, 3, 2)
        plt.imshow(sample[1])
        plt.subplot(1, 3, 3)
        plt.imshow(sample[2])
        plt.tight_layout()
        plt.show()


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(0.5)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(0.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, sample in enumerate(train_loader):
        data, label = sample
        if cuda:
            data = tuple(d.cuda() for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        loss_outputs = loss_fn(*outputs)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                batch_idx * len(data[0]),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                np.mean(losses),
            )
            for metric in metrics:
                message += "\t{}: {}".format(metric.name(), metric.value())

            print(message, end="\r")
            losses = []

    total_loss /= batch_idx + 1
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if cuda:
                data = tuple(d.cuda() for d in data)

            outputs = model(*data)

            loss_inputs = outputs
            loss_outputs = loss_fn(*loss_inputs)

            loss = (
                loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            )
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def fit(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    checkpoint_dir="./checkpoints/id_emb",
    metrics=[],
    start_epoch=0,
):

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics
        )

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
    parser.add_argument("--data_dir", default="../vggface2_crop_arcfacealign_224")
    parser.add_argument("--n_epochs", default=500)
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", default=0.0003)
    parser.add_argument("--log_interval", default=5)
    parser.add_argument("--margin", default=1.0)
    args = parser.parse_args()

    data_dir = args.data_dir
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    log_interval = args.log_interval
    margin = args.margin
    cuda = torch.cuda.is_available()

    # dataset
    label_dict = get_label_dict_vggface(data_dir)
    train_set, test_set = get_train_test_set_vggface(data_dir, 0.2)
    # test_VGGFace2Dataset(train_set, label_dict)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = VGGFace2Dataset(train_set, label_dict, transform=transform)
    test_dataset = VGGFace2Dataset(test_set, label_dict, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    from models.id_emb import IdEmbedder

    embedder = IdEmbedder(cfg)

    model = TripletNet(embedder)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(
        train_loader,
        test_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        cuda,
        log_interval,
    )
