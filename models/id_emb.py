"""
Author : conor.k
Date   : 8/1, 2022
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

# Configuration
from omegaconf import DictConfig, OmegaConf
import hydra

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocessor = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class IdEmbedder(nn.Module):
    def __init__(self, args):
        super(IdEmbedder, self).__init__()

        if "resnet" in args.id_emb.network:
            resnet = torch.hub.load(
                "pytorch/vision:v0.10.0",
                args.id_emb.network,
                pretrained=args.id_emb.train_embedder,
            )
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc1 = nn.Linear(args.id_emb.emb_size * 4, 512 * 2)
        self.bn1 = nn.BatchNorm1d(args.id_emb.emb_size * 2)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Linear(args.id_emb.emb_size * 2, args.id_emb.emb_size)
        self.bn2 = nn.BatchNorm1d(args.id_emb.emb_size)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.normalize(x)
        return x


config_path = "../configs"
config_name = "config.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def test_IdEmbedder(cfg):

    embedder = IdEmbedder(cfg)
    sample = torch.rand((2, 3, cfg.id_emb.image_size, cfg.id_emb.image_size))
    out = embedder(sample)
    print(out.shape)


if __name__ == "__main__":
    print("#" * 7, " Id Embedder Test ", "*" * 7)
    # BUILD TEST
    test_IdEmbedder()
