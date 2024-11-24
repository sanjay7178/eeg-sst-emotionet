from torch import nn, Tensor
from einops.layers.torch import Rearrange
from models.conformer import TransformerEncoder, ClassificationHead


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class EEGConformer(nn.Module):
    def __init__(self, num_classes=8, emb_size=40, depth=6):
        super(EEGConformer, self).__init__()
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        self.classification_head = ClassificationHead(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x, out = self.classification_head(x)
        return out
