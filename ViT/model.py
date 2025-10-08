import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                      kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # patch embedding
        x = self.patch(x)  # [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, embed_dim]

        # class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, embed_dim]

        # position + dropout
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim,
                 num_patches, dropout, num_head, activation,
                 num_encoders, num_classes):
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_head,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)  # [B, num_patches+1, embed_dim]
        x = self.encoder_layer(x)
        x = self.mlp_head(x[:, 0, :])  # use [CLS] token as input
        return x

if __name__ == "__main__":
    # 模拟一个输入
    x = torch.randn(8, 1, 28, 28)  # batch=8, 灰度图

    # 计算 patch 数量
    patch_size = 7
    num_patches = (28 // patch_size) ** 2

    model = ViT(in_channels=1, patch_size=patch_size, embed_dim=64,
                num_patches=num_patches, dropout=0.1, num_head=4,
                activation='relu', num_encoders=2, num_classes=10)

    y = model(x)
    print(y.shape)

