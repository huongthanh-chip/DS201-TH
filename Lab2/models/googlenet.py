import torch
from torch import nn

class InceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1_x1: int,
            ch3_x3: int,
            ch3_x3_reduce: int,
            ch5_x5: int,
            ch5_x5_reduce: int,
            pool_proj: int,
    ):
        super().__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv2d (
                in_channels = in_channels,
                kernel_size = 1,
                out_channels = ch1_x1
            ),
            nn.ReLU()
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d (
                in_channels = in_channels,
                kernel_size = 1,
                out_channels = ch3_x3_reduce
            ),
            nn.ReLU(),

            nn.Conv2d (
                in_channels = ch3_x3_reduce,
                kernel_size = 3,
                out_channels = ch3_x3,
                padding = 1,
            ),
            nn.ReLU()
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d (
                in_channels = in_channels,
                kernel_size = 1,
                out_channels = ch5_x5_reduce
            ),
            nn.ReLU(),

            nn.Conv2d (
                in_channels = ch5_x5_reduce,
                out_channels = ch5_x5,
                kernel_size = 5,
                padding = 2
            ),
            nn.ReLU()
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d (
                kernel_size = 3,
                stride = 1,
                ceil_mode = True,
                padding = 1
            ),

            nn.Conv2d (
                in_channels = in_channels,
                kernel_size = 1,
                out_channels = pool_proj
            ),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.branch_1(x)
        out2 = self.branch_2(x)
        out3 = self.branch_3(x)
        out4 = self.branch_4(x)

        output = torch.cat(
            [out1, out2, out3, out4],
            dim = 1
        )

        return output

class GoogLeNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int
    ):
        super().__init__()
        self.conv_1 = nn.Sequential (
            nn.Conv2d (
                in_channels = in_channels,
                out_channels = 64, #(B,64,112,112)
                kernel_size = 7,
                stride = 2,
                padding = 3
            ),
            nn.ReLU()
        )

        self.maxp_2 = nn.MaxPool2d ( #(B,64,56,56)
            kernel_size = 3,
            stride = 2,
            ceil_mode = True
        )

        self.conv_3 = nn.Sequential (
            nn.Conv2d (
                in_channels = 64,
                out_channels = 64, #(B,64,56,56)
                kernel_size = 1,
                stride = 1
            ),
            nn.ReLU()
        )

        self.conv_4 = nn.Sequential (
            nn.Conv2d (
                in_channels = 64,
                out_channels = 192, #(B,192,56,56)
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU()
        )

        self.maxp_5 = nn.MaxPool2d ( #(B,192,28,28)
            kernel_size = 3,
            stride = 2,
            ceil_mode = True
        )

        self.inception_3a = InceptionBlock( #(B,256,28,28)
            in_channels = 192,
            ch1_x1 = 64,
            ch3_x3 = 128,
            ch3_x3_reduce = 96,
            ch5_x5 = 32,
            ch5_x5_reduce = 16,
            pool_proj = 32
        )

        self.inception_3b = InceptionBlock( #(B,480,28,28)
            in_channels=256,
            ch1_x1=128,
            ch3_x3_reduce=128,
            ch3_x3=192,
            ch5_x5_reduce=32,
            ch5_x5=96,
            pool_proj=64
        )

        self.maxp_7 = nn.MaxPool2d ( #(B,480,14,14)
            kernel_size = 3,
            stride = 2,
            ceil_mode = True
        )

        self.inception_4a = InceptionBlock( #(B,512,14,14)
            in_channels=480,
            ch1_x1=192,
            ch3_x3_reduce=96,
            ch3_x3=208,
            ch5_x5_reduce=16,
            ch5_x5=48,
            pool_proj=64
        )

        self.inception_4b = InceptionBlock( #(B,512,14,14)
            in_channels=512,
            ch1_x1=160,
            ch3_x3_reduce=112,
            ch3_x3=224,
            ch5_x5_reduce=24,
            ch5_x5=64,
            pool_proj=64
        )

        self.inception_4c = InceptionBlock( #(B,512,14,14)
            in_channels=512,
            ch1_x1=128,
            ch3_x3_reduce=128,
            ch3_x3=256,
            ch5_x5_reduce=24,
            ch5_x5=64,
            pool_proj=64
        )

        self.inception_4d = InceptionBlock( #(B,528,14,14)
            in_channels=512,
            ch1_x1=112,
            ch3_x3_reduce=144,
            ch3_x3=288,
            ch5_x5_reduce=32,
            ch5_x5=64,
            pool_proj=64
        )

        self.inception_4e = InceptionBlock( #(B,832,14,14)
            in_channels=528,
            ch1_x1=256,
            ch3_x3_reduce=160,
            ch3_x3=320,
            ch5_x5_reduce=32,
            ch5_x5=128,
            pool_proj=128
        )

        self.maxp_8 = nn.MaxPool2d( #(B,832,7,7)
            kernel_size=3,
            stride=2,
            ceil_mode=True
        )

        self.inception_5a = InceptionBlock( #(B,832,7,7)
            in_channels=832,
            ch1_x1=256,
            ch3_x3_reduce=160,
            ch3_x3=320,
            ch5_x5_reduce=32,
            ch5_x5=128,
            pool_proj=128
        )

        self.inception_5b = InceptionBlock( #(B,1024,7,7)
            in_channels=832,
            ch1_x1=384,
            ch3_x3_reduce=192,
            ch3_x3=384,
            ch5_x5_reduce=48,
            ch5_x5=128,
            pool_proj=128
        )

        self.avgp_9 = nn.AvgPool2d( #(B,1024,1,1)
            kernel_size=7,
            stride=1
        )

        self.dropout = nn.Dropout(p=0.4)

        self.fc = nn.Linear(
            in_features=1024,
            out_features=num_classes
        )

    def forward(self, images: torch.Tensor):
        """
        images: (B, in_channels, 224, 224)
        """
        features = self.conv_1(images)
        features = self.maxp_2(features)
        features = self.conv_3(features)
        features = self.conv_4(features)
        features = self.maxp_5(features)

        features = self.inception_3a(features)
        features = self.inception_3b(features)
        features = self.maxp_7(features)

        features = self.inception_4a(features)
        features = self.inception_4b(features)
        features = self.inception_4c(features)
        features = self.inception_4d(features)
        features = self.inception_4e(features)
        features = self.maxp_8(features)

        features = self.inception_5a(features)
        features = self.inception_5b(features)

        features = self.avgp_9(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)

        logits = self.fc(features)
        return logits
