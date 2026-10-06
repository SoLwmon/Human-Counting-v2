"""OSNet person re-identification model (architecture from torchreid, MIT licence).

Turns a person crop into a 512-number "appearance fingerprint". Two crops of the same
person give similar fingerprints; different people give different ones.
Weights: osnet_x0_25_msmt17.pt (pretrained, ~3 MB).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Conv1x1(ConvLayer):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 1)


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True)

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(g))))


class OSBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        mid = cout // 4
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = sum(self.gate(branch(x1)) for branch in (self.conv2a, self.conv2b, self.conv2c, self.conv2d))
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.conv3(x2) + identity)


def _layer(cin, cout, reduce):
    layers = [OSBlock(cin, cout), OSBlock(cout, cout)]
    if reduce:
        layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
    return nn.Sequential(*layers)


class OSNet(nn.Module):
    def __init__(self, channels=(16, 64, 96, 128), feature_dim=512):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.conv1 = ConvLayer(3, c0, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = _layer(c0, c1, True)
        self.conv3 = _layer(c1, c2, True)
        self.conv4 = _layer(c2, c3, False)
        self.conv5 = Conv1x1(c3, c3)
        self.fc = nn.Sequential(nn.Linear(c3, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class ReID:
    """Wraps OSNet: give it a frame + person boxes, get back unit-length fingerprints."""
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights="osnet_x0_25_msmt17.pt"):
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items() if not k.startswith("classifier")}
        # Model size (x0_25 / x0_5 / x0_75 / x1_0) is read from the weights: first layer has 16/32/48/64 channels
        width = state["conv1.conv.weight"].shape[0] / 64
        self.model = OSNet(channels=tuple(int(c * width) for c in (64, 256, 384, 512)))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def embed(self, frame, boxes):
        if len(boxes) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        h, w = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = int(max(x1, 0)), int(max(y1, 0))
            x2, y2 = int(min(x2, w)), int(min(y2, h))
            crop = frame[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)]
            crop = cv2.resize(crop, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
            crops.append((crop - self.MEAN) / self.STD)
        x = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2).copy())
        feats = F.normalize(self.model(x), dim=1)
        return feats.numpy()
