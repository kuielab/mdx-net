import torch
import torch.nn as nn


class Conv_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, bias=True):

        super(Conv_TDF, self).__init__()

        self.use_tdf = bn is not None

        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )
            )

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU()
                )

    def forward(self, x):
        for h in self.H:
            x = h(x)

        return x + self.tdf(x) if self.use_tdf else x


class TFC_TDF(nn.Module):
    def __init__(self, in_c, l, g, f, bn, bias=True):

        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None

        c = in_c
        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=g, kernel_size=k, stride=1, padding=k // 2),
                    nn.BatchNorm2d(g),
                    nn.ReLU(),
                )
            )
            c += g

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.BatchNorm2d(g),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.BatchNorm2d(g),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.BatchNorm2d(g),
                    nn.ReLU()
                )

    def forward(self, x):
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)

        return x_ + self.tdf(x_) if self.use_tdf else x_

