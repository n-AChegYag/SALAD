import torch.nn as nn
import torch

##############################
#        Blocks
##############################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Down(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(Down, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Up, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


##############################
#        Generator
##############################

class ME(nn.Module):

    def __init__(self, in_channels=1, out_channels=512):
        super().__init__()

        self.down1 = Down(in_channels, 64, normalize=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512, dropout=0.5)
        self.down5 = Down(512, 512, dropout=0.5)
        self.down6 = Down(512, 512, dropout=0.5)
        self.down7 = Down(512, 512, dropout=0.5)
        self.down8 = Down(512, out_channels, normalize=False, dropout=0.5)

    def forward(self, x):

        x = self.down1(x)       # 256 -> 128
        x = self.down2(x)       # 128 -> 64
        x = self.down3(x)       # 64 -> 32
        x = self.down4(x)       # 32 -> 16
        x = self.down5(x)       # 16 -> 8
        x = self.down6(x)       # 8 -> 4
        x = self.down7(x)       # 4 -> 2
        x = self.down8(x)       # 2 -> 1

        return x


class MD(nn.Module):

    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()

        self.up1 = Up(in_channels, 512, dropout=0.5)
        self.up2 = Up(512, 512, dropout=0.5)
        self.up3 = Up(512, 512, dropout=0.5)
        self.up4 = Up(512, 512, dropout=0.5)
        self.up5 = Up(512, 256)
        self.up6 = Up(256, 128)
        self.up7 = Up(128, 64)
        self.up8 = Up(64, 1)

    def forward(self, x):

        x = self.up1(x)       # 1 -> 2
        x = self.up2(x)       # 2 -> 4
        x = self.up3(x)       # 4 -> 8
        x = self.up4(x)       # 8 -> 16
        x = self.up5(x)       # 16 -> 32
        x = self.up6(x)       # 32 -> 64
        x = self.up7(x)       # 64 -> 128
        x = self.up8(x)       # 128 -> 256

        return x


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)


class MLP(nn.Module):
    def __init__(self, in_channels=512):
        super(MLP, self).__init__()

        self.input_fc = nn.Linear(in_channels, 256)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.hidden_fc = nn.Linear(256, 128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.output_fc = nn.Linear(128, 1)

    def forward(self, x):

        x = self.input_fc(x)
        x = self.relu1(x)
        x = self.hidden_fc(x)
        x = self.relu2(x)
        x = self.output_fc(x)

        return x


if __name__ == '__main__':

    import os
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    m_e = ME()
    m_d = MD()
    d_i = Discriminator(1)
    d_f = MLP(512)

    m_e.cuda()
    m_d.cuda()
    d_i.cuda()
    d_f.cuda()

    x = np.zeros((2, 1, 256, 256)).astype('float32')
    x = torch.from_numpy(x)
    x = x.cuda()

    z = np.zeros((2, 512, 1, 1)).astype('float32')
    z = torch.from_numpy(z)
    z = z.cuda()

    z_p = m_e(x)
    x_h = m_d(z_p)
    rf_f = d_f(z_p.squeeze())
    x_p = m_d(z)
    z_h = m_e(x_p)
    rf_i = d_i(x_p)

    print(
        'x_h size: {}'.format(x_h.shape),
        'z_p size: {}'.format(z_p.shape),
        'rf_f size: {}'.format(rf_f.shape),
        'z_h size: {}'.format(z_h.shape),
        'x_p size: {}'.format(x_p.shape),
        'rf_i size: {}'.format(rf_i.shape),
        sep='\n'
    )