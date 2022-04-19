import torch
import torch.nn as nn
from model import AIA_Transformer, AIA_Transformer_merge, AHAM, AHAM_ori
from ptflops import get_model_complexity_info

class aia_irm_trans_ri(nn.Module):
    def __init__(self):
        super(aia_irm_trans_ri, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer(64, 64, num_layers=4)
        self.aham = AHAM(input_channel=64)


        self.de1 = dense_decoder()
        self.de2 = dense_decoder()



    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_r_input, x_i_input = x[:,0,:,:], x[:,1,:,:]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim = 1)
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x) #BCTF
        x_last , x_outputlist = self.dual_trans(x_ri) #BCTF, #BCTFG
        x_ri = self.aham(x_outputlist) #BCTF


        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim=1)
        x_com=torch.stack((x_real, x_imag), dim=1)


        return x_com



class dense_encoder(nn.Module):
    def __init__(self, width =64):
        super(dense_encoder, self).__init__()
        self.in_channels = 2
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x

class dense_encoder_mag(nn.Module):
    def __init__(self, width =64):
        super(dense_encoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x


class dense_decoder(nn.Module):
    def __init__(self, width =64):
        super(dense_decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width =width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        return out

class dense_decoder_masking(nn.Module):
    def __init__(self, width =64):
        super(dense_decoder_masking, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width =width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1,1)),
            nn.Sigmoid()
        )
        self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1))
        #self.maskrelu = nn.ReLU(inplace=True)
        self.maskrelu = nn.Sigmoid()
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        out = self.mask(out)
        out = self.maskrelu(self.maskconv(out))  # mask
        return out

class SPConvTranspose2d(nn.Module): #sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module): #dilated dense block
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

if __name__ == '__main__':
    model = aia_irm_trans_ri()
    model.eval()
    x = torch.FloatTensor(4, 2, 10, 161)
    x = model(x)
    macs, params = get_model_complexity_info(model, (2, 100, 161), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)