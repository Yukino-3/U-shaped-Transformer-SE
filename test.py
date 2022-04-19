import torch
import argparse
import os
import numpy as np
from istft import ISTFT
from transformer import aia_irm_trans_ri
import soundfile as sf
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
def _prepare_device(n_gpu: 4, cudnn_deterministic=False):
    """Choose to use CPU or GPU depend on "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        cudnn_deterministic (bool): repeatability
            cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of the experiment, set use_cudnn_deterministic to True
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        if cudnn_deterministic:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        #device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:{}" if torch.cuda.is_available() else "cpu")
    return device
def enhance(args):
    model = aia_irm_trans_ri()

    checkpoint = torch.load(args.Model_path)
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    #model.cuda()
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    model.to(device)

    # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path        
        file_list = os.listdir(mix_file_path)
        istft = ISTFT(filter_length=320, hop_length=160, window='hanning')
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                             esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

def enhance_ri(args):
    model = aia_irm_trans_ri()
    checkpoint = torch.load(args.Model_path)['model_state_dict']
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path        
        file_list = os.listdir(mix_file_path)
        istft = ISTFT(filter_length=320, hop_length=160, window='hanning')
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                             esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1



if __name__ == '__main__':

    
    
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/home/campus.ncl.ac.uk/b7015716/Pycharmproject/DNS-Challenge-master/testclips/')
    parser.add_argument('--esti_file_path', type=str, default='./estimated_audio/dns_nonblind_test/aia_merge_dns300_best')
    parser.add_argument('--snr', type=list, default=[-5, 0, 5, 10, 15, 20])     #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/vb_aia_merge_new.pth.tar',
                        help='The place to save best model')
    args = parser.parse_args()
    print(args)
    enhance(args=args)    
     
