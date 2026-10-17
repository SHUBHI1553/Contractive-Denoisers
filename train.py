import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pywt
import h5py
import random
from pytorch_wavelets import DWTForward, DWTInverse
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ssim(img1, img2, C1=0.01 ** 2, C2=0.03 ** 2):
    mu1 = img1.mean(dim=(2, 3), keepdim=True)
    mu2 = img2.mean(dim=(2, 3), keepdim=True)
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=(2, 3), keepdim=True)
    SSIM_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    SSIM_den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    SSIM = (SSIM_num / SSIM_den).mean()
    return 1 - SSIM  # loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        l1_loss = F.l1_loss(output, target)
        ssim_loss = ssim(output, target)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss


# -----patches -----
class PatchDataset(Dataset):
    def __init__(self, files, patch_size, stride, transform=None):
        self.files = files
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = []
        self.calculate_patches()

    def calculate_patches(self):
        for file in self.files:
            image = Image.open(file).convert("RGB")
            width, height = image.size
            for i in range(0, height, self.stride):
                for j in range(0, width, self.stride):
                    if i + self.patch_size <= height and j + self.patch_size <= width:
                        self.patches.append((file, i, j))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        file, i, j = self.patches[idx]
        image = Image.open(file).convert("RGB")  
        patch = image.crop((j, i, j + self.patch_size, i + self.patch_size))
        if self.transform:
            patch = self.transform(patch)
        return patch


# ----- dataloader -----
def create_dataloaders(train_dir, batch_size, patch_size, stride, val_split=0.1, seed=42):
    def is_valid_image(file_path):
        try:
            with Image.open(file_path).convert("RGB") as img:
                img.verify()
            return True
        except Exception:
            return False

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(0, 0)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.RandomRotation(degrees=(270, 270))
        ]),
        transforms.ToTensor()
    ])

    all_files = sorted([f for f in glob.glob(os.path.join(train_dir, "*")) if is_valid_image(f)])
    print(f"Found {len(all_files)} images in BSD500")

    train_files, val_files = train_test_split(all_files, test_size=val_split, random_state=seed)
    train_dataset = PatchDataset(train_files, patch_size, stride, transform)
    val_dataset = PatchDataset(val_files, patch_size, stride, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ----- core layers -----

class gradientlayer(nn.Module):
    def __init__(self, grad_in):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(grad_in))

    def forward(self, x, y):
        alpha_clamped = torch.clamp(self.alpha, 0.0, 0.9999)
        return (1 - alpha_clamped) * x + alpha_clamped * y


class proxlayer(nn.Module):
    def __init__(self, prox_in, wavefamily="db4", levels=5):
        super().__init__()
        
        self.lamda1 = nn.Parameter(prox_in[0] * torch.ones((1, 3, 3, 32, 32), dtype=torch.float32))
        self.lamda2 = nn.Parameter(prox_in[1] * torch.ones((1, 3, 3, 16, 16), dtype=torch.float32))
        self.lamda3 = nn.Parameter(prox_in[2] * torch.ones((1, 3, 3, 8, 8), dtype=torch.float32))

        
        self.softplus = nn.Softplus(beta=20, threshold=20)

        self.xfwt = DWTForward(J=levels, wave=pywt.Wavelet(wavefamily), mode="periodization").to(device)
        self.xinvfwt = DWTInverse(wave=pywt.Wavelet(wavefamily), mode="periodization").to(device)

        
        self.thresh = nn.ReLU()

    def forward(self, x, y):
        
        lam1 = 1e-6 + self.softplus(self.lamda1)
        lam2 = 1e-6 + self.softplus(self.lamda2)
        lam3 = 1e-6 + self.softplus(self.lamda3)

        
        (yl, yh) = self.xfwt(x)

        
        yhnew = [
            self.thresh(yh[0] - lam1) - self.thresh(-yh[0] - lam1),
            self.thresh(yh[1] - lam2) - self.thresh(-yh[1] - lam2),
            self.thresh(yh[2] - lam3) - self.thresh(-yh[2] - lam3),
        ]

       
        return self.xinvfwt((yl, yhnew))


class LipschitzConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape, alpha_ref,
                 kernel_size=3, stride=1, padding=1):
        super(LipschitzConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.input_shape = input_shape
        self.alpha_ref = alpha_ref  

    def forward(self, x):
        with torch.no_grad():
            self.project_to_1_lipschitz()

        alpha_val = torch.clamp(self.alpha_ref, 0.0, 0.9999)
        scale = 1.0 / (1.0 - alpha_val + 1e-6)
        return scale * self.conv(x)

    def project_to_1_lipschitz(self):
        W = self.conv.weight
        W_fft = torch.fft.fft2(W, s=self.input_shape)
        W_mat = W_fft.permute(2, 3, 0, 1).reshape(-1, self.conv.out_channels, self.conv.in_channels)
        max_singular = torch.linalg.norm(W_mat, ord=2, dim=(1, 2)).max()
        if max_singular > 1.0:
            self.conv.weight.data /= (max_singular + 1e-3)


class ProxLipBlock(nn.Module):
    """
    Block: LipConv_scaled( Prox( Gradient(z, y) ) )
    """
    def __init__(self, grad_layer, prox_module, lipconv_scaled):
        super().__init__()
        self.grad = grad_layer
        self.prox = prox_module
        self.lipconv = lipconv_scaled

    def forward(self, z, y):
        g = self.grad(z, y)
        p = self.prox(g, y)
        return self.lipconv(p)



class nonexpansivenn_with_conv(nn.Module):
    def __init__(self, prox_in, wavefamilyset, input_shape,
                 grad_in=0.1, levels=5, num_of_layers=10):
        super().__init__()
        self.layers = nn.ModuleList()
        C = 3
        for _ in range(num_of_layers):
            for wave in wavefamilyset:
                g = gradientlayer(grad_in)
                prox = proxlayer(prox_in, wave, levels)
                lip_scaled = LipschitzConv2d(C, C, input_shape, alpha_ref=g.alpha)
                block = ProxLipBlock(g, prox, lip_scaled)

                self.layers.append(block)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out, x)
        return out


# ----- metrics -----
def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i], Img[i], data_range=data_range)
    return (PSNR / Img.shape[0])


# ----- main -----
if __name__ == '__main__':
    train_dir = "./data/BSD500"
    batch_size = 256
    patch_size = 64
    sigma = 15
    input_shape = (64, 64)
    stride = 4
    val_split = 0.1

    loader_train, loader_val = create_dataloaders(train_dir, batch_size, patch_size, stride, val_split)
    sample = next(iter(loader_train))
    print("Sample batch shape:", sample.shape)
    print("Channels in batch:", sample.shape[1])

    no_layers = 10
    grad_in = 0.1
    levels = 3
    prox_in = [0.0001, 0.001, 0.001, 0.01, 0.1, 0.1]
    wavefamilyset = ["db4", "haar", "sym4"]
    lr = 0.0001
    num_epochs = 50

    model = nonexpansivenn_with_conv(
        prox_in=prox_in,
        wavefamilyset=wavefamilyset,
        input_shape=input_shape,
        grad_in=grad_in,
        levels=levels,
        num_of_layers=no_layers
    ).cuda()

   
    saved_model_path = "./outputs/checkpoints"
    os.makedirs(saved_model_path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedLoss(alpha=0.7).cuda()
    train_losses = []
    train_psnrs = []
    val_losses = []
    val_psnrs = []

    print("Training started !!!")

    for epoch in range(num_epochs):
        if epoch <= 10:
            current_lr = lr
        elif epoch <= 20:
            current_lr = lr / 10.
        else:
            current_lr = lr / 100.

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for i, data in enumerate(loader_train, 0):
            model.train()
            optimizer.zero_grad()

            img_train = data
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=sigma / 255.)
            imgn_train = img_train + noise
            imgn_train, img_train = imgn_train.to(device), img_train.to(device)

            out_train = model(imgn_train)
            loss = criterion(out_train, img_train)
            loss.backward()
            optimizer.step()

            model.eval()
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            if (i + 1) % 100 == 0:
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                      (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

        torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.
        psnr_val = 0.
        for j, val_data in enumerate(loader_val, 0):
            val_noise = torch.FloatTensor(val_data.size()).normal_(mean=0, std=sigma / 255.)
            val_noisydata = val_data + val_noise
            val_noisydata, val_data = val_noisydata.to(device), val_data.to(device)
            val_denoised = model(val_noisydata)
            val_loss_inter = criterion(val_denoised, val_data)
            psnr_val_inter = batch_PSNR(val_denoised, val_data, 1.)
            val_loss += val_loss_inter.item()
            psnr_val += psnr_val_inter

        print("[epoch %d] validation_loss: %.4f PSNR_validation: %.4f" %
              (epoch + 1, val_loss / (j + 1), psnr_val / (j + 1)))
        train_losses.append(loss.item())
        train_psnrs.append(psnr_train)
        val_losses.append(val_loss / (j + 1))
        val_psnrs.append(psnr_val / (j + 1))


        act_str = os.path.join(
            saved_model_path,
            f"sigma_{sigma}_epoch_{epoch + 1}.pth"
        )
        torch.save(model.state_dict(), act_str)
        
