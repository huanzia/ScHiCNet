import os, sys
sys.path.append(".")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from math import log10

from torch.utils.tensorboard import SummaryWriter
# import wandb

from Utils.loss.schicnet_loss import GeneratorLoss_v4 as G1_loss
from Utils.loss.SSIM import ssim
from Models.schicnet import schicnet_Block
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module
from ProcessData.PrepareData_tensorMouse import GSE162511Module


class schicnet_trainer:
    def __init__(self, epoch=300, batch_s=1, cellN=1, celline='Human', percentage=0.75):
        self.epochs = epoch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = schicnet_Block().to(self.device)
        # self.model.init_params()

        self.cell_line = celline
        self.cell_no = cellN
        self.ratio = f"Downsample_{percentage}"
        self.runs = f"{self.cell_line}{self.cell_no}_{percentage}"

        # you con record your experiment using wandb
        # wandb.init(
        #     project='schicnet',
        #     settings=wandb.Settings(init_timeout=150)  # Set to 120 seconds or higher
        # )

        # wandb.run.name = f'edsrv6_{celline}_cell_{cellN}_percentage_{percentage}_batch_{batch_s}'
        # wandb.run.save()

        root_dir = '../pretrained'
        self.out_dir = os.path.join(root_dir, f"{self.ratio}_{self.cell_line}{self.cell_no}")
        self.out_dirM = os.path.join(self.out_dir, 'metrics')
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.out_dirM, exist_ok=True)

        if self.cell_line == 'Human':
            DataModule = GSE130711Module(batch_size=batch_s, cell_No=cellN, percent=percentage)
        elif self.cell_line == 'Dros':
            DataModule = GSE131811Module(batch_size=batch_s, cell_No=cellN, percent=percentage)
        elif self.cell_line == 'Mouse':
            DataModule = GSE162511Module(batch_size=batch_s, cell_No=cellN, percent=percentage)
        else:
            raise ValueError("Unsupported cell line.")

        DataModule.prepare_data()
        DataModule.setup(stage='fit')
        self.train_loader = DataModule.train_dataloader()
        self.valid_loader = DataModule.val_dataloader()

        self.criterion = G1_loss(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.tb = SummaryWriter(log_dir=os.path.join('runs', self.runs))

    def fit_model(self):
        best_ssim = 0
        best_psnr = 0
        ssim_scores, psnr_scores, mse_scores, mae_scores = [], [], [], []

        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch(epoch)
            valid_metrics = self.validate_one_epoch(epoch)

            if valid_metrics['ssim'] > best_ssim:
                best_ssim = valid_metrics['ssim']

                torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"bestg_40kb_c40_s40_{self.cell_line}{self.cell_no}_edsrv6Net.pth"))
                print(f"New best SSIM: {best_ssim:.4f}")

            if valid_metrics['ssim'] == best_ssim and valid_metrics['psnr'] > best_psnr:
                best_ssim = valid_metrics['ssim']
                torch.save(self.model.state_dict(), os.path.join(self.out_dir,
                                                                 f"bestg_40kb_c40_s40_{self.cell_line}{self.cell_no}_edsrv6Net.pth"))
                print(f"New best SSIM: {best_ssim:.4f}")

            if valid_metrics['psnr'] > best_psnr:
                best_psnr = valid_metrics['psnr']
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"bestg_psnr_40kb_c40_s40_{self.cell_line}{self.cell_no}_edsrv6Net.pth"))
                print(f"New best PSNR: {best_psnr:.4f}")

            ssim_scores.append(float(valid_metrics['ssim'].cpu()))
            psnr_scores.append(float(valid_metrics['psnr']))
            mse_scores.append(float(valid_metrics['mse'].cpu()))
            mae_scores.append(float(valid_metrics['mae'].cpu()))

        # Save final model & metrics
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"finalg_{self.cell_line}{self.cell_no}_edsrv6Net.pth"))
        np.savetxt(os.path.join(self.out_dirM, "ssim.txt"), np.array(ssim_scores), fmt="%.6f", delimiter=',')
        np.savetxt(os.path.join(self.out_dirM, "psnr.txt"), np.array(psnr_scores), fmt="%.6f", delimiter=',')
        np.savetxt(os.path.join(self.out_dirM, "mse.txt"), np.array(mse_scores), fmt="%.6f", delimiter=',')
        np.savetxt(os.path.join(self.out_dirM, "mae.txt"), np.array(mae_scores), fmt="%.6f", delimiter=',')

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0
        bar = tqdm(self.train_loader, desc=f"[Epoch {epoch}] Training")

        for lr, hr, _ in bar:
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.model(lr)

            loss_dict = self.criterion(sr, hr)
            loss = loss_dict['total']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * lr.size(0)
            bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(self.train_loader.dataset)
        self.tb.add_scalar("train/loss", avg_loss, epoch)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        valid_result = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'mae': 0, 'nsamples': 0}
        total_loss = 0

        with torch.no_grad():
            for lr, hr, _ in tqdm(self.valid_loader, desc=f"[Epoch {epoch}] Validation"):
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.model(lr)

                loss_dict = self.criterion(sr, hr)
                loss = loss_dict['total']
                total_loss += loss.item() * lr.size(0)

                mse = ((sr - hr) ** 2).mean()
                mae = (sr - hr).abs().mean()
                ssim_score = ssim(sr, hr)

                valid_result['mse'] += mse * lr.size(0)
                valid_result['mae'] += mae * lr.size(0)
                valid_result['ssims'] += ssim_score * lr.size(0)
                valid_result['nsamples'] += lr.size(0)

        valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
        valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']

        self.tb.add_scalar("valid/loss", total_loss / valid_result['nsamples'], epoch)
        self.tb.add_scalar("valid/ssim", valid_result['ssim'], epoch)
        self.tb.add_scalar("valid/psnr", valid_result['psnr'], epoch)

        # wandb.log({"epoch": epoch, "ssim": valid_result['ssim'], "psnr": valid_result['psnr']})
        print(f"Validation Results — Epoch {epoch} | PSNR: {valid_result['psnr']:.4f}, SSIM: {valid_result['ssim']:.4f}")

        return valid_result


if __name__ == "__main__":
    trainer = schicnet_trainer(epoch=300, batch_s=1, cellN=1, percentage=0.75, celline='Mouse')
    trainer.fit_model()
    print("\nTraining complete.\n")

