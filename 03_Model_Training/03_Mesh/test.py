import os
from pathlib import Path
import numpy as np
import torch
import cv2
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE

num_workers = 2
torch.set_num_threads(4)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent
project_root = current_folder.parent

test_dir = os.path.join(project_root, 'data', 'data_2026_01_20_mesh')
output_dir = os.path.join(current_folder, 'outputs')
output_video_file = os.path.join(output_dir, 'output.mp4')

fps = 10
step = 10
size = (256, 128)  # 반반 나란히라서 width 2배
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
checkpoint_path = os.path.join(output_dir, 'best_vae-epoch=185-val_loss=394.0456.ckpt')
image_dir = os.path.join(output_dir, 'images')
os.makedirs(image_dir, exist_ok=True)

window_size = 151
batch_size = 32
persistent_workers = True if num_workers > 0 else False

def test():
    dataset = WificamDataset(test_dir, window_size)
    _, test_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=False)
    dataset_test = Subset(dataset, test_idx)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    out = cv2.VideoWriter(output_video_file, fourcc, fps, size)

    model = VAE.load_from_checkpoint(
        checkpoint_path,
        window_size=window_size,
        num_subcarriers=NUM_SUBCARRIERS,
    )
    model.to(device)
    model.eval()

    idx = 0
    for batch in tqdm(dataloader_test):
        spectrogram, image = batch
        spectrogram = spectrogram.to(device)
        image = image.to(device)

        with torch.no_grad():
            reconstruction, mu, logvar = model(spectrogram)

        image = image.permute(0, 2, 3, 1).cpu().numpy()
        reconstruction = reconstruction.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(len(reconstruction)):
            data_content = (np.clip(image[i][..., ::-1], 0, 1) * 255).astype(np.uint8)
            pred_content = (np.clip(reconstruction[i][..., ::-1], 0, 1) * 255).astype(np.uint8)

            # 원본 | 복원 나란히 붙이기
            combined = np.concatenate([data_content, pred_content], axis=1)

            idx += 1
            if idx % step == 0:
                out.write(combined)

    out.release()
    print(f"✅ Video saved to: {output_video_file}")

if __name__ == '__main__':
    test()
