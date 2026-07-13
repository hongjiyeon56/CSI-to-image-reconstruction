import json
import os
from glob import glob

import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


CSI_VALID_SUBCARRIER_INDEX = [i for i in range(6, 32)] + [i for i in range(33, 59)]
NUM_SUBCARRIERS = len(CSI_VALID_SUBCARRIER_INDEX)


class WificamDataset(Dataset):
    def __init__(self, base_dir, window_size, is_train=True):
        self.base_dir = base_dir
        self.window_size = window_size
        self.is_train = is_train

        self.csi_amplitudes = []
        self.image_paths = []
        self.load_data()

    def load_data(self):
        data_paths = glob(os.path.join(self.base_dir, '**', 'csi.csv'), recursive=True)
        for data_path in data_paths:
            data_dir = os.path.dirname(data_path)
            df = pd.read_csv(data_path).sort_values(by='id')

            raw_csi = df['data'].apply(json.loads).values
            raw_csi = np.array([np.array(x, dtype=np.int32) for x in raw_csi])
            real = raw_csi[:, [i * 2 for i in CSI_VALID_SUBCARRIER_INDEX]]
            imag = raw_csi[:, [i * 2 - 1 for i in CSI_VALID_SUBCARRIER_INDEX]]
            amplitude = np.sqrt(real**2 + imag**2).astype(np.float32)

            all_image_files = glob(os.path.join(data_dir, '*.png'))
            readable_id_to_path = {}
            for f in all_image_files:
                test_img = cv2.imread(f)
                if test_img is not None:
                    try:
                        img_id = int(os.path.basename(f).split('.')[0])
                        # Keep the original filename to preserve zero padding (e.g., 00942.png).
                        readable_id_to_path[img_id] = f
                    except ValueError:
                        continue

            if not readable_id_to_path:
                print(f'Warning: No valid images found in {data_dir}. Skipping this directory.')
                continue

            readable_image_ids = np.array(sorted(readable_id_to_path.keys()))

            num_samplies = len(amplitude) - self.window_size
            for i in range(num_samplies):
                target_id = df.iloc[i]['id'] + (self.window_size // 2)
                best_img_id = readable_image_ids[np.abs(readable_image_ids - target_id).argmin()]

                self.csi_amplitudes.append(amplitude[i:i + self.window_size])
                self.image_paths.append(readable_id_to_path[int(best_img_id)])
                
        self.csi_amplitudes = np.array(self.csi_amplitudes)

        # 1. log transform
        self.csi_amplitudes = np.log(self.csi_amplitudes + 1e-6)

        # 2. global normalization
        global_mean = self.csi_amplitudes.mean()
        global_std  = self.csi_amplitudes.std()

        self.csi_amplitudes = (self.csi_amplitudes - global_mean) / (global_std + 1e-6)

        # 3. clipping (안정화)
        self.csi_amplitudes = np.clip(self.csi_amplitudes, -3, 3)
    
    def __len__(self):
            return len(self.image_paths)
    
    def __getitem__(self, index):
        spectrogram = torch.from_numpy(self.csi_amplitudes[index]).float()

        # smoothing
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, T, S)
        spectrogram = torch.nn.functional.avg_pool2d(
            spectrogram,
            kernel_size=(1, 3),  # 시간 방향 1, 주파수 방향 3
            stride=1,
            padding=(0, 1)
        )
        spectrogram = spectrogram.squeeze()

        spectrogram = torch.clamp(spectrogram, -3, 3)

        # Noise Injection 추가
        if self.is_train:
            noise_std = 0.003  # 0.01 -> 0.003
            spectrogram = spectrogram + torch.randn_like(spectrogram) * noise_std

        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return spectrogram, image, image_path
