import json, os, cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

CSI_VALID_SUBCARRIER_INDEX = [i for i in range(6, 32)] + [i for i in range(33, 59)]
NUM_SUBCARRIERS = len(CSI_VALID_SUBCARRIER_INDEX)

class WificamDataset(Dataset):
    def __init__(self, base_dir, window_size):
        self.base_dir, self.window_size = base_dir, window_size
        self.csi_amplitudes, self.image_paths = [], []
        self.load_data()
    
    def load_data(self):
        csv_paths = glob(os.path.join(self.base_dir, '**', 'csi.csv'), recursive=True)
        for path in csv_paths:
            d_dir = os.path.dirname(path)
            df = pd.read_csv(path).sort_values('id')
            raw = np.array([json.loads(x) for x in df['data'].values])
            
            real = raw[:, [i * 2 for i in CSI_VALID_SUBCARRIER_INDEX]]
            imag = raw[:, [i * 2 - 1 for i in CSI_VALID_SUBCARRIER_INDEX]]
            amp = np.sqrt(real**2 + imag**2).astype(np.float32)
            
            img_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in glob(os.path.join(d_dir, '*.png'))])
            img_ids = np.array(img_ids)

            for i in range(len(amp) - self.window_size):
                target = df.iloc[i + self.window_size//2]['id']
                best_img = img_ids[np.abs(img_ids - target).argmin()]
                self.csi_amplitudes.append(amp[i:i+self.window_size])
                self.image_paths.append(os.path.join(d_dir, f'{best_img}.png'))

    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        csi = torch.from_numpy(self.csi_amplitudes[idx])
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (128, 128))
        return csi, torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
