from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int
    dataset_type: str = "vocos"
    stride_size: int = None
    window_size: int = None


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset_class = {"vocos": VocosDataset, "vocos_cache": VocosCacheDataset}.get(cfg.dataset_type, VocosDataset)
        dataset = dataset_class(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return {"audio": y[0]}


class VocosCacheDataset(Dataset):
    """
    Dataset that loads audio and provides cache information for CachedFeatures.
    Filters files based on cached audio_length vs num_samples requirement.
    """
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.sampling_rate == 16000, "VocosCacheDataset requires sampling_rate=16000"

        self.sampling_rate = 16000  # Always 16kHz
        self.num_samples = cfg.num_samples
        self.stride_size = cfg.stride_size
        self.window_size = cfg.window_size
        self.cache_dir = cfg.filelist_path
        self.train = train

        # Load audio_length metadata file
        cache_dir_path = Path(self.cache_dir)
        metadata_path = cache_dir_path / "meta.json"
        with open(metadata_path, "r") as f:
            self.meta = json.load(f)

        # Filter filelist based on cached audio_length from metadata
        self.filelist = []
        self.audio_lengths = {}

        for cache_id, row in self.meta.items():
            if row["length"] >= self.num_samples:
                self.filelist.append(cache_id)

        print(f"VocosCacheDataset: Filtered {len(self.filelist)} / Original {len(self.meta)} (required length: >= {self.num_samples})")

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int):
        cache_id = self.filelist[index]
        feat_length = self.meta[cache_id]["length"]
        audio_path = self.meta[cache_id]["audio_path"]

        # Determine start and end indices
        if feat_length < self.num_samples:
            # Should not happen due to filtering, but handle gracefully
            start_index = 0
            end_index = feat_length
        elif self.train:
            # Random crop during training
            start_index = np.random.randint(low=0, high=feat_length - self.num_samples + 1)
            end_index = start_index + self.num_samples
        else:
            # During validation, take always the first segment for determinism
            start_index = 0
            end_index = self.num_samples

        # Load and process audio for discriminator
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        audio_start = start_index * self.stride_size
        audio_end = audio_start + self.window_size + (self.num_samples - 1) * self.stride_size
        audio = waveform[0][audio_start:audio_end]

        cache_path = Path(self.cache_dir) / f"{cache_id}.pt"
        feats = torch.load(cache_path, map_location="cpu")
        feats = feats[start_index:end_index, :]

        return {
            "audio": audio,
            "feature": feats,
        }
