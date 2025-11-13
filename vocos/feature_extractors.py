from typing import List

import torch
import torchaudio
from encodec import EncodecModel
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from torch import nn, no_grad

from vocos.modules import safe_log


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class SSLFeatures(FeatureExtractor):
    def __init__(
        self,
        ssl_model: str = "microsoft/wavlm-large",
        layer: int = -1,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(ssl_model)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model)
        self.layer = layer

    def forward(self, audio: torch.Tensor, **kwargs):
        x = self.processor(raw_speech=audio.cpu().numpy(), sampling_rate=16000, padding=True, return_tensors="pt")
        with no_grad():
            outputs = self.model(**{k: t.to(audio.device) for k, t in x.items()})
        if self.layer == -1:
            return outputs.last_hidden_state.detach().transpose(1, 2)
        else:
            outputs = self.model(output_hidden_states=True, **{k: t.to(audio.device) for k, t in x.items()})
            return outputs.hidden_states[self.layer].detach().transpose(1, 2)


class CachedFeatures(FeatureExtractor):
    def forward(self, batch_dict: dict, **kwargs):
        """
        Load cached features from disk based on cache_path and indices.

        Args:
            batch_dict: Dictionary containing:
                - "cache_path": str or list of str (full paths to cached feature files)
                - "start_index": int or list of int (start indices)
                - "end_index": int or list of int (end indices)

        Returns:
            Tensor: Features of shape (B, C, L) where B is batch size,
                   C is hidden_dim, L is sequence length
        """
        cache_paths = batch_dict.get("cache_path")
        start_indices = batch_dict.get("start_index")
        end_indices = batch_dict.get("end_index")

        # Handle single sample vs batched
        if isinstance(cache_paths, str):
            cache_paths = [cache_paths]
            start_indices = [start_indices]
            end_indices = [end_indices]

        features_list = []
        for cache_path, start_idx, end_idx in zip(cache_paths, start_indices, end_indices):
            # Load cached features: shape (seq_len, hidden_dim)
            feats = torch.load(cache_path, map_location="cpu")
            # Slice and transpose: (hidden_dim, seq_len_slice)
            feats = feats[start_idx:end_idx, :].transpose(0, 1)
            features_list.append(feats)

        # Stack into batch: (batch_size, hidden_dim, seq_len)
        device = batch_dict["audio"].device
        features = torch.stack(features_list, dim=0).to(device)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)
