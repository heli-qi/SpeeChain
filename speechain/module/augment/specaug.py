from typing import List

import torch

from speechain.module.abs import Module

class SpecAugment(Module):
    """
    Batch-level SpecAugment.
    Implemented based on
        'SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition'
        reference: https://arxiv.org/pdf/1904.08779.pdf
    This implementation is inspired by
        https://github.com/espnet/espnet/blob/36e824be58ea6c6844e3d87b11e382f90ba4fb22/espnet2/layers/time_warp.py#L9
        https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/augment.py#L116

    """
    def module_init(self,
                    time_warp: bool = True,
                    time_warp_window: int = 5,
                    time_warp_mode: str = 'bicubic',
                    freq_mask: bool = True,
                    freq_mask_width: int or List[int] = 30,
                    freq_mask_num: int = 2,
                    time_mask: bool = True,
                    time_mask_width: int or List[int] = 40,
                    time_mask_num: int = 2,
                    time_mask_ratio: float = 1.0,
                    feat_norm: bool = True):
        """

        Args:
            time_warp:
            time_warp_window:
            time_warp_mode:
            freq_mask:
            freq_mask_width:
            freq_mask_num:
            time_mask:
            time_mask_width:
            time_mask_num:
            time_mask_ratio:
            feat_norm:

        """
        assert time_warp or freq_mask or time_mask, \
            "You must specify at least one type of augmentation in SpecAugment!"
        self.feat_dim = None
        if self.input_size is not None:
            self.feat_dim = self.input_size
            self.output_size = self.input_size

        # time warping arguments
        self.time_warp = time_warp
        self.time_warp_window = time_warp_window
        self.time_warp_mode = time_warp_mode

        # frequency masking arguments
        self.freq_mask = freq_mask
        if isinstance(freq_mask_width, int):
            freq_mask_width = [0, freq_mask_width]
        elif not isinstance(freq_mask_width, List):
            raise ValueError
        if self.feat_dim is not None:
            assert freq_mask_width[1] < self.feat_dim, \
                "The number of maximum frequency masking bins cannot be larger than the feature dimension! " \
                f"freq_mask_width[1]={freq_mask_width[1]} and self.feat_dim={self.feat_dim}."
        self.freq_mask_width = freq_mask_width
        self.freq_mask_num = freq_mask_num

        # time masking arguments
        self.time_mask = time_mask
        if isinstance(time_mask_width, int):
            time_mask_width = [0, time_mask_width]
        elif not isinstance(time_mask_width, List):
            raise ValueError
        self.time_mask_width = time_mask_width
        self.time_mask_num = time_mask_num
        self.time_mask_ratio = time_mask_ratio

        # used for deciding masking values
        self.feat_norm = feat_norm


    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """
        Both the time warping and time masking are done within the minimum length of all the utterance in the input batch.

        This practice is to make sure that the time warping and masking are done in the effective area of the input data
        and the feature length information are still valid after augmentation.

        Args:
            feat:
            feat_len:

        Returns:

        """
        batch_size, feat_dim = feat.size(0), feat.size(-1)
        time_maxlen, time_minlen = feat_len.max().item(), feat_len.min().item()
        # --- Time Warping --- #
        if self.time_warp:
            # create channel dimension: (batch_size, time_maxlen, feat_dim) -> (batch_size, 1, time_maxlen, feat_dim)
            if feat.dim() == 3:
                feat = feat.unsqueeze(1)

            # time_minlen must be larger than 2 times of the warping window length
            # otherwise, the input is too short to be warped (do nothing to the feature)
            if time_minlen > 2 * self.time_warp_window + 1:
                # center ∈ {time_warp_window + 1, ..., time_minlen - time_warp_window - 1}
                warp_center = torch.randint(low=self.time_warp_window + 1, high=time_minlen - self.time_warp_window,
                                            size=(1,))[0].item()
                # position ∈ {1, ..., time_minlen - 1} (consider the range of the center)
                warp_pos = torch.randint(low=warp_center - self.time_warp_window, high=warp_center + self.time_warp_window,
                                         size=(1,))[0].item()
                # interpolate the left and right parts of the selected center within time_minlen to protect feat_len
                # align_corners=True to keep in line with the original paper
                left_warp = torch.nn.functional.interpolate(feat[:, :, :warp_center], size=(warp_pos, feat_dim),
                                                            mode=self.time_warp_mode, align_corners=True)
                right_warp = torch.nn.functional.interpolate(feat[:, :, warp_center: time_minlen],
                                                             size=(time_minlen - warp_pos, feat_dim),
                                                             mode=self.time_warp_mode, align_corners=True)
                feat[:, :, :warp_pos] = left_warp
                feat[:, :, warp_pos: time_minlen] = right_warp

            # remove the redundant channel dimension
            feat = feat.view(batch_size, time_maxlen, feat_dim)


        # --- Feature Masking (Frequency axis or Time axis) --- #
        # overall mask
        mask = None
        # frequency mask generation
        if self.freq_mask:
            # lazily check the frequency masking width during training if self.feat_dim is not initialized
            if self.feat_dim is None:
                assert self.feat_dim == feat_dim and self.freq_mask_width[1] < feat_dim, \
                    "The number of maximum frequency masking bins cannot be larger than the feature dimension! " \
                    f"self.freq_mask_width[1]={self.freq_mask_width[1]} and feat_dim={feat_dim}."

            # randomly select the masking length for each masking operation in each utterance
            # (batch_size, freq_mask_num, 1), mask_len ∈ {self.freq_mask_width[0], ..., self.freq_mask_width[1]}
            mask_len = torch.randint(self.freq_mask_width[0], self.freq_mask_width[1] + 1,
                                     size=(batch_size, self.freq_mask_num), device=feat.device).unsqueeze(2)
            # randomly select the masking position for each masking operation in each utterance
            # (batch_size, freq_mask_num, 1), mask_pos ∈ {0, ..., feat_dim - mask_len.max - 1}
            mask_pos = torch.randint(0, max(1, feat_dim - mask_len.max().item()),
                                     size=(batch_size, self.freq_mask_num), device=feat.device).unsqueeze(2)
            # (1, 1, feat_dim)
            freq_axis = torch.arange(feat_dim, device=feat.device)[None, None, :]
            # (batch_size, freq_mask_num, feat_dim) -> (batch_size, 1, feat_dim)
            feat_mask = (mask_pos <= freq_axis) * (freq_axis < (mask_pos + mask_len))
            mask = feat_mask.any(dim=1, keepdim=True)

        # time mask generation
        if self.time_mask:
            # the maximum time masking width cannot be larger than ratio × minimum time sequence length
            time_mask_width = self.time_mask_width
            time_mask_width[1] = min(time_mask_width[1], int(self.time_mask_ratio * time_minlen))

            # randomly select the time masking length for each masking operation in each utterance
            # (batch_size, 1, time_mask_num), mask_len ∈ {time_mask_width[0], ..., time_mask_width[1]}
            mask_len = torch.randint(time_mask_width[0], time_mask_width[1] + 1,
                                     size=(batch_size, self.time_mask_num), device=feat.device).unsqueeze(1)
            # randomly select the time masking position for each masking operation in each utterance
            # (batch_size, 1, time_mask_num), mask_pos ∈ {0, ..., time_minlen - mask_len.max - 1}
            mask_pos = torch.randint(0, max(1, time_minlen - mask_len.max().item()),
                                     size=(batch_size, self.time_mask_num), device=feat.device).unsqueeze(1)
            # (1, time_maxlen, 1)
            time_axis = torch.arange(time_maxlen, device=feat.device)[None, :, None]
            # (batch_size, time_maxlen, time_mask_num) -> (batch_size, time_maxlen, 1)
            time_mask = (mask_pos <= time_axis) * (time_axis < (mask_pos + mask_len))
            time_mask = time_mask.any(dim=-1, keepdim=True)
            # combine time mask with frequency mask if both are specified
            # (batch_size, time_maxlen, 1) or (batch_size, 1, feat_dim) = (batch_size, time_maxlen, feat_dim)
            mask = time_mask if mask is None else torch.logical_or(mask, time_mask)

        # one-shot feature masking
        if mask is not None:
            mask_value = 0.0 if self.feat_norm else feat.mean()
            feat = feat.masked_fill(mask, mask_value)

        return feat, feat_len


    def extra_repr(self) -> str:
        output = ""
        if self.time_warp:
            output += f"time_warp_window={self.time_warp_window}, " \
                      f"time_warp_mode={self.time_warp_mode}, "

        if self.freq_mask:
            output += f"\nfreq_mask_width={self.freq_mask_width}, " \
                      f"freq_mask_num={self.freq_mask_num}, "

        if self.time_mask:
            output += f"\ntime_mask_width={self.time_mask_width}, " \
                      f"time_mask_num={self.time_mask_num}, " \
                      f"time_mask_ratio={self.time_mask_ratio}"

        return output
