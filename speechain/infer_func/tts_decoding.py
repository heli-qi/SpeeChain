"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09

"""
import math
import torch


def auto_regression(enc_text: torch.Tensor,
                    enc_text_mask: torch.Tensor,
                    reduction_factor: int,
                    decode_one_step,
                    feat_dim: int,
                    spk_ids: torch.Tensor = None,
                    spk_feat: torch.Tensor = None,
                    stop_threshold: float = 0.5,
                    maxlen_ratio: int or float = 10.0,
                    continual_steps: int = 0,
                    use_before: bool = False):
    """

    Args:
        # --- Input Testing Data --- #
        enc_text:
        enc_text_mask:
        spk_ids:
        spk_feat:
        # --- Auto-Regression Process Controlling --- #
        reduction_factor:
        decode_one_step:
        feat_dim:
        stop_threshold:
        maxlen_ratio:
        continual_steps:
            Reference: Sec 3.1 in 'Generating synthetic audio data for attention-based speech recognition systems'
                https://arxiv.org/pdf/1912.09257
        use_before:

    """
    # --- Initialization Stage --- #
    batch_size = enc_text.size(0)
    enc_text_len = enc_text_mask.sum(dim=-1).squeeze()
    logits_threshold = -math.log(1 / stop_threshold - 1)

    # Different from the beam searching, syn_feat_maxlen is individually calculated for each utterance.
    # Since the utterances in a batch usually have the similar lengths for efficient batch-level decoding,
    # the text lengths are very likely to vary in a large range especially for subword and word tokenizers.
    # + 1 here is to consider the silence frames at the beginning
    hypo_feat_maxlen = enc_text_len * maxlen_ratio / reduction_factor + 1
    cuda_device = enc_text.device

    # Initial silence inputs as the first frames at the beginning of TTS decoding
    hypo_feat = torch.zeros((batch_size, 1, feat_dim), dtype=torch.float, device=cuda_device)
    hypo_feat_len = torch.ones(batch_size, dtype=torch.int, device=cuda_device)

    # --- Auto-Regressive Acoustic Feature Generation --- #
    stop_flags = torch.zeros(batch_size, dtype=torch.bool, device=cuda_device)
    stop_points = torch.zeros(batch_size, dtype=torch.int, device=cuda_device)
    # keep looping until all the synthetic utterances in the batch meet their stop flags
    while stop_flags.sum() < batch_size:
        pred_stop, pred_feat_before, pred_feat_after, _, _, _, _, _ = decode_one_step(
            enc_text=enc_text, enc_text_mask=enc_text_mask,
            feat=hypo_feat, feat_len=hypo_feat_len,
            spk_feat=spk_feat, spk_ids=spk_ids, is_test=True
        )

        # attach the new synthetic frames to the end of synthetic frames obtained so far
        # (batch_size, curr_len, feat_dim) + (batch_size, 1, feat_dim) = (batch_size, curr_len + 1, feat_dim)
        pred_feat = pred_feat_before[:, -1].unsqueeze(1) if use_before else pred_feat_after[:, -1].unsqueeze(1)
        # attach the silence to the utterances that has already been finished
        pred_feat[stop_flags] = 0
        hypo_feat = torch.cat([hypo_feat, pred_feat], dim=1)
        hypo_feat_len[~stop_flags] += 1

        # update the stop flags for all the utterances
        curr_steps = hypo_feat.size(1)
        pred_stop = pred_stop[:, -1].squeeze()
        # update the stop points where the stop token is met at the first time only
        stop_points[(pred_stop > logits_threshold) & (stop_points == 0)] = curr_steps
        # there are two stop conditions:
        # 1. stop token is met and continual_steps of frames have been generated
        # 2. maxlen of this utterance is met
        stop_flags = ((stop_points != 0) & (curr_steps >= stop_points + continual_steps)) | \
                     (hypo_feat_len >= hypo_feat_maxlen)

    # remove the redundant silence parts at the beginning of the synthetic frames
    # the silence parts at the end are not removed here
    # hypo_feat should be kept in the form of a matrix for a faster vocoder processing
    hypo_feat, hypo_feat_len = hypo_feat[:, 1:], hypo_feat_len - 1

    # reduction_factor recovery
    if reduction_factor > 1:
        assert feat_dim % reduction_factor == 0
        hypo_feat = hypo_feat.reshape(
            batch_size, hypo_feat.size(1) * reduction_factor, feat_dim // reduction_factor
        )
        hypo_feat_len *= reduction_factor

    return dict(
        hypo_feat=hypo_feat,
        hypo_feat_len=hypo_feat_len,
        feat_token_len_ratio=hypo_feat_len / enc_text_len
    )
