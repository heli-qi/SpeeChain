"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07

    Modified from
    https://github.com/huggingface/transformers/blob/518bd02c9b71291333ef374f055a4d1ac3042654/src/transformers/generation_beam_search.py

"""
import torch
import numpy as np

class BeamHypotheses(object):
    """
    Beam Hypothesis Container.

    """
    def __init__(self, beam_size: int, max_length: int, length_penalty: float):
        """
        Initialize n-best list of hypotheses.

        Args:
            beam_size: int
                The number of beams used in this container
            max_length: int
                The maximal length of the generated hypotheses
            length_penalty: float
                The penalty you put on the hypothesis lengths.
                The larger length_penalty is, the longer your hypotheses will be.
                length_penalty=1 means no penalty on lengths.
        """
        # static variables
        self.max_length = max_length - 1  # ignoring bos_token
        self.beam_size = beam_size
        self.length_penalty = length_penalty # length_penalty=1 means no penalty on length

        # dynamic variables
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses generated so far.
        """
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        """
        Add a new hypothesis to the container.

        Args:
            hyp: (hypo_len,)
                The generated hypothesis transcript.
            sum_logprobs: float
                The sum of log probability of each token prediction in the hypothesis.

        """
        # normalize the sum of log probability by the hypothesis length
        score = sum_logprobs / (len(hyp) ** self.length_penalty)

        # some beams remain undone or the score is better than the worst score so far
        if len(self) < self.beam_size or score > self.worst_score:
            self.beams.append((score, hyp))

            # remove the worst hypothesis and update the worst score
            if len(self) > self.beam_size:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            # update the worst score
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, curr_len: int = None):
        """
        whether the beam searching of this container is done or not

        Args:
            best_sum_logprobs: float
                The best log-prob sum we get in the current time step
            curr_len: int
                The length of the current input hypothesis

        Returns:
            A flag that indicates whether the beam searching of this container is done.
            True means the container already has 'beam_size' hypotheses and the current hypothesis is not better than anyone of them.
            False means either the container has some empty beams or the current input hypothesis is better than the worst hypothesis.

        """
        # some beams are empty
        if len(self) < self.beam_size:
            return False
        # no beam is empty
        else:
            if curr_len is None:
                curr_len = self.max_length
            curr_score = best_sum_logprobs / curr_len ** self.length_penalty

            # whether the current score is worse than the worst score
            return curr_score <= self.worst_score


def beam_searching(enc_feat: torch.Tensor,
                   enc_feat_mask: torch.Tensor,
                   decode_one_step,
                   vocab_size: int,
                   sos_eos: int = None,
                   beam_size: int = 1,
                   maxlen_ratio: float = 1.0,
                   length_penalty: float = 1.0,
                   padding_idx: int = 0,
                   sent_per_beam: int = 1):
    """
    Batch version of beam searching to enable parallel computation.
    The basic idea is reshaping batch_size sentences into (batch_size * beam_size) sentences.

    batch_size > 1 is mainly used to speed up the pseudo text generation during training.
    For testing, usually set batch_size=1 for evaluating the processing time to mimic real-world application.

    Args:
        enc_feat: (batch_size, feat_maxlen, edim)
            The encoded features.
        enc_feat_mask: (batch_size, 1, feat_maxlen)
            The masks for the encoded features.
        decode_one_step:
            The function that decodes the hypothesis for one time step and get the next prediction.
        vocab_size: int
            The number of tokens in the vocabulary dictionary
        sos_eos: int
            The index of the <sos/eos> token.
        beam_size: int
            The number of beams used for each hypothesis sentence.
        maxlen_ratio: float
            The ratio of the hypothesis max length to encoded feature length.
            Postive values mean the relative ratio.
            Negative values mean the absolute max length of the hypothesis sentence.
        padding_idx: int
            The index of the padding token.
        sent_per_beam: int
            The number of sentences in each beam that are returned in this function.
            Mainly used for data augmentation.

    Returns:

    """
    # para init
    batch_size = enc_feat.size(0)
    feat_maxlen = enc_feat.size(1)
    hypo_maxlen = int(feat_maxlen * maxlen_ratio) if maxlen_ratio > 0 else int(-maxlen_ratio)
    cuda_device = enc_feat.device
    if sos_eos is None:
        sos_eos = vocab_size - 1

    # --- Input Data Reshaping --- #
    # (batch_size, feat_maxlen, edim) -> (batch_size, 1, feat_maxlen , edim)
    enc_feat = enc_feat.unsqueeze(1).contiguous()
    # (batch_size, 1, feat_maxlen , edim) -> (batch_size, beam_size, feat_maxlen, edim)
    enc_feat = enc_feat.repeat(1, beam_size, 1, 1)
    # (batch_size, beam_size, feat_maxlen, edim) -> (batch_size × beam_size, frame, edim)
    enc_feat = enc_feat.view(-1, enc_feat.size(2), enc_feat.size(3)).contiguous()

    # (batch_size, 1, feat_maxlen) -> (batch_size, 1, 1, feat_maxlen)
    enc_feat_mask = enc_feat_mask.unsqueeze(1).contiguous()
    # (batch_size, 1, 1, feat_maxlen) -> (batch_size, beam_size, 1, feat_maxlen)
    enc_feat_mask = enc_feat_mask.repeat(1, beam_size, 1, 1)
    # (batch_size, beam_size, 1, feat_maxlen) -> (batch_size × beam_size, 1, feat_maxlen)
    enc_feat_mask = enc_feat_mask.view(-1, enc_feat_mask.size(2), enc_feat_mask.size(3)).contiguous()

    # --- Registers Initialization --- #
    # build a hypothesis container for each sentence in the batch
    generated_hyps = [BeamHypotheses(beam_size, hypo_maxlen, length_penalty) for _ in range(batch_size)]
    # Done flags for all sentences in the batch
    done = [False for _ in range(batch_size)]
    # scores of all beam containers (batch_size × beam_size,)
    beam_scores = torch.zeros((batch_size * beam_size,), dtype=torch.float, device=cuda_device)
    # start tokens for all sentences in a batch (batch_size × beam_size, 1)
    hypo_text = torch.full((batch_size * beam_size, 1), sos_eos, dtype=torch.long, device=cuda_device)
    hypo_text_len = torch.ones((batch_size * beam_size,), dtype=torch.long, device=cuda_device)

    # --- Beam Searching Algorithm --- #
    while hypo_text_len.max() < hypo_maxlen:
        # (batch_size × beam_size, curr_len) -> (batch_size × beam_size, curr_len, vocab_size)
        curr_outputs = decode_one_step(enc_feat=enc_feat, enc_feat_mask=enc_feat_mask,
                                       text=hypo_text, text_len=hypo_text_len)['output'].detach()

        # (batch_size × beam_size, curr_len, vocab_size) -> (batch_size × beam_size, 1, vocab_size)
        scores = curr_outputs[:, -1, :]

        # Calculate the score of the obtained token predictions
        next_scores = torch.log_softmax(scores, dim=-1) + beam_scores.unsqueeze(-1).expand_as(scores)

        # Arrange all beams of the same sentence into a single row to pick up the best predictions across all beams.
        # (batch_size × beam_size, vocab_size) -> (batch_size, beam_size × vocab_size)
        next_scores = next_scores.view(batch_size, -1)

        # Pick up (beam_size × beam_size) tokens from (beam_size * vocab_size) candidates for algorithm robustness
        # mainly two usage:
        #   1. for different tokens of each beam in the first time step
        #   2. when meeting an eos token, complement the beams with the rest predictions
        assert beam_size <= vocab_size, "beam_size cannot be larger than vocab_size!"
        next_scores, next_tokens = torch.topk(next_scores, beam_size * beam_size, dim=1, largest=True, sorted=True)

        # batch-level beam results for all sentence, each element is a tri-tuple (score, token_id, effective_beam_id)
        next_batch_beam = []
        # looping each sentence in a batch
        for batch_idx in range(batch_size):
            # if the current sentence is already finished, adding a padding token and continue
            if done[batch_idx]:
                # for the finished sentence, the padding token is added
                next_batch_beam.extend([(0, padding_idx, 0)] * beam_size)
                continue

            # sentence-level beam results for all beams in the current sentence,
            # each element is a tri-tuple (score, token_id, effective_beam_id)
            next_sent_beam = []
            # looping each beam, there are (beam_size * batch_size) candidate predictions in total.
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # the index number of the beam in a single sentence, range from 0 to beam_size-1
                # beam_id = beam_token_id // vocab_size
                beam_id = torch.div(beam_token_id, vocab_size, rounding_mode='floor')
                # the index number of the real token, range from 0 to vocab_size-1
                token_id = beam_token_id % vocab_size
                # the index number of the beam across all sentences, range from 0 to batch_size*beam_size-1
                effective_beam_id = batch_idx * beam_size + beam_id

                # if the eos token is met, the predictions will be either saved as a hypothesis or simple removed.
                if token_id.item() == sos_eos:
                    # the top beam_size elements in next_tokens have the largest prob,
                    # so the ones other than the first beam_size elements will be ignored even though eos is met.
                    if beam_token_rank >= beam_size:
                        continue
                    generated_hyps[batch_idx].add(hypo_text[effective_beam_id].clone(), beam_token_score.item())
                # add the predictions into the temporary results.
                else:
                    # make sure that different tokens are selected for different beams in the first time step
                    add_flag = True
                    if hypo_text_len.max() == 1 and len(next_sent_beam) != 0:
                        for item in next_sent_beam:
                            if item[1] == token_id or item[2] == effective_beam_id:
                                add_flag = False
                                break
                    if add_flag:
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # only get beam_size predictions from beam_size * beam_size candidates
                if len(next_sent_beam) == beam_size:
                    break

            # update the done flag of the current sentence
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), curr_len=hypo_text_len.max().item()
            )
            next_batch_beam.extend(next_sent_beam)

        if all(done):
            break

        # Summary of the results of the selected predictions for all beams in the current time step
        # (batch_size * beam_size), beam score and beam index
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = hypo_text.new([x[1] for x in next_batch_beam])
        beam_idx = hypo_text.new([x[2] for x in next_batch_beam])

        # Update the length of generated sentences
        hypo_text = torch.cat([hypo_text[beam_idx], beam_tokens.unsqueeze(1)], dim=1)
        hypo_text_len = torch.sum(hypo_text != padding_idx, dim=-1)

        # align encoder_out with input_tokens
        enc_feat = enc_feat[beam_idx]
        enc_feat_mask = enc_feat_mask[beam_idx]

    # --- Post-processing --- #
    # for the predictions that end without an eos token at the end because of the max length
    for batch_idx in range(batch_size):
        # pass the hypothesis sentence that ends with eos
        if done[batch_idx]:
            continue
        # check whether they can be added into final beam searching results
        for beam_id in range(beam_size):
            effective_beam_id = batch_idx * beam_size + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = hypo_text[effective_beam_id].clone()
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # --- Length Calculation --- #
    hypo_text_len = hypo_text_len.new(batch_size * sent_per_beam)
    hypo_text_list = []
    hypo_text_prob = []
    # looping each sentence
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        # looping each beam
        for j in range(sent_per_beam):
            effective_batch_idx = sent_per_beam * i + j
            _hypo = sorted_hyps.pop()

            # remove the sos tokens at the beginning of hyp
            best_hyp = _hypo[1][1:]
            hypo_text_len[effective_batch_idx] = len(best_hyp)
            hypo_text_list.append(best_hyp)
            hypo_text_prob.append(_hypo[0])

    # --- Padding --- #
    # the sentences shorter than the maximal length or there is only one sentence in a single batch
    if hypo_text_len.min().item() != hypo_text_len.max().item() or batch_size == 1:
        sent_max_len = min(hypo_text_len.max().item(), hypo_maxlen)
        hypo_text = torch.full((batch_size * sent_per_beam, sent_max_len), padding_idx,
                               dtype=torch.long, device=cuda_device)
        for i in range(len(hypo_text_list)):
            # padding pseudo transcripts
            hypo_text[i, :hypo_text_len[i]] = hypo_text_list[i]
    # all the sentences are equally long
    elif batch_size > 1:
        hypo_text = torch.stack(hypo_text_list).to(cuda_device)
    else:
        raise RuntimeError

    return dict(
        hypo_text=hypo_text,
        hypo_text_len=hypo_text_len,
        hypo_text_prob=torch.Tensor(hypo_text_prob)
    )