import torch

from speechain.utilbox.train_util import make_mask_from_len


class CTCPrefixScorer:
    """This class implements the CTC prefix scorer of Algorithm 2 in
    reference: https://www.merl.com/publications/docs/TR2017-190.pdf.
    Official implementation: https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py
    Arguments
    ---------
    x : torch.Tensor
        The encoder states.
    enc_lens : torch.Tensor
        The actual length of each enc_states sequence.
    batch_size : int
        The size of the batch.
    beam_size : int
        The width of beam.
    blank_index : int
        The index of the blank token.
    eos_index : int
        The index of the end-of-sequence (eos) token.
    ctc_window_size: int
        Compute the ctc scores over the time frames using windowing based on attention peaks.
        If 0, no windowing applied.
    """

    def __init__(self, ctc_logits, enc_feat_len, batch_size, beam_size, blank_index, eos_index):
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.max_enc_len = ctc_logits.size(1)
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.vocab_size = ctc_logits.size(-1)
        self.device = ctc_logits.device
        self.minus_inf = -1e20
        self.last_frame_index = enc_feat_len - 1

        # mask frames > enc_lens
        mask = ~make_mask_from_len(enc_feat_len, return_3d=False)
        mask = mask.unsqueeze(-1).expand(-1, -1, ctc_logits.size(-1))
        if ctc_logits.is_cuda:
            mask = mask.cuda(self.device)
        # mask the extra part with -inf while the blank index is masked by 0
        ctc_logits.masked_fill_(mask, self.minus_inf)
        ctc_logits[:, :, self.blank_index].masked_fill_(mask[:, :, self.blank_index], 0)

        # (max_enc_len, batch_size * beam_size, vocab_size), _nb means non-blank posteriors
        ctc_logits_nb = ctc_logits.transpose(0, 1)
        # (max_enc_len, batch_size * beam_size, vocab_size), _b means blank posteriors
        ctc_logits_b = ctc_logits_nb[:, :, self.blank_index].unsqueeze(2).expand(-1, -1, self.vocab_size)

        # (2, max_enc_len, batch_size * beam_size, vocab_size)
        self.ctc_logits = torch.stack([ctc_logits_nb, ctc_logits_b])

        # The first index of each sentence.
        self.beam_offset = torch.arange(batch_size, device=self.device) * self.beam_size
        # The first index of each candidate.
        self.cand_offset = torch.arange(batch_size, device=self.device) * self.vocab_size

    def forward_step(self, g, state):
        """This method if one step of forwarding operation
        for the prefix ctc scorer.
        Arguments
        ---------
        g : torch.Tensor
            The tensor of prefix label sequences, h = g + c.
        state : tuple
            Previous ctc states.
        """

        prefix_length = g.size(1)
        last_char = [gi[-1] for gi in g] if prefix_length > 0 else [0] * len(g)

        if state is None:
            # (max_enc_len, 2, batch_size, beam_size)
            r_prev = torch.full(
                (self.max_enc_len, 2, self.batch_size, self.beam_size),
                self.minus_inf,
                device=self.device,
            )

            # Accumulate blank posteriors at each step
            r_prev[:, 1] = torch.cumsum(self.ctc_logits[0, :, :, self.blank_index], 0).unsqueeze(1)
            r_prev = r_prev.view(-1, 2, self.batch_size * self.beam_size)
            psi_prev = 0.0
        else:
            r_prev, psi_prev = state

        scoring_table = None
        x_inflate = (
            self.ctc_logits.unsqueeze(3)
            .repeat(1, 1, 1, self.beam_size, 1)
            .view(
                2, -1, self.batch_size * self.beam_size, self.vocab_size
            )
        )

        # Prepare forward probs
        r = torch.full(
            (
                self.max_enc_len,
                2,
                self.batch_size * self.beam_size,
                self.vocab_size,
            ),
            self.minus_inf,
            device=self.device,
        )
        r.fill_(self.minus_inf)

        # (Alg.2-6)
        if prefix_length == 0:
            r[0, 0] = x_inflate[0, 0]
        # (Alg.2-10): phi = prev_nonblank + prev_blank = r_t-1^nb(g) + r_t-1^b(g)
        r_sum = torch.logsumexp(r_prev, 1)
        phi = r_sum.unsqueeze(2).repeat(1, 1, self.vocab_size)

        # (Alg.2-10): if last token of prefix g in candidates, phi = prev_b + 0
        for i in range(self.batch_size * self.beam_size):
            phi[:, i, last_char[i]] = r_prev[:, 1, i]

        # Start, end frames for scoring (|g| < |h|).
        # Scoring based on attn peak if ctc_window_size > 0
        start = max(1, prefix_length)
        end = self.max_enc_len

        # Compute forward prob log(r_t^nb(h)) and log(r_t^b(h)):
        for t in range(start, end):
            # (Alg.2-11): dim=0, p(h|cur step is nonblank) = [p(prev step=y) + phi] * p(c)
            rnb_prev = r[t - 1, 0]
            # (Alg.2-12): dim=1, p(h|cur step is blank) = [p(prev step is blank) + p(prev step is nonblank)] * p(blank)
            rb_prev = r[t - 1, 1]
            r_ = torch.stack([rnb_prev, phi[t - 1], rnb_prev, rb_prev]).view(
                2, 2, self.batch_size * self.beam_size, self.vocab_size
            )
            r[t] = torch.logsumexp(r_, 1) + x_inflate[:, t]

        # Compute the predix prob, psi
        psi_init = r[start - 1, 0].unsqueeze(0)
        # phi is prob at t-1 step, shift one frame and add it to the current prob p(c)
        phix = torch.cat((phi[0].unsqueeze(0), phi[:-1]), dim=0) + x_inflate[0]
        # (Alg.2-13): psi = psi + phi * p(c)
        psi = torch.logsumexp(
            torch.cat((phix[start:end], psi_init), dim=0), dim=0
        )

        # (Alg.2-3): if c = <eos>, psi = log(r_T^n(g) + r_T^b(g)), where T is the length of max frames
        for i in range(self.batch_size * self.beam_size):
            psi[i, self.eos_index] = r_sum[
                self.last_frame_index[i // self.beam_size], i
            ]

        # Exclude blank probs for joint scoring
        psi[:, self.blank_index] = self.minus_inf

        return psi - psi_prev, (r, psi, scoring_table)

    def permute_mem(self, memory, index):
        """This method permutes the CTC model memory
        to synchronize the memory index with the current output.
        Arguments
        ---------
        memory : No limit
            The memory variable to be permuted.
        index : torch.Tensor
            The index of the previous path.
        Return
        ------
        The variable of the memory being permuted.
        """
        r, psi, scoring_table = memory
        # The index of top-K vocab came from in (t-1) timesteps.
        best_index = (
            index
            + (self.beam_offset.unsqueeze(1).expand_as(index) * self.vocab_size)
        ).view(-1)
        # synchronize forward prob
        psi = torch.index_select(psi.view(-1), dim=0, index=best_index)
        psi = (
            psi.view(-1, 1)
            .repeat(1, self.vocab_size)
            .view(self.batch_size * self.beam_size, self.vocab_size)
        )

        # synchronize ctc states
        if scoring_table is not None:
            effective_index = (
                index // self.vocab_size + self.beam_offset.view(-1, 1)
            ).view(-1)
            selected_vocab = (index % self.vocab_size).view(-1)
            score_index = scoring_table[effective_index, selected_vocab]
            score_index[score_index == -1] = 0
            best_index = score_index + effective_index * self.vocab_size

        r = torch.index_select(
            r.view(
                -1, 2, self.batch_size * self.beam_size * self.vocab_size
            ),
            dim=-1,
            index=best_index,
        )
        r = r.view(-1, 2, self.batch_size * self.beam_size)

        return r, psi
