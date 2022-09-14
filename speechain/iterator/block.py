"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List

from speechain.iterator.abs import Iterator
import numpy as np

class BlockIterator(Iterator):
    """
    The strategy of this iterator is to generate batches with the same frame amount. For sequence-to-sequence tasks,
    the input and output data are usually various in length. If the batch size is a fixed number, the data volume of
    a single batch may constantly change during training. This may either cause a CUDA memory error (out of GPU
    memory) or large idle GPU memories.

    It can be considered as the strategy that always gives out a batch as a rectangle with the same 'area' if we treat
    the sequence amount as the length and the maximum sequence length as the width.

    """
    def iter_init(self, batch_frames: int):
        """
        All input-ouput pairs in the dataset will be grouped into batches according to the sum of their
        lengths. The lengths used for grouping is in self.data_len

        Args:
            batch_frames: int
                The maximum frame amount of a batch.
                If the data is in the format of audio waveforms, batch_frames is the amount of sampling points.
                If the data is in the format of acoustic features, batch_frames is the amount of time frames.

        """
        if not isinstance(batch_frames, int):
            batch_frames = int(batch_frames)
        # configuration initialization
        assert batch_frames > 0, \
            f"batch_frames must be a positive integer, but got {batch_frames}."
        self.batch_frames = batch_frames

        # divide the data into individual batches by their lengths
        batches = []
        current_batch_frames = 0
        current_batch = []
        for index in self.sorted_data:
            current_batch.append(index)
            current_batch_frames += self.data_len[index]

            if current_batch_frames >= self.batch_frames:
                batches.append(current_batch)
                current_batch = []
                current_batch_frames = 0

        # add the remaining samples as a single batch
        if len(current_batch) > 0:
            batches.append(current_batch)

        return batches
