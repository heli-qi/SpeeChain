"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List, Dict

from speechain.iterator.abs import Iterator
import numpy as np


class BlockIterator(Iterator):
    """
    The strategy of this iterator is to generate batches with the same amount of data lengths. For sequence-to-sequence
    tasks, the data instances are usually different in data length. If there is a fixed number of data instances in each
    batch, the data volume of a single batch may constantly change during training. This may either cause a CUDA memory
    error (out of GPU memory) or large idle GPU memories.

    It can be considered as the strategy that always gives 'rectangles' with similar 'areas' if we treat the number of
    data instances in a batch as the rectangle length and the maximal data length as the rectangle width.

    """
    def batches_generate_fn(self, data_index: List[str], data_len: Dict[str, int], batch_len: int = None) \
            -> List[List[str]]:
        """
        All the data instances in the built-in Dataset object will be grouped into batches with the same total lengths.
        The lengths used for grouping is given in data_len. The customized argument batch_len specifies the total length
        that each batch should have.

        Args:
            data_index
            data_len
            batch_len: int = None
                The total data length of all the data instances in a batch.
                If the data is in the format of audio waveforms, batch_len is the amount of sampling points.
                If the data is in the format of acoustic features, batch_len is the amount of time frames.

        """
        assert batch_len is not None, "batch_len cannot be None and must be specified!"
        if not isinstance(batch_len, int):
            batch_len = int(batch_len)
        # configuration initialization
        assert batch_len > 0, f"batch_len must be a positive integer, but got {batch_len}."

        # divide the data into individual batches by their lengths
        batches = []
        current_batch_frames = 0
        current_batch = []
        for index in data_index:
            current_batch.append(index)
            current_batch_frames += data_len[index]

            if current_batch_frames >= batch_len:
                batches.append(current_batch)
                current_batch = []
                current_batch_frames = 0

        # add the remaining instances as a single batch
        if len(current_batch) > 0:
            batches.append(current_batch)

        return batches
