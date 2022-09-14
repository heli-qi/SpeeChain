"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from speechain.iterator.abs import Iterator


class PieceIterator(Iterator):
    """
    The strategy of this iterator is to generate batches with the same sequence amount. Each batch will have exactly
    the same number of sequences no matter how long they are. Mainly used for testing the sequence-to-sequence
    models like ASR with batch_size=1.

    """
    def iter_init(self, batch_size: int):
        """
        All input-ouput pairs in the dataset will be grouped into batches with exactly the same amount of samples.
        The data lengths here are used for sorting all the samples to make sure that the samples in a single batch
        have similar lengths.

        Args:
            batch_size: int

        """
        assert batch_size > 0 and isinstance(batch_size, int), \
            f"batch_size must be a positive integer, but got {batch_size}."
        self.batch_size = batch_size

        # divide the data into individual batches with equal amount of samples
        batches = [self.sorted_data[i: i + self.batch_size]
                   for i in range(0, len(self.sorted_data) - self.batch_size + 1, self.batch_size)]
        # in case that there are several uncovered samples at the end of self.sorted_data
        if len(self.sorted_data) % self.batch_size != 0:
            remaining = len(self.sorted_data) % self.batch_size
            batches.append(self.sorted_data[-remaining:])

        return batches
