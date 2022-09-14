"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import torch
import warnings
from speechain.module.abs import Module

class FeatureNormalization(Module):
    """
    The feature normalization frontend that makes every feature dimension the distribution with 0 mean and 1 variance.

    As SpeechBrain, we also provide four kinds of feature normalization and their levels increase from local to global.
        1. utterance-level normalization: the mean and std are calculated on each individual utterance.
        2. batch-level normalization: the mean and std are calculated on all the utterances in a training batch.
        3. group-level normalization: the mean and std are calculated on all the utterances in a group.
            The group here means where the utterance comes from, so it can be any kinds of data domains
            such as different speakers, genders, source and target domains in Domain Adaptation scenario, and so on...
        4. global-level normalization: the mean and std are calculated on all the utterances in the training set.

    We approximate group-level and global-level mean & std by taking their moving average during training.
    Different from SpeechBrain, we initialize all the mean & std variables lazily in the forward() function.
    Another difference is that our moving average is calculated on each batch as BatchNorm does.

    In the DDP mode, the mean & std will be synchronized across all the processes before being used to normalize the
    input utterances. The synchronization method is different in different scenarios.

    1. group-level normalization where each input utterance has different group id (group_ids = torch.Tensor,
        e.g. different utterances in a single batch may belong to different speakers).
        In this scenario, the mean & std vectors of each utterance and the group ids will be gathered across all the
        processes. Then, the mean & std vectors will be picked up depending on the group id and the mean & std of the
        specific group will be calculated.

    2. global-level normalization or group-level normalization where all the input utterances have the same group id
        (group_ids = str or int, e.g. all the utterances in the batch come from either the source domain or the target domain).
        In this scenario, the summation of mean & std vectors will be gathered instead of all of them to reduce the data
        communication volume across all the processes. The real mean & std vectors will be recovered by the batch size
        of each process.

    """
    def module_init(self,
                    norm_type: str = 'global',
                    mean_norm: bool = True,
                    std_norm: bool = True,
                    clamp: float = 1e-10,
                    max_epoch_num: int = 4):
        """

        Args:
            norm_type: str
                The type of feature normalization.
                The type must be one of 'utterance', 'batch', 'group', and 'global'
            mean_norm: bool
                Controls whether the feature vectors will be normalized by their means
            std_norm: bool
                Controls whether the feature vectors will be normalized by their standard variance
            clamp: float
                Clamping threshold for the standard variance before division.
            max_epoch_num: int
                The maximum number of epochs used to calculate the moving average.
                The value of this variable is usually equal to the number of warmup epochs.

        """
        self.norm_type = norm_type
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.clamp = clamp
        self.max_epoch_num = max_epoch_num

        if self.input_size is not None:
            self.output_size = self.input_size


    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor,
                group_ids: torch.Tensor or str or int = None, epoch: int = None):
        """

        Args:
            feat: (batch, length, channel)
            feat_len: (batch)
            group_ids: (batch)
            epoch:

        Returns:

        """
        batch_size = feat.size(0)

        # --- Mean and Standard Variance Initialization --- #
        # calculate the mean values of all channels of all the input utterances
        curr_means = None if not self.mean_norm else \
            torch.stack([feat[i][:feat_len[i]].mean(dim=0) for i in range(batch_size)])

        # calculate the std values of all channels of all the input utterances
        curr_stds = None if not self.std_norm else \
            torch.clamp(
                input=torch.stack([feat[i][:feat_len[i]].std(dim=0) for i in range(batch_size)]), min=self.clamp
            )


        # --- Perform Normalization based on Different branches --- #
        # utterance-level normalization or group-level normalization without group_ids
        if self.norm_type == 'utterance' or (self.norm_type == 'group' and group_ids is None):
            if self.training and (self.norm_type == 'group' and group_ids is None):
                warnings.warn("You are training group-level feature normalization without giving group_ids, "
                              "so it's replaced by the utterance-level normalization now!")
            feat = feat - curr_means.unsqueeze(1) if curr_means is not None else feat
            feat = feat / curr_stds.unsqueeze(1) if curr_stds is not None else feat

        # global-level & batch-level & group-level normalization (with group_ids)
        else:
            # only gather the batch sizes from other processes in the DDP model of training
            all_batch_size = None
            if self.training:
                all_batch_size = self.gather_scalars(batch_size, feat.device) if self.distributed else batch_size

            # group-level normalization with tensor group_ids (input utterances belong to different groups)
            if self.norm_type == 'group' and isinstance(group_ids, torch.Tensor):
                # only update the mean and std of the specific group during training
                if self.training:
                    # DDP mode
                    if self.distributed:
                        # gather all the group ids from other processes
                        all_group_ids = self.gather_vectors(group_ids, all_batch_size)
                        # gather all the mean vectors from other processes
                        all_curr_means = None if curr_means is None else \
                            self.gather_matrices(curr_means, all_batch_size)
                        # gather all the std vectors from other processes
                        all_curr_stds = None if curr_stds is None else \
                            self.gather_matrices(curr_stds, all_batch_size)
                    # single-GPU mode
                    else:
                        # not perform gathering
                        all_group_ids = group_ids
                        all_curr_means = curr_means
                        all_curr_stds = curr_stds

                    # record the mean of all groups in the current batch
                    group_mean_dict = self.sort_data_by_group(
                        raw_data=all_curr_means, group_ids=all_group_ids
                    )

                    # record the std of all groups in the current batch
                    group_std_dict = self.sort_data_by_group(
                        raw_data=all_curr_stds, group_ids=all_group_ids
                    )

                    # register the mean, std, and batch numbers into the buffer
                    group_keys = list(group_mean_dict.keys()) if group_mean_dict is not None \
                        else list(group_std_dict.keys())
                    for group_id in group_keys:
                        self.register_mean_std_batch(
                            curr_aver_mean=group_mean_dict[group_id].mean(dim=0) if group_mean_dict is not None else None,
                            curr_aver_std=group_std_dict[group_id].mean(dim=0) if group_std_dict is not None else None,
                            prefix=group_id, epoch=epoch
                        )

                # normalize the feature by the group mean and std if group values are registered
                # normalize the feature by the utterance mean and std if group values are not registered
                for i in range(batch_size):
                    group_id = group_ids[i].item() if group_ids is not None else None

                    if self.mean_norm:
                        feat[i] -= curr_means[i] if not hasattr(self, f"{group_id}_mean") else \
                            self.get_buffer(f"{group_id}_mean")
                    if self.std_norm:
                        feat[i] /= curr_stds[i] if not hasattr(self, f"{group_id}_std") else \
                            self.get_buffer(f"{group_id}_std")

            # batch-level & global-level normalization (these two scenarios share the batch-level mean & std)
            else:
                # only calculate the batch-level mean and std during training
                if self.training:
                    # gather the mean and std from the other processes in the DDP mode
                    if self.distributed:
                        # gather the sums of batch means from all the processes
                        batch_mean_sum = curr_means.sum(dim=0) if curr_means is not None else None
                        all_batch_mean_sums = self.gather_vectors(batch_mean_sum) if batch_mean_sum is not None else None
                        batch_mean = None if all_batch_mean_sums is None else \
                            all_batch_mean_sums.sum(dim=0) / all_batch_size.sum()

                        # gather the sums of batch stds from all the processes
                        batch_std_sum = curr_stds.sum(dim=0) if curr_stds is not None else None
                        all_batch_std_sums = self.gather_vectors(batch_std_sum) if batch_std_sum is not None else None
                        batch_std = None if all_batch_std_sums is None else \
                            all_batch_std_sums.sum(dim=0) / all_batch_size.sum()

                    # single-GPU mode
                    else:
                        batch_mean = curr_means.mean(dim=0) if curr_means is not None else None
                        batch_std = curr_stds.mean(dim=0) if curr_stds is not None else None

                # do nothing for batch-level mean and std during evaluation
                else:
                    batch_mean = None
                    batch_std = None


                # batch-level normalization
                if self.norm_type == 'batch':
                    # normalize the input utterances by the batch mean and std during training
                    if self.training:
                        feat = feat - batch_mean if batch_mean is not None else feat
                        feat = feat / batch_std if batch_std is not None else feat
                    # normalize the input utterances by the utterance-specific mean and std during evaluation
                    else:
                        feat = feat - curr_means.unsqueeze(1) if curr_means is not None else feat
                        feat = feat / curr_stds.unsqueeze(1) if curr_stds is not None else feat

                # global-level normalization or
                # group-level normalization with str or int group_ids (input utterances belong to the same group)
                else:
                    assert self.norm_type in ['global', 'group'], \
                        f"norm_type can only be one of 'utterance', 'batch', 'group', 'global', " \
                        f"but got norm_type={self.norm_type}!"
                    if self.norm_type == 'group':
                        assert isinstance(group_ids, (str, int)), \
                            f"If all the utterances in a single batch belong to the same group, " \
                            f"you should give group_ids as a string or integer. " \
                            f"But got type(group_ids)={type(group_ids)}."

                    # only update the mean and std during training
                    prefix = 'global' if self.norm_type == 'global' else group_ids
                    if self.training:
                        self.register_mean_std_batch(curr_aver_mean=batch_mean,
                                                     curr_aver_std=batch_std,
                                                     prefix=prefix, epoch=epoch)

                    feat = feat - self.get_buffer(f"{prefix}_mean") if curr_means is not None else feat
                    feat = feat / self.get_buffer(f"{prefix}_std") if curr_stds is not None else feat

        return feat, feat_len


    @staticmethod
    def gather_scalars(scalar: int, device: torch.device) -> torch.LongTensor:
        # gather the input scalars
        all_scalars = [torch.LongTensor([0]).cuda(device) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(all_scalars, torch.LongTensor([scalar]).cuda(device))
        return torch.LongTensor(all_scalars)


    @staticmethod
    def gather_vectors(vector: torch.Tensor, all_batch_size: torch.Tensor = None) -> torch.Tensor:
        # vectors of all the processes may have different length
        if all_batch_size is not None:
            curr_batch_size = all_batch_size[torch.distributed.get_rank()].item()
            max_batch_size = all_batch_size.max().item()
            if curr_batch_size < max_batch_size:
                vector = torch.cat((vector, torch.zeros(max_batch_size - curr_batch_size,
                                                        dtype=vector.dtype, device=vector.device)))
            all_vectors = [torch.Tensor([0 for _ in range(max_batch_size)]).type_as(vector).cuda(vector.device)
                           for _ in range(torch.distributed.get_world_size())]
        # all the vectors are equal in length
        else:
            all_vectors = [torch.zeros_like(vector, device=vector.device)
                           for _ in range(torch.distributed.get_world_size())]

        # gather the vectors from other processes to all_vectors
        torch.distributed.all_gather(all_vectors, vector)

        # remove the padding
        return torch.stack(all_vectors) if all_batch_size is None else \
            torch.cat([all_vectors[i][:all_batch_size[i]] for i in range(len(all_vectors))])


    @staticmethod
    def gather_matrices(matrix: torch.Tensor, all_batch_size: torch.Tensor) -> torch.Tensor:
        curr_batch_size = all_batch_size[torch.distributed.get_rank()].item()
        max_batch_size = all_batch_size.max().item()
        # padding the matrix if necessary
        if curr_batch_size < max_batch_size:
            matrix = torch.cat((matrix, torch.zeros(max_batch_size - curr_batch_size, matrix.size(-1),
                                                    device=matrix.device)))

        # gather the matrices from other processes to all_matrices
        all_matrices = [torch.zeros_like(matrix, device=matrix.device)
                        for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(all_matrices, matrix)

        # remove the padding
        return torch.cat([all_matrices[i][:all_batch_size[i]] for i in range(len(all_matrices))])


    @staticmethod
    def sort_data_by_group(raw_data: torch.Tensor, group_ids: torch.Tensor):
        """

        Args:
            raw_data:
            group_ids:

        Returns:

        """
        if raw_data is None:
            return None
        else:
            group_dict = dict()
            # loop each group id
            for i in range(group_ids.size(0)):
                curr_group = group_ids[i].item()
                # initialize the group list if not existed
                if curr_group not in group_dict.keys():
                    group_dict[curr_group] = []
                group_dict[curr_group].append(raw_data[i])
            # turn each group list into a 2d tensor
            return {group_id: torch.stack(group_list) for group_id, group_list in group_dict.items()}


    def register_mean_std_batch(self,
                                curr_aver_mean: torch.Tensor,
                                curr_aver_std: torch.Tensor,
                                prefix: str, epoch: int):
        """

        Args:
            curr_aver_mean:
            curr_aver_std:
            prefix:
            epoch:

        """
        # update the observed global batch number
        if epoch is None or not hasattr(self, f"{prefix}_batch"):
            self.register_buffer(f"{prefix}_batch", torch.LongTensor([1]).cuda(device=curr_aver_mean.device))
        elif epoch <= self.max_epoch_num:
            self.register_buffer(f"{prefix}_batch", self.get_buffer(f"{prefix}_batch") + 1)

        # update the observed global mean & std only in the predefined batch number
        if epoch is None or epoch <= self.max_epoch_num:
            # get the weight of the global average values
            curr_weight = 1 / self.get_buffer(f"{prefix}_batch")

            # update the observed global mean
            if self.mean_norm:
                if not hasattr(self, f"{prefix}_mean"):
                    self.register_buffer(f"{prefix}_mean", curr_aver_mean)
                else:
                    prev_aver_mean = self.get_buffer(f"{prefix}_mean")
                    self.register_buffer(f"{prefix}_mean",
                                         curr_weight * curr_aver_mean + (1 - curr_weight) * prev_aver_mean)

            # update the observed global std
            if self.std_norm:
                if not hasattr(self, f"{prefix}_std"):
                    self.register_buffer(f"{prefix}_std", curr_aver_std)
                else:
                    prev_aver_std = self.get_buffer(f"{prefix}_std")
                    self.register_buffer(f"{prefix}_std",
                                         curr_weight * curr_aver_std + (1 - curr_weight) * prev_aver_std)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Lazily register all the buffer variables ending with '_batch', '_std', or '_mean' from state_dict to self.

        """
        for key in state_dict.keys():
            if key.startswith(prefix):
                input_name = key[len(prefix):].split('.', 1)[0]

                if '_' in input_name and input_name.split('_')[-1] in ['batch', 'std', 'mean']:
                    self.register_buffer(input_name, state_dict[key])
                else:
                    unexpected_keys.append(key)


    def extra_repr(self) -> str:
        return f"norm_type={self.norm_type}, mean_norm={self.mean_norm}, std_norm={self.std_norm}"
