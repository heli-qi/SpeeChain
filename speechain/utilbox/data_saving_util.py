import os
import torch
import numpy as np
import soundfile as sf

from typing import List, Any, Union

from speechain.utilbox.tensor_util import to_cpu


def save_data_by_format(file_format: str, save_path: str, file_name_list: Union[List[str] or str],
                        file_content_list: Union[List, np.ndarray, torch.Tensor],
                        group_ids: List[str] = None, sample_rate: int = None):
    """
    Save data in the specified format to disk.

    Args:
        file_format (str):
            The format of the output files. It can be one of 'npy', 'npz', 'wav' or 'flac'.
        save_path (str):
            The directory where the files will be saved.
        file_name_list (List[str]):
            A list of strings with the names of the files to be saved.
        file_content_list (List):
            A list with the content of the files to be saved.
        group_ids (List[str], optional):
            A list of strings with the group ids of the files. If provided, it will be used to create a subdirectory
            for each group of files inside the save_path. Defaults to None.
        sample_rate (int, optional):
            The sample rate of the audio files, required for 'wav' and 'flac' formats. Defaults to None.

    Returns:
        Dict[str, str]:
            A dictionary that maps the original file names to their corresponding file paths in disk.

    Raises:
        NotImplementedError:
            If the file format is not supported.

    Notes:
        - The content of the files can be of any type as long as it can be converted to numpy arrays (using the
          to_cpu() function if necessary).
        - If sample_rate is not None, it will be saved along with the data for 'npz', 'wav' and 'flac' formats.
    """

    # record all the data file paths with their names
    name2file_path = {}

    if not isinstance(file_name_list, List):
        file_name_list = [file_name_list]
    if not isinstance(file_content_list, List):
        file_content_list = [file_content_list]

    # loop over each file name and content pair
    for i, (name, content) in enumerate(zip(file_name_list, file_content_list)):
        # convert content to numpy array if it is a PyTorch tensor
        if isinstance(content, torch.Tensor):
            content = to_cpu(content, tgt='numpy')

        # if group_ids is not None, create subdirectory for this file's group
        if group_ids is not None:
            file_save_path = os.path.join(save_path, group_ids[i])
        else:
            file_save_path = save_path

        # make sure the saving folder exists
        os.makedirs(file_save_path, exist_ok=True)
        file_path = os.path.join(file_save_path, f'{name}.{file_format}')
        # check the existence of the target file to make sure that data saving won't fail due to the system errors that cannot be captured by Python
        while not os.path.exists(file_path):
            # save the file in the specified format
            if file_format == 'npy':
                np.save(file_path, content.astype(np.float32))

            elif file_format == 'npz':
                assert sample_rate is not None, "sample_rate must be provided for npz format"
                np.savez(file_path, feat=content.astype(np.float32), sample_rate=sample_rate)

            elif file_format == 'wav':
                assert sample_rate is not None, "sample_rate must be provided for wav format"
                sf.write(file=file_path, data=content, samplerate=sample_rate, format='WAV',
                         subtype=sf.default_subtype('WAV'))

            elif file_format == 'flac':
                assert sample_rate is not None, "sample_rate must be provided for flac format"
                sf.write(file=file_path, data=content, samplerate=sample_rate, format='FLAC',
                         subtype=sf.default_subtype('FLAC'))

            else:
                raise NotImplementedError(f"File format '{file_format}' is not supported. "
                                          f"Please use one of the supported formats: 'npy', 'npz', 'wav', 'flac'")

        # map the original file names to their corresponding file paths in disk
        name2file_path[name] = file_path

    return name2file_path
