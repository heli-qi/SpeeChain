import os
import argparse
import numpy as np

from functools import partial
from multiprocessing import Pool
from typing import List, Union

from speechain.utilbox.import_util import get_idle_gpu
from speechain.utilbox.type_util import str2list
from speechain.utilbox.data_loading_util import parse_path_args, load_idx2data_file, search_file_in_subfolder, read_data_by_path
from speechain.utilbox.spk_util import extract_spk_feat
from speechain.utilbox.md_util import save_md_report


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--hypo_path', type=str, required=True,
                        help="The path of your TTS experimental folder. All the files named 'idx2xxx_wav' and "
                             "'idx2xxx_spk' will be automatically found out and used for speaker similarity evaluation.")
    parser.add_argument('--refer_path', type=str, required=True,
                        help="The path of the ground-truth data folder. All the files named 'spk2aver_xxx_spk_feat' will be "
                             "automatically found out and used as the reference. The hypo 'idx2xxx_wav' and refer "
                             "'idx2wav' will be matched by the data indices. You can also directly specify the path of "
                             "your target 'spk2aver_xxx_spk_feat' file by this argument.")
    parser.add_argument("--spk_emb_model_list", type=str2list, default=['ecapa', 'xvector'],
                        help="Speaker recognition model to use for extracting speaker embeddings. "
                             "Must be 'xvector' or 'ecapa' or both of them. (default: ['ecapa', 'xvector'])")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="Number of CPU processes for extracting speaker embeddings. "
                             "Ignored if 'ngpu' is provided. (default: 8)")
    parser.add_argument("--ngpu", type=int, default=0,
                        help="Number of GPUs for extracting speaker embeddings. "
                             "If not provided, extraction will be done on CPUs. (default: 0)")

    return parser.parse_args()

def main(hypo_path: str, refer_path: str, spk_emb_model_list: Union[List[str], str] = ['ecapa', 'xvector'],
         batch_size: int = 10, ncpu: int = 8, ngpu: int = 0):
    """
        Processes the data from a given path using specified speaker embedding models, computes cosine similarity
        between reference and hypothesis features, and finally saves the results in Markdown format.

        Arguments:
            hypo_path (str):
                The path to the hypothesis data. It can be either a directory containing multiple files or a single file.
            refer_path (str):
                The path to the reference data. Similar to hypo_path, it can be a directory or a file.
            spk_emb_model_list (Union[List[str], str], optional):
                A list of speaker embedding models to use. The models should be either 'ecapa' or 'xvector'.
                Defaults to ['ecapa', 'xvector'].
            batch_size (int, optional):
                The size of the batch to use for feature extraction. Defaults to 10.
            ncpu (int, optional):
                The number of CPU cores to use for the processing. Defaults to 8.
            ngpu (int, optional):
                The number of GPU cores to use for the processing. Defaults to 0, meaning no GPU is used.

        Notes:
            For each speaker embedding model, the function calculates the cosine similarity between the average feature
            vectors of the reference and hypothesis for each speaker. The results are then saved in a markdown file named
            '[model_name]_similarity_results.md' in the same directory as the input hypothesis data.

            The function is multi-process friendly and will automatically distribute the work across the specified number
            of CPU cores or GPU cores.

        Raises:
            AssertionError: If the provided speaker embedding models are not 'ecapa' or 'xvector'.
            AssertionError: If the provided paths do not exist or do not contain the required data.
    """

    # --- 1. Argument Preparation stage --- #
    # argument checking
    if not isinstance(spk_emb_model_list, List):
        spk_emb_model_list = [spk_emb_model_list]
    for i in range(len(spk_emb_model_list)):
        spk_emb_model_list[i] = spk_emb_model_list[i].lower()
        assert spk_emb_model_list[i] in ['ecapa', 'xvector'], \
            f"Your input spk_emb_model should be one of ['ecapa', 'xvector'], but got {spk_emb_model_list[i]}!"

    refer_path = parse_path_args(refer_path)
    # for folder input, automatically find out all the idx2xxx_wav candidates as refer_idx2wav
    if os.path.isdir(refer_path):
        refer_list = search_file_in_subfolder(
            refer_path, lambda x: x.startswith('spk2aver') and x.endswith('spk_feat'))
        assert len(refer_list) > 0

    # for file input, directly use it as hypo_idx2wav
    else:
        assert os.path.basename(refer_path).startswith('spk2aver') and \
               os.path.basename(refer_path).endswith('spk_feat'), \
            "Please give the correct path of your target spk2aver_(xxx)_spk_feat!"
        refer_list = [refer_path]

    refer_data_per_model = {}
    for s2a_path in refer_list:
        spk_emb_model_name = os.path.basename(s2a_path).replace('spk2aver_', '').replace('_spk_feat', '')
        if spk_emb_model_name not in refer_data_per_model.keys():
            refer_data_per_model[spk_emb_model_name] = []

        refer_data_per_model[spk_emb_model_name].append(
            {spk: read_data_by_path(feat_path) for spk, feat_path in load_idx2data_file(s2a_path).items()})

    device_list = get_idle_gpu(ngpu, id_only=True) if ngpu > 0 else [-1 for _ in range(ncpu)]
    n_proc = len(device_list) if ngpu > 0 else ncpu

    hypo_path = parse_path_args(hypo_path)
    if os.path.isdir(hypo_path):
        # automatically find out all the idx2xxx_wav candidates as hypo_idx2wav
        hypo_idx2wav_list = search_file_in_subfolder(hypo_path, lambda x: x.startswith('idx2') and x.endswith('wav'))
        assert len(hypo_idx2wav_list) > 0
    else:
        assert os.path.basename(hypo_path).startswith('idx2') and \
               os.path.basename(hypo_path).endswith('wav'), \
            "Please give the correct path of your hypo idx2_(xxx)_wav!"
        hypo_idx2wav_list = [hypo_path]

    for hypo_idx2wav in hypo_idx2wav_list:
        exp_folder_path = os.path.dirname(hypo_idx2wav)
        # read the source files from a string into a Dict, str -> Dict[str, str]
        hypo_idx2spk = load_idx2data_file(os.path.join(exp_folder_path, 'idx2ref_spk'))
        hypo_idx2wav = load_idx2data_file(hypo_idx2wav).items()

        spk_list, spk2wav_dict = sorted(set(hypo_idx2spk.values())), {}
        for spk in spk_list:
            if spk not in spk2wav_dict.keys():
                spk2wav_dict[spk] = {idx: wav for idx, wav in hypo_idx2wav if hypo_idx2spk[idx] == spk}

        func_args = [[dict(list(spk2wav_dict.items())[i::n_proc]), device_list[i]] for i in range(n_proc)]
        for spk_emb_model in spk_emb_model_list:
            if not os.path.exists(os.path.join(exp_folder_path, f'{spk_emb_model}_similarity_results.md')):
                print(f'Start to create {spk_emb_model}_similarity_results.md in {exp_folder_path}!')

                # initialize the arguments for the execution function
                extract_spk_feat_func = partial(extract_spk_feat, spk_emb_model=spk_emb_model, batch_size=batch_size)

                # start the executing jobs
                with Pool(n_proc) as executor:
                    extraction_results = executor.starmap(extract_spk_feat_func, func_args)

                # gather the results from all the processes
                hypo_spk2aver_spk_feat = {}
                for _, _hypo_spk2aver_spk_feat in extraction_results:
                    hypo_spk2aver_spk_feat.update(_hypo_spk2aver_spk_feat)

                # pick up the corresponding ground-truth in refer_data_per_model[spk_emb_model]
                refer_spk2aver_spk_feat, key_match_flag, hypo_spk_keys = None, False, set(hypo_spk2aver_spk_feat.keys())
                for refer_data in refer_data_per_model[spk_emb_model]:
                    refer_spk_keys = set(refer_data.keys())
                    key_match_flag = len(hypo_spk_keys.difference(refer_spk_keys)) == 0
                    if key_match_flag:
                        refer_spk2aver_spk_feat = refer_data
                        break
                if not key_match_flag:
                    print(f"None of the refer 'spk2aver_spk_feat' in your given refer_path {refer_path} match "
                          f"{exp_folder_path}, so it will be skipped!")
                    continue

                # calculate the cosine similarity of the average feature for each speaker
                spk_similarity = {}
                for spk in refer_spk2aver_spk_feat.keys():
                    refer_feat, hypo_feat = refer_spk2aver_spk_feat[spk], hypo_spk2aver_spk_feat[spk]
                    if len(refer_feat.shape) > 1:
                        refer_feat = refer_feat.squeeze(0)
                    if len(hypo_feat.shape) > 1:
                        hypo_feat = hypo_feat.squeeze(0)

                    spk_similarity[spk] = \
                        np.dot(refer_feat, hypo_feat) / (np.linalg.norm(refer_feat) * np.linalg.norm(hypo_feat))

                save_md_report(metric_results=spk_similarity, metric_name=f'{spk_emb_model}_similarity',
                               save_path=exp_folder_path, desec_sort=False)

if __name__ == '__main__':
    args = parse()
    main(**vars(args))
