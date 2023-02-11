import argparse
import os
import re
import torch

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from typing import List, Dict
from collections import OrderedDict

from speechain.utilbox.import_util import parse_path_args, get_idle_gpu
from speechain.utilbox.data_loading_util import search_file_in_subfolder
from speechain.utilbox.type_util import str2list, str2dict


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model_file_path', type=str2list, required=True,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--para_mapping', type=str2dict, required=True,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--save_path', type=str, default=None,
                        help="The source folder where your target files are placed. (default: None)")
    parser.add_argument('--ncpu', type=int, default=1,
                        help="The number of processes you want to use to do the job. (default: 1)")
    return parser.parse_args()


def modify_para_and_save(model_file_path: List[str], para_mapping: Dict, save_path: str = None):
    for m_f_p in tqdm(model_file_path):
        model_para = torch.load(m_f_p, map_location=torch.device('cpu'))
        _src_modules = OrderedDict()
        # loop each name-parameter pair in the model
        for name, para in model_para.items():
            # loop each source-target mapping pair
            for src, tgt in para_mapping.items():
                # . at the tails is for making the name unique
                src, tgt = src + '.', tgt + '.'
                # change the parameter name in the middle
                if src in name:
                    name = name.replace(src, tgt)
            # record the parameter no matter whether its name is modified or not
            _src_modules[name] = para

        model_file_dir, model_file_name = os.path.dirname(m_f_p), os.path.basename(m_f_p)
        torch.save(_src_modules, os.path.join(model_file_dir if save_path is None else save_path, model_file_name))


def main(model_file_path: str or List[str], para_mapping: Dict, save_path: str = None, ncpu: int = 1):

    if save_path is not None:
        save_path = parse_path_args(save_path)

    if isinstance(model_file_path, str):
        model_file_path = parse_path_args(model_file_path)
        if os.path.isdir(model_file_path):
            model_file_path = search_file_in_subfolder(model_file_path,
                                                       lambda x: x.endswith('.pth') and x != 'checkpoint.pth')
        else:
            model_file_path = [model_file_path]
    else:
        model_file_path = [parse_path_args(m_f_p) for m_f_p in model_file_path]

    # initialize the arguments for the execution function
    modify_para_and_save_func = partial(modify_para_and_save, para_mapping=para_mapping, save_path=save_path)
    func_args = [model_file_path[i::ncpu] for i in range(ncpu)]

    # start the executing jobs
    with Pool(ncpu) as executor:
        executor.map(modify_para_and_save_func, func_args)


if __name__ == '__main__':
    # args = parse()
    # main(**vars(args))

    main(
        model_file_path='recipes/offline_tts2asr/libritts_librispeech',
        para_mapping={
            'feed_forward.fdfwd_layers.0': 'feed_forward.in_layer',
            'feed_forward.fdfwd_layers.3': 'feed_forward.out_layer'
        },
        ncpu=8
    )

