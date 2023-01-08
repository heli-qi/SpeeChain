${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/dev-clean/ecapa --sum_file_name idx2ecapa_spk_feat
${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/dev-clean/xvector --sum_file_name idx2xvector_spk_feat
${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/dev-other/ecapa --sum_file_name idx2ecapa_spk_feat
${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/dev-other/xvector --sum_file_name idx2xvector_spk_feat

${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/test-clean/ecapa --sum_file_name idx2ecapa_spk_feat
${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/test-clean/xvector --sum_file_name idx2xvector_spk_feat
${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/test-other/ecapa --sum_file_name idx2ecapa_spk_feat
${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/utilbox/folder_summarizer.py --src_folder datasets/libritts/data/wav/test-other/xvector --sum_file_name idx2xvector_spk_feat