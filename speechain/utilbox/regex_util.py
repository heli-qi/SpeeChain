"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.04

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import re

# regex collection #

"""
regex_key_val function to extract pair key and value from following format 
<key><space><value>
UTT_1 FEAT_PATH_1
"""
# regex_key_val = re.compile('(^[^\s]+) (.+)$', re.MULTILINE)
regex_key_val = re.compile(r"^(^[^\s]+)\s(.*)$", re.MULTILINE)
regex_key = re.compile(r'^([^\s]+)$', re.MULTILINE)

# return all the sub-strings not including '<' and '>' that are surrounded by a pair of angle brackets
# e.g. <exp_root> √ <exp<root> × <exp>root> ×
regex_angle_bracket = re.compile(r"<[^<>]*>")