"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.04
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
regex_key = re.compile('^([^\s]+)$', re.MULTILINE)
