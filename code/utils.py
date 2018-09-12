import os
import re


def file_findall(dir_path, name_regex):
    return [file_name for file_name in os.listdir(dir_path) 
            if re.fullmatch(name_regex, file_name)]