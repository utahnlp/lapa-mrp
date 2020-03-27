#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts build dictionary and data into numbers, and seralize into pickle file.

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.mtool.codec.mrp import read as mrp_read
from utility.data_helper import *
from parser.Dict import *
import logging
import random
import argparse

logger = logging.getLogger("mrp_utils")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def mrp_utils_parser():
    parser = argparse.ArgumentParser(description='mrp_utils for selecting ids')

    parser.add_argument('--suffix', default=".mrp", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--input_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    return parser

parser = mrp_utils_parser()
opt = parser.parse_args()

input_files = folder_to_files_path(opt.input_folder, opt.suffix)
id_files = folder_to_files_path(opt.input_folder, ".ids")

for input_file in input_files:
    with open(input_file, 'r') as fp:
        graph_dict = {}
        for graph, _ in mrp_read(fp):
            graph_dict[graph.id] = graph
        for id_file in id_files:
            x = 0
            with open(id_file, "r") as idfp, open(id_file+".conllu","w+") as cfp:
                for line in idfp:
                    id = line.rstrip("\n")
                    g = json.dumps(graph_dict[id].encode(), indent=None, ensure_ascii = False)
                    cfp.write(g)
                    cfp.write("\n")
                    x = x +1
            logger.info("{} is written into {}".format(x, id_file+".conllu"))
