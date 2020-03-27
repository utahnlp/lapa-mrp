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
    parser = argparse.ArgumentParser(description='mrp_utils for selecting ids, parition')

    parser.add_argument('--suffix', default=".mrp", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--input_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    parser.add_argument('--training_ids', default="", type=str,
                        help="""ids must be in training set""")
    parser.add_argument('--dev_ids', default="", type=str,
                        help="""ids must be in dev set""")
    parser.add_argument('--test_ids', default="", type=str,
                        help="""ids must be in dev set""")
    parser.add_argument('--follow_ids_only', type=bool, default=False,
                                help="whether just follow ids for spliting")
    return parser

parser = mrp_utils_parser()
opt = parser.parse_args()

input_files = folder_to_files_path(opt.input_folder, opt.suffix)

if opt.follow_ids_only:
    assert (opt.training_ids or opt.dev_ids or opt.test_ids), "follow_ids requires offer ids for at least one split"

if opt.training_ids:
    with open(opt.training_ids, "r") as fp:
        train_ids = [line.rstrip('\n') for line in fp]
else:
    train_ids = []


if opt.dev_ids:
    with open(opt.dev_ids, "r") as fp:
        dev_ids = [line.rstrip('\n') for line in fp]
else:
    dev_ids = []

if opt.test_ids:
    with open(opt.test_ids, "r") as fp:
        test_ids = [line.rstrip('\n') for line in fp]
else:
    test_ids = []


all_ids = []
for input_file in input_files:
    train_set = []
    dev_set = []
    test_set = []
    remaining_set = []
    with open(input_file, 'r') as fp:
        graphs = list(mrp_read(fp))

    # for the framework, a graph id may duplicate more than once.
    # only select those haven't been used graphs
    deduplicated_graphs = []
    for graph, _ in graphs:
        if graph.id not in all_ids:
            deduplicated_graphs.append(graph)
            all_ids.append(graph.id)
        else:
            continue

    total = len(deduplicated_graphs)
    if opt.follow_ids_only:
        train_total = total
        dev_total = total
        test_total = total
    else:
        train_total = int(total * 0.925+0.5)
        dev_total = int(total*0.0375+0.5)
        test_total = total - train_total - dev_total

    for graph in deduplicated_graphs:
        is_train_available = (len(train_set) < train_total)
        is_dev_available = (len(dev_set) < dev_total)
        is_test_available = (len(test_set) < test_total)
        if graph.id in train_ids and is_train_available:
            train_set.append(graph)
        elif graph.id in dev_ids and is_dev_available:
            dev_set.append(graph)
        elif graph.id in test_ids and is_test_available:
            test_set.append(graph)
        else:
            remaining_set.append(graph)
    logger.info("total = {}, reamining_set =  {}, train_set = {}, dev_set={}, test_set = {}".format(total, len(remaining_set),len(train_set), len(dev_set), len(test_set)))

    if not opt.follow_ids_only:
        random.shuffle(remaining_set)
        for graph in remaining_set:
            remaining_train_available = train_total - len(train_set)
            remaining_dev_available = dev_total - len(dev_set)
            remaining_test_available = test_total - len(test_set)
            if remaining_train_available > 0:
                train_set.append(graph)
            elif remaining_dev_available > 0:
                dev_set.append(graph)
            else:
                test_set.append(graph)

    # write out different set
    train_split_file = input_file+".train"
    train_split_ids_file= input_file+".train_ids"
    with open(train_split_file, 'w') as out_f, open(train_split_ids_file, 'w') as id_f:
        for graph in train_set:
            out_f.write(json.dumps(graph.encode(), indent=None, ensure_ascii = False))
            out_f.write("\n")
            id_f.write(graph.id)
            id_f.write("\n")

        logger.info("training split for {} :{}".format(train_split_file,len(train_set)))

    dev_split_file = input_file+".dev"
    dev_split_ids_file= input_file+".dev_ids"
    with open(dev_split_file, 'w') as out_f, open(dev_split_ids_file, 'w') as id_f:
        for graph in dev_set:
            out_f.write(json.dumps(graph.encode(), indent=None, ensure_ascii = False))
            out_f.write("\n")
            id_f.write(graph.id)
            id_f.write("\n")

        logger.info("dev split for {} :{}".format(dev_split_file,len(dev_set)))

    test_split_file = input_file+".test"
    test_split_ids_file= input_file+".test_ids"
    with open(test_split_file, 'w') as out_f, open(test_split_ids_file, 'w') as id_f:
        for graph in test_set:
            out_f.write(json.dumps(graph.encode(), indent=None, ensure_ascii = False))
            out_f.write("\n")
            id_f.write(graph.id)
            id_f.write("\n")

        logger.info("test split for {} :{}".format(test_split_file,len(test_set)))
