#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to run the model to parse a file. Input file should contain each sentence per line
A file containing output will be generated at the same folder unless output is specified.
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-05-30
'''

from torch import cuda
from parser.AMRProcessors import *
from src.train import read_dicts,get_parser

logger = logging.getLogger("amr.parser")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def generate_parser():
    parser = get_parser()
    parser.add_argument('--with_graphs', type=int,default=1)
    parser.add_argument("--input",default=None,type=str,
                        help="""input file path""")
    parser.add_argument("--set_wiki",default=False,type=bool,
                        help="""whether predict wiki""")
    parser.add_argument("--text",default=None,type=str,
                        help="""a single sentence to parse""")
    parser.add_argument('--amr_preprocess', default=False, type=bool,
                        help="""weather to do amr preprocess""")
    return parser

if __name__ == "__main__":
    global opt
    opt = generate_parser().parse_args()
    opt.lemma_dim = opt.dim
    opt.high_dim = opt.dim
    opt.cuda = len(opt.gpus)

    logger.info(opt)

    if opt.cuda and opt.gpus[0] != -1:
        cuda.set_device(opt.gpus[0])
    dicts = read_dicts(opt.build_folder)

    Parser = AMRParser(opt,dicts)
    if opt.input.endswith(".mrp_conllu"):
        filepath = opt.input
        out_mrp = filepath+"_parsed_mrp"
        out_txt = filepath+"_parsed_txt"
        logger.info("processing {}".format(filepath))
        n = 0
        with open(out_txt,'w+') as out_txt_f, open(out_mrp,'w+') as out_mrp_f:
            dataset = readFeaturesInput([filepath])
            for id, data in dataset.items():
                input_snt = data["input_snt"]
                if opt.amr_preprocess:
                    new_data = Parser.feature_extractor.preprocess(line, whiteSpace=False) #phrase from fixed joints.txt file
                    new_data['example_id'] = data['example_id']
                    data = new_data

                output_mrp_graph,output = Parser.parse_one_preprocessed_data(data, opt.set_wiki, opt.normalize_mod)
                # write txt amr
                out_txt_f.write("# ::snt "+input_snt+ "\n")
                out_txt_f.write(output[0])
                out_txt_f.write("\n")

                # write mrp
                out_mrp_f.write(json.dumps(output_mrp_graph[0].encode(), indent=None, ensure_ascii = False))
                out_mrp_f.write("\n")
        logger.info("done processing {}".format(filepath))
    elif opt.input.endswith(".txt"):
        filepath = opt.input
        out_mrp = filepath+"_parsed_mrp"
        out_txt = filepath+"_parsed_txt"
        logger.info("processing {}".format(filepath))
        n = 0
        with open(out_txt,'w+') as out_txt_f, open(out_mrp,'w+') as out_mrp_f:
            with open(filepath,'r') as f:
                line = f.readline()
                while line != '' :
                    if line.strip() != "":
                        output_mrp_graph,output = Parser.parse_batch([line.strip()], opt.set_wiki, opt.normalize_mod)
                        out_txt_f.write("# ::snt "+line + "\n")
                        out_txt_f.write(output[0])
                        out_txt_f.write("\n")
                        # write mrp
                        out_mrp_f.write(json.dumps(output_mrp_graph[0].encode(), indent=None, ensure_ascii = False))
                        out_mrp_f.write("\n")
        logger.info("done processing {}".format(filepath))
    elif opt.text:
        output = Parser.parse_one(opt.text, opt.set_wiki, opt.normalize_mod)
        # actually there is only one snt, and one elem in output
        logger.info("Parsing Result:\n # ::snt {}\n{}".format(opt.text, output))
        # actually there is only one snt, and one elem in output
    else:
        logger.info("option -input [file] or -text [sentence] is required.")
