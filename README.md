# LAPA-MRP

******************

This repo is our system submission to MRP 2019 shared task at CoNLL 2019: LAPA-MRP. Our model ranked 1st in the AMR subtask, 5th in UCCA, 6th in PSD and  7th in DM.
Please cite our paper [Amazon at MRP 2019: Parsing Meaning Representation with Lexical and Phrasal Anchoring](https://www.aclweb.org/anthology/K19-2013/) when using the code.

```
@article{cao2019amazon,
  title={Amazon at MRP 2019: Parsing Meaning Representations with Lexical and Phrasal Anchoring},
  author={Cao, Jie and Zhang, Yi and Youssef, Adel and Srikumar, Vivek},
  journal={CoNLL 2019},
  pages={138},
  year={2019}
}
```


# Usage

******************

## Required Software

   - Install pyenv or other python environment manager

   In our case, we use pyenv and its plugin pyenv-virtualenv to set up
   the python environment. Please follow the detailed steps in
   https://github.com/pyenv/pyenv-virtualenv for details. Alternative
   environments management such as conda will be fine.

   - Install required packages

   ```bash
   pyenv install 3.6.5
   # in our default setting, we use `pyenv activate py3.6.5_torch` to
   # activate the envivronment, please change this according to your preference.

   pyenv virtualenv 3.6.5 py3.6.5_torch
   pyenv activate py3.6.5_torch
   ```

   - Checkout this project and install the requirements

   ```bash
       git clone git@github.com:utahnlp/lapa-mrp.git lapa-mrp
   ```
   `src` and `parser` folder is the source code directory for nerual models.

   `Expt` folder is a folder for experiment managing, which includes all the commands(Expt/mrp_scripts/commands), config files(Expt/mrp_scripts/configs) to launch the experiments, and store all experiment outputs. In this repo, except `Expt/mrp_scirpts/commands/env.sh` contains the global variables, all model hyperparameters and reltaed configurations will be assigned in the config files in Expt/mrp_scripts/configs, each of them is corresponding to a model. For a detailed description for folders in `Expt` folder, please refer to [Expt README file](Expt/README.md)


## Data Preparing

### Data placement

 - MRP 2019 dataset download and placement

 Download all the mrp datasets into `Expt/data/mrp_data_ro/download/mrp_data/`

Then unzip it in place, it will generate a folder into `Expt/data/mrp_data_ro/download/mrp_data/mrp_data/`
The all related files for mrp2019 will be stored in the mrp folder as the original folder structure as follows:

```
mrp
└── 2019
    ├── companion
    ├── evaluation
    ├── sample
    └── training

```

`companion` is the companion tokenization, POS, Lemma,  dependency parsing dataset for all the training data.
`training` contains all the original mrp files for training the parser.

```
training
├── amr
├── dm
├── eds
├── Makefile
├── psd
├── README.txt
├── README.ucca.txt
└── ucca
```

 See more details on "http://mrp.nlpl.eu/2019/index.php?page=4"


### Data splitting

Since the mrp 2019 dataset didn't offer a training/dev/test splits, and the final evluation dataset is heldout during competition.
We need to split the training dataset byourself. For SDP data, we follow the previous SDP dataset, we use 00-19 as training, 20 as dev set.
For AMR, and UCCA, we use 25:1:1 for splits

We offered a script to generate the training/dev/test/ splits ids, and we also offer the ids we used for training/dev/test in the code repo.

`mrp_scripts/commands/mrp_split_amr.sh` and `mrp_scripts/commands/mrp_split_amr_conllu.sh`

You can specify our ids as input to generate the training/dev/test splits, otherwise, it will generate a random splits.



### Data Preprocessing

For AMR, we first provide a script `amr_prep_rules_data.sh`, which includes `preprocessing_amr.sh`, `amr_rule_system_build.sh`, and `amr_data_build.sh`.

```bash
pushd Expt/mrp__scripts/commands
./amr_prep_rules_data.sh ${sub_name}
popd
```
 This script will generate a pickle file with some dictionaries as the input for our model, which will be stored at `Expt/data/mrp-data/${sub_name}`.
 ${sub_name} is any unique name to identify the preporocessing setting. For example, you can store "amr_bert_large_uncased" and "amr_bert_large_cased" for different tokenization.

For DM and PSD, we also provide the following scripts for preprocessingA, we also partially support EDS preprocessing and parsing in our lexical-anchoring framework, but the main code of it are still under developing.
```bash
pushd Expt/mrp__scripts/commands
./dm_preprocessing.sh ${sub_name}
./dm_prep_rules_data.sh ${sub_name}
./psd_preprocessing.sh ${sub_name}
./psd_prep_rules_data.sh ${sub_name}
popd
```


| MR      | config_file         | Ours((P/R/F1) | MRP TOP1/3/5(F1)  |
|---------|---------------------|---------------|-------------------|
| AMR(1)  | [LatentAlignment+charEmb](Expt/mrp_scripts/configs/base_amr/base_t1_amr_ori_char.sh)                    | 75/71/73.38   | 72.94/71.97/71.72 |
| PSD(6)  | [ExplicitAlignment+charEmb](Expt/mrp_scripts/configs/base_psd/base_t1_psd_char.sh)                    | 89/89/88.75   | 90.76/89.91/88.77 |
| DM(7)   | [ExplicitAlignment+charEmb](Expt/mrp_scripts/configs/base_dm/base_t1_dm_char.sh)                   | 93/92/92.14   | 94.76/94.32/93.74 |
| UCCA(5) | [ELMo-self-attentive](https://github.com/nikitakit/self-attentive-parser) | 76/68/71.65   | 81.67/77.80/73.22 |
| EDS     | N/A                 | N/A           | 94.47/90.75/89.10 |


For UCCA, we use the phrasal-anchoring framework. We first transform a UCCA graph into a PTB tree, then we use the offshelf [consistutent tree parser](https://github.com/nikitakit/self-attentive-parser)  to parse the PTB, finally we transform it back to UCCA. Hence, the preprocessing for UCCA is mainly to transform them into PTB tree. Hence, the preprocessing for UCCA has the following two steps:

```bash
./ucca_preprocessing.sh ${sub_name}
./ucca_tree_build.sh ${sub_name}
```


## Training and Evalution 

After we generate the processed data by running the above preprocessing scripts. We can start train our models for each of the tasks (We don't use multitask learning in this repo, but the universal lexical-anchoring framework makes it easy for future extensions)

For AMR, DM, PSD training, they share the same framework, the only difference is that AMR using latent alignment, while DM and PSD use the explicit alignment. Hence, we offer a universal training and evaluation script for the different models, it takes a config file as the input to support different configurations for different task as the script `full_pipe.sh`:

```bash
pushd $CODE_BASE/mrp/Expt/mrp-scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./train.sh $config_file
model_name=gpus_${CUDA_VISIBLE_DEVICES}valid_best.pt
echo "load model"${model_name}
./test_prebuild.sh ${config_file} ${model_name}
./mtool.sh ${config_file} ${frame}
popd
```

`${config_file}` is the path of config file. All the config files are listed in `mrp_scripts/configs/`, more details for the configuration file, please see the comments in the configure file for more details. ${frame} is the value in mtools, which are amr, dm, psd, ucca, eds.


For UCCA training, we use the offshelf [consistutent tree parser](https://github.com/nikitakit/self-attentive-parser), the submitted version is using the ELMo-based self-attentive encoder. We only offer the preprocessing code used in the above preprocessing steps. For training steps, please refer to t
he original repo for details.


# Code and Contribution Guidelines

******************

## `src`

This folder stores the source code for all the commands which can be directly launched. When you want to add a specific command, then just add it here.
It may call other commands or modules in the other folders.

## `parser`

This folder contains the core code for parsing. For each task, it must have the following code:

- `XXXProcessors.py`: this file include all the preprocessing, post processing, parser class,  such as `XXXInputPreprocessor` which add lem, ner, pos to the input dataset; `XXXDecoder`, which decoder the concepts, relations with the probabilities, and then connect them into a graph; `XXXParser` is a class for evaluation and command usage, initialize a parser with a model, which can directly parse a sentence into a graph, which is useful for evaluation, an API or demo.


- `models/XXXConceptModel.py`: It is the concept identification model in graph-based parsing. In each file, a pytorch module called XXX_Concept_Classifier should be implemented. 

- `models/XXXRelModel.py`: It is the relation encoder model in graph-based parsing, which involve the root identification, and relation classication.
Hence, it require to implement the modules like `XXXRootEncoder`, `XXXRelEEncoder`

- `models/MultiPassRelModel.py`: It is the relation identification model, which depends on the above `XXXRelModel.py`, but shared by different tasks. It is based on a biaffine classifier model.

- `models/XXXGraphModel.py`: the code for variational inference for the latent alignment and explicit alignment.

- `DataIterator.py` is the dataset utils for wrap prepared dataset into inputs

- `BertDataiterator.py` is the dataset utils for Bert preprocessed dataset, the main difference is the token2index map, which are need to tranform bert tokenization back to the original token

- `Optim.py`: all about the optimization

- `modules`: some helper like bert utils, char_utils, GumbelSoftmax utils, and all kinds of encoders.


## `utility`

This folder contains all the utils for different meaning representations. Each of them are stored in the `${frame}_utils` folder.
