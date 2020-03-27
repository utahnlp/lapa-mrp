# Experiment Workspace: `Expt`

   `Expt` is the workspace for our experiments. Initially, it contains the following 3 main folders,
    We breifly give an overview of each of them, and the subfolders in them.

## `mrp_scripts`

    It contains all the bash scripts that organize the running of our
   experiments.  Worth to menthion, our principle of running neural
   experiments is seperating the reciept from its cook tool.  Hence,
   all cook tools are placed in `mrp_scripts/commands/`, and all cook
   receipts are placed in `mrp_scripts/configs`.

   During developing, build the core of cook tools in `tensorflow`
   code directory, and then wrap that as a configurable bash script in
   `mrp_scrpts/commands`. Then, write various receipts by setting
   different switch or hyperparamters. Finally, a cook tool(command)
   feeded by a receipt(config) will build a model as you wish.

    Worth to mention, a `env.sh` in commands folder is a special
   initialization code, which will set the important global variables
   that will used in our model.  It will be fine to run the code
   without any customization on env.sh However, please check the
   details in Expt/commands/env.sh script, which contains the global
   variables in our model. Once you checkout the code, all the
   environment varibles will be set with the relative path in the
   env.sh script

## `data`

   - `mrp_data_ro`

    It is a folder for read-only data, once data generated in this
   folder, it will keep unchanged, e.g. glove pretrained embedding,
   original data set, data splits etc.

   - `mrp_data`

   It the proprocessed data for traning, including dictionaries, tokenization, prepared training data.
   In our case, we will generate seperate folders for different MRP datasets

## `pre_logs`

    logs fiels for preprocessing all the datasets


## `workdir`

   It is folder to store all the experiment results organized by per folder one modeling receipt.

   - training.log or training_restore.log

   generated from `./train.sh` or `./train_restore.sh`, it shows all the training logs.
   Each dataset will have a seperate folders, which are corresponding to the folder structure of the config files in `mrp_scripts`

   - **results**

    It will store the results for dev and test set, and the their evluation scores by mtool.

   - **models**

    models will save all the best models.

   - **summary**

   This folder is for writing event and all kinds summaries, that can be read by tensorboard.
