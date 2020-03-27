#!/bin/bash

# source the environment script
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUILD_DIR=$ROOT_DIR/build/amr2eng/
BUILD_BACKUP_DIR=$ROOT_DIR/build/backup/amr2eng/

if [ -e $BUILD_DIR ]; then
    read -p "$BUILD_DIR existed, rebuild it:(y/n) " userInput
    if [[ $userInput == 'y' ]]; then
        today=`date +%m-%d.%H:%M`
        mv -i ${BUILD_DIR} ${BUILD_BACKUP_DIR%?}_${today}
        echo "rename original build folder to "${BUILD_BACKUP_DIR%?}_${today}
        mkdir -p $BUILD_DIR
        mkdir -p $BUILD_DIR/input/training
        mkdir -p $BUILD_DIR/input/dev
        mkdir -p $BUILD_DIR/input/test
    else:
        echo "Just use the original folder ${BUILD_DIR}"
    fi
fi

# Make sure when you try to use amr2eng, use the split folder pointing to the isi alignment folder in the AMR repo, and change the constants.py
pushd $ROOT_DIR/AMR_FEATURE/
if [ -d./bin/stanford-corenlp-full-2018-10-25/ ]; then
  echo "stanford zip existed"
else
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip -P ./bin/
    unzip stanford-corenlp-full-2018-10-05.zip
fi


if [ -f ${BUILD_DIR}/input/training/training.combined.txt_pre_processed ]; then
  echo "${BUILD_DIR}/input/training existed"
else
    javac src/convertingAMR.java -cp "bin/stanford-corenlp-full-2018-10-05/*:bin/json-20170516.jar" -d bin/
    java -Damr.split_folder=$CODE_BASE/amr_data/e25/data/alignments/split/ -Damr.build_folder=${BUILD_DIR} -cp "bin/stanford-corenlp-full-2018-10-05/*:bin/json-20170516.jar:bin/." convertingAMR &> $ROOT_DIR/Expt/pre_logs/amr2eng.log
fi

# remove the head description lines, starting with # AMR-ENG
echo "remove the head description lines, starting with # AMR-ENG"
sed -i.bak '/\# AMR-English/d' ${BUILD_DIR}/input/training/training.combined.txt_pre_processed
sed -i.bak '/\# AMR-English/d' ${BUILD_DIR}/input/dev/dev.combined.txt_pre_processed
sed -i.bak '/\# AMR-English/d' ${BUILD_DIR}/input/test/test.combined.txt_pre_processed
popd


