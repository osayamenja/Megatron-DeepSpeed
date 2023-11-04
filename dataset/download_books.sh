#!/bin/bash
set -ex
## The links below are invalid due to copyright
# wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
# wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx

DIR=$(pwd)

if ! test -f "${DIR}/gpt2-vocab.json" || ! test -f "${DIR}/gpt2-merges.txt" ; then
  source "${DIR}/download_vocab.sh"
fi

#DATASET_FILES_DIR="dataset-test"
#if ! test -d "${DIR}/${DATASET_FILES_DIR}"; then
#  mkdir -p ${DATASET_FILES_DIR}
#fi

# needed for extraction
apt install zstd

# monology/pile-uncopyrighted from Hugging Face.
# Below downloads, extracts and preprocesses the train, test and eval datasets

dataset_names=("00.jsonl" "test.jsonl" "val.jsonl")
output_file_names=("pile_uncopyrighted_train" "pile_uncopyrighted_test" "pile_uncopyrighted_val")

for i in "${!dataset_names[@]}"; do
  URL_KEY=${dataset_names[$i]}
   # prevents recurrent downloads since files are huge.
  if ! test -f "${DIR}/${dataset_names[$i]}.zst"; then

    if ((i == 0)); then
      echo "Downloading training dataset..."
      URL_KEY="train/${dataset_names[$i]}"
    fi
    URL_KEY="${URL_KEY}.zst"
    run_cmd="wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/${URL_KEY}"
    printf "%s\n" "${run_cmd}"
    eval "${run_cmd}"
    printf "Download finished!\n"
  fi

  # Do only once as it takes time to extract since files are huge.
  if ! test -f "${DIR}/${dataset_names[$i]}"; then
    printf "Extracting...\n"
    run_cmd="unzstd ${dataset_names[$i]}.zst"
    echo "${run_cmd}"
    eval "${run_cmd}"
  fi

  printf "preprocessing...\n"
  # get number of CPU cores
  WORKERS=$(< /proc/cpuinfo grep -c processor)
  run_cmd="python3 ./../tools/preprocess_data.py \
       --input ${dataset_names[$i]} \
       --output-prefix ${output_file_names[$i]} \
       --vocab-file ${DIR}/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ${DIR}/gpt2-merges.txt \
       --workers ${WORKERS} \
       --append-eod"
  eval "${run_cmd}"
  rm "${dataset_names[$i]}.zst" # remove unneeded file
done
