#!/bin/bash
set -ex

DIR=$(pwd)

if ! test -f "${DIR}/gpt2-vocab.json" || ! test -f "${DIR}/gpt2-merges.txt" ; then
  source "${DIR}/download_vocab.sh"
fi

# allenai/C4/en from Hugging Face.
# Below downloads, extracts and preprocesses the datasets

dataset_names=("c4-train.00000-of-01024.json")
output_file_names=("c4-train.00000-of-01024.json")

for i in "${!dataset_names[@]}"; do
  URL_KEY=${dataset_names[$i]}
   # prevents recurrent downloads since files are huge.
  if ! test -f "${DIR}/${dataset_names[$i]}.gz"; then
    URL_KEY="${URL_KEY}.gz"
    run_cmd="wget https://huggingface.co/datasets/allenai/c4/resolve/main/en/${URL_KEY}"
    printf "%s\n" "${run_cmd}"
    eval "${run_cmd}"
    printf "Download finished!\n"
  fi

  # Do only once as it takes time to extract since files are huge.
  if ! test -f "${DIR}/${dataset_names[$i]}"; then
    printf "Extracting...\n"
    run_cmd="gzip -d ${dataset_names[$i]}.gz"
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
