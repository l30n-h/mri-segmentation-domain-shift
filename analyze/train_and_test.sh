set -e
set -o xtrace

trainers=(
  # "MA_noscheduler_depth5_wd0_ep120"
  # "MA_noscheduler_depth5_wd0_ep120_noDA"
  # "MA_noscheduler_depth5_wd0_SGD_ep120"
  # "MA_noscheduler_depth5_wd0_SGD_ep120_noDA"
  # "MA_noscheduler_depth5_wd0_ep120_nogamma"
  # "MA_noscheduler_depth5_wd0_ep120_nomirror"
  # "MA_noscheduler_depth5_wd0_ep120_norotation"
  # "MA_noscheduler_depth5_wd0_ep120_noscaling"
  # "MA_noscheduler_depth5_wd0_SGD_ep120_nogamma"
  # "MA_noscheduler_depth5_wd0_SGD_ep120_nomirror"
  # "MA_noscheduler_depth5_wd0_SGD_ep120_norotation"
  # "MA_noscheduler_depth5_wd0_SGD_ep120_noscaling"
  # "MA_noscheduler_depth5_wd0_bn_ep120"
  # "MA_noscheduler_depth5_wd0_bn_ep120_noDA"
  # "MA_noscheduler_depth5_wd0_bn_SGD_ep120"
  # "MA_noscheduler_depth5_wd0_bn_SGD_ep120_noDA"

  # "MA_noscheduler_depth7_bf24_wd0_ep360_noDA"
  # "MA_noscheduler_depth5_ep360_noDA"
)
declare -A fold_id_mapping
fold_id_mapping['siemens15']=0
fold_id_mapping['siemens3']=1
fold_id_mapping['ge15']=2
fold_id_mapping['ge3']=3
fold_id_mapping['philips15']=4
fold_id_mapping['philips3']=5

epochs=('010' '020' '030' '040' '080' '120')
#epochs=('010' '020' '030' '040' '080' '120' '200' '280' '360')

#export RESULTS_FOLDER="$HOME/data/nnUNet_trained_models"
export RESULTS_FOLDER="$HOME/archive/old/nnUNet-container/data/nnUNet_trained_models"

for trainer in "${trainers[@]}"
do
  for fold_name in "${!fold_id_mapping[@]}"
  do
    fold_id="${fold_id_mapping[$fold_name]}"
    python code/nnUNet/nnunet/run/run_training.py 2d nnUNetTrainerV2_${trainer} Task601_cc359_all_training ${fold_id}

    for epoch in "${epochs[@]}"
    do
      if test "$(jobs | wc -l)" -ge 3; then
        wait -n
      fi
      {
        folder_name="${trainer}-ep${epoch}-${fold_name}"
        testout_dir="data/testout/${folder_name}"
        mkdir ${testout_dir}
        python code/nnUNet/nnunet/training/network_training/masterarbeit/inference/predict_preprocessed.py -f ${fold_id} -o ${testout_dir} -tr nnUNetTrainerV2_${trainer} -t Task601_cc359_all_training -m 2d -chk model_ep_${epoch} --disable_tta --num_threads_nifti_save=4 -d code/analyze/name_paths_dict_small.json
        mv ${testout_dir} archive/old/nnUNet-container/data/testout2/Task601_cc359_all_training/
      } &
    done
    wait
  done
done