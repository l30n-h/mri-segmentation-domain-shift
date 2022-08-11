set -e
set -o xtrace

trainer_base='MA_noscheduler_depth5_wd0'
trainers=(
  "${trainer_base}_ep120"
  "${trainer_base}_ep120_noDA"
  "${trainer_base}_SGD_ep120"
  "${trainer_base}_SGD_ep120_noDA"
  #"${trainer_base}_ep120_nogamma"
  #"${trainer_base}_ep120_nomirror"
  #"${trainer_base}_ep120_norotation"
  #"${trainer_base}_ep120_noscaling"
  #"${trainer_base}_SGD_ep120_nogamma"
  #"${trainer_base}_SGD_ep120_nomirror"
  #"${trainer_base}_SGD_ep120_norotation"
  #"${trainer_base}_SGD_ep120_noscaling"
)
declare -A fold_id_mapping
fold_id_mapping['siemens15']=0
fold_id_mapping['siemens3']=1
fold_id_mapping['ge15']=2
fold_id_mapping['ge3']=3
fold_id_mapping['philips15']=4
fold_id_mapping['philips3']=5
epochs=('010' '020' '030' '040' '080' '120')


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
        #python code/nnUNet/nnunet/training/network_training/masterarbeit/inference/predict_simple.py -f ${fold_id} -o ${testout_dir} -tr nnUNetTrainerV2_${trainer} -i data/nnUNet_raw/nnUNet_raw_data/Task601_cc359_all_training/imagesTs/ -t Task601_cc359_all_training -m 2d -chk model_ep_${epoch} --disable_tta --num_threads_nifti_save=4
        python code/nnUNet/nnunet/training/network_training/masterarbeit/inference/predict_preprocessed.py -f ${fold_id} -o ${testout_dir} -tr nnUNetTrainerV2_${trainer} -t Task601_cc359_all_training -m 2d -chk model_ep_${epoch} --disable_tta --num_threads_nifti_save=4
        mv ${testout_dir} archive/old/nnUNet-container/data/testout/Task601_cc359_all_training/
      } &
    done
    wait
  done
done