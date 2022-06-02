set -e
set -o xtrace

trainer_base='MA_noscheduler_depth5_wd0'
trainers=(
  "${trainer_base}_ep120"
  "${trainer_base}_ep120_noDA"
  "${trainer_base}_SGD_ep120"
  "${trainer_base}_SGD_ep120_noDA"
)
#fold_names=('siemens15' 'siemens3' 'ge15' 'ge3' 'philips15' 'philips3')
#epochs=('010' '020' '030' '040' '080' '120')

fold_names=('siemens15' 'siemens3' 'ge15' 'ge3')
epochs=('040' '120')

export MA_USE_TEST_HOOKS=TRUE

for trainer in "${trainers[@]}"
do
  for fold_i in "${!fold_names[@]}"
  do
    fold_name="${fold_names[$fold_i]}"

    for epoch in "${epochs[@]}"
    do
      if test "$(jobs | wc -l)" -ge 1; then
        wait -n
      fi
      {
        folder_name="${trainer}-ep${epoch}-${fold_name}"
        testout_dir="data/testout/${folder_name}"
        mkdir ${testout_dir}
        python code/nnUNet/nnunet/training/network_training/masterarbeit/inference/predict_simple.py -f ${fold_i} -o ${testout_dir} -tr nnUNetTrainerV2_${trainer} -i data/nnUNet_raw/nnUNet_raw_data/Task601_cc359_all_training/imagesTs-small/ -t Task601_cc359_all_training -m 2d -chk model_ep_${epoch} --disable_tta --num_threads_nifti_save=4
        mv ${testout_dir}/activations archive/old/nnUNet-container/data/testout/Task601_cc359_all_training/${folder_name}/activations-small
      } &
    done
    wait
  done
done