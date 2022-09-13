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
)
declare -A fold_id_mapping
fold_id_mapping['siemens15']=0
fold_id_mapping['siemens3']=1
fold_id_mapping['ge15']=2
fold_id_mapping['ge3']=3
# fold_id_mapping['philips15']=4
# fold_id_mapping['philips3']=5

epochs=('010' '020' '030' '040' '080' '120')
#epochs=('010' '020' '030' '040' '080' '120' '200' '280' '360')

#export RESULTS_FOLDER="$HOME/data/nnUNet_trained_models"
export RESULTS_FOLDER="$HOME/archive/old/nnUNet-container/data/nnUNet_trained_models"

#testset_suffix=''
testset_suffix='-small'
activations_suffix="${testset_suffix}-fullmap"

export MA_USE_TEST_HOOKS=TRUE

for trainer in "${trainers[@]}"
do
  for fold_name in "${!fold_id_mapping[@]}"
  do
    fold_id="${fold_id_mapping[$fold_name]}"

    for epoch in "${epochs[@]}"
    do
      if test "$(jobs | wc -l)" -ge 1; then
        wait -n
      fi
      {
        folder_name="${trainer}-ep${epoch}-${fold_name}"
        testout_dir="data/testout/${folder_name}"
        archive_testout_dir="archive/old/nnUNet-container/data/testout/Task601_cc359_all_training/${folder_name}"
        mkdir ${testout_dir}
        python code/nnUNet/nnunet/training/network_training/masterarbeit/inference/predict_preprocessed.py -f ${fold_id} -o ${testout_dir} -tr nnUNetTrainerV2_${trainer} -k code/analyze/ids_small.json -t Task601_cc359_all_training -m 2d -chk model_ep_${epoch} --disable_tta --num_threads_nifti_save=4
        rm -rf ${archive_testout_dir}/activations${activations_suffix}
        mv -f ${testout_dir}/prediction_raw/activations ${archive_testout_dir}/activations${activations_suffix}
        mv -f ${testout_dir}/prediction_raw/prediction_args.json ${archive_testout_dir}/prediction_args${testset_suffix}.json
        mv -f ${testout_dir}/prediction_raw/summary.json ${archive_testout_dir}/summary${testset_suffix}.json
      } &
    done
    wait
  done
done