
EXPERIMENT_NAME="msrvtt"  # the name can be msrvtt, ucf, activityNet
PROJ_ROOT=""                        # root directory for saving experiment logs
DATADIR="/dataset/"$EXPERIMENT_NAME    # dataset directory

CONFIG="configs/train_vdm_autoencoder.yaml"

# run
export TOKENIZERS_PARALLELISM=false
python main.py \
--base $CONFIG \
-t --gpus 0, \
--name $EXPERIMENT_NAME \
--logdir $PROJ_ROOT \
--auto_resume True \
lightning.trainer.num_nodes=1 \
data.params.train.params.data_root=$DATADIR \
data.params.train.params.dataset_name=$EXPERIMENT_NAME \
data.params.validation.params.dataset_name=$EXPERIMENT_NAME \
data.params.validation.params.data_root=$DATADIR \
model.params.first_stage_config.params.ckpt_path=$AEPATH
