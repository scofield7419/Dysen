
PROJ_ROOT=""                      # root directory for saving experiment logs
EXPERIMENT_NAME="msrvtt"          # experiment name, which can be ucf, msrvtt, activityNet
DATADIR="/dataset/"EXPERIMENT_NAME  # dataset directory
CKPT_PATH="models/ae/ae_"$EXPERIMENT_NAME".ckpt"    # pretrained video autoencoder checkpoint

CONFIG="configs/train_vdm.yaml"

# run
export TOKENIZERS_PARALLELISM=false
python main.py \
--base $CONFIG \
-t --gpus 0, \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume True \
lightning.trainer.num_nodes=1 \
data.params.train.params.data_root=$DATADIR \
data.params.train.params.dataset_name=$EXPERIMENT_NAME \
data.params.validation.params.dataset_name=$EXPERIMENT_NAME \
data.params.validation.params.data_root=$DATADIR \
model.params.first_stage_config.params.ckpt_path=$CKPT_PATH

