EXPERIMENT_NAME='activityNet'
RESUME=''  # for the save log data
DATADIR='./dataset/'$EXPERIMENT_NAME # for the training data path
CKPT_PATH='./models/'$EXPERIMENT_NAME
VDM_MODEL='model.ckpt'

CONFIG="configs/train_vdm.yaml"

# run
export OPENAI_API_KEY=""
python scene_managing.py \
user_options.config=$CONFIG\
user_options.resume=$RESUME\
data.data_root=$DATADIR \
reinforce_settings.ckpt_root=$CKPT_PATH
