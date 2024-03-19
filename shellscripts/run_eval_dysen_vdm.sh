EXPERIMENT_NAME="msrvtt"
DATACONFIG="configs/train_vdm.yaml"
PREDCITPATH='${Your_Path}/2048x16x256x256x3-samples.npz'
GOLDPATH='/dataset/'$EXPERIMENT_NAME
RESDIR='results/'$EXPERIMENT_NAME'/fvd'

mkdir -p $res_dir
python scripts/eval_cal_fvd_kvd.py \
    --yaml ${DATACONFIG} \
    --gold_path ${GOLDPATH} \
    --predict_path ${PREDCITPATH} \
    --batch_size 32 \
    --num_workers 4 \
    --n_runs 10 \
    --res_dir ${RESDIR} \
    --n_sample 2048
