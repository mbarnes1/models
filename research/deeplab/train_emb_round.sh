#!/bin/bash

# Set defaults
TRAIN_BATCH_SIZE=2  # batch size
NUM_CLONES=1  # number of GPUs
TRAINING_NUMBER_OF_STEPS=100000
TRAIN_SPLIT="train"
MODEL_VARIANT="xception_65"
OUTPUT_STRIDE=16
DECODER_OUTPUT_STRIDE=4
DATASET="cityscapes"
BASE_LEARNING_RATE=0.01
FINE_TUNE_BATCH_NORM=False
ADAM=False
NUM_READERS=4
NUM_THREADS=8
SPECTRAL=True
SEMANTIC_BLOCKING=False
PACKING_RADIUS=1.0
FAST_GRAD=False
IGNORE_LABEL=True
EMBEDDING_DIMENSION=0
LOCATION=None  # either input or xception
BASE_ATROUS_RATE=6  # if X, use --atrous_rates=X --atrous_rates=2X --atrous_rates=3X
SCRATCH_PROJECT=/pylon5/ir5fp7p/mbarnes
HOME_PROJECT=$HOME
TF_INITIAL_CHECKPOINT=${SCRATCH_PROJECT}"/data/models/tensorflow/deeplab_imagenet/model.ckpt"
VAL_GT_DIR=${SCRATCH_PROJECT}"/data/datasets/cityscapes/gtFine/val/"
VAL_SEMANTIC_DIR=${SCRATCH_PROJECT}"/exp/cityscapes/train_on_train_set/vis/google/raw_segmentation_results/"
DATASET_DIR=${SCRATCH_PROJECT}"/data/datasets/cityscapes/tfrecord_instances/"


NCPUS_ROUND=5
NCPU=16
GPUTYPE='k80'  # k80 or p100
TIME=48:00:00


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --train_batch_size)
        TRAIN_BATCH_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
    --num_clones)
        NUM_CLONES="$2"
        shift # past argument
        shift # past value
        ;;
    --training_number_of_steps)
        TRAINING_NUMBER_OF_STEPS="$2"
        shift # past argument
        shift # past value
        ;;
    --base_learning_rate)
        BASE_LEARNING_RATE="$2"
        shift # past argument
        shift # past value
        ;;
    --adam)
        ADAM="$2"
        shift # past argument
        shift # past value
        ;;
    --fine_tune_batch_norm)
        FINE_TUNE_BATCH_NORM="$2"
        shift # past argument
        shift # past value
        ;;
    --exp_name)
        EXP_NAME="$2"
        shift # past argument
        shift # past value
        ;;
    --num_readers)
        NUM_READERS="$2"
        shift # past argument
        shift # past value
        ;;
    --num_threads)
        NUM_THREADS="$2"
        shift # past argument
        shift # past value
        ;;
    --spectral)
        SPECTRAL="$2"
        shift # past argument
        shift # past value
        ;;
    --semantic_blocking)
        SEMANTIC_BLOCKING="$2"
        shift # past argument
        shift # past value
        ;;
    --packing_radius)
        PACKING_RADIUS="$2"
        shift # past argument
        shift # past value
        ;;
    --fast_grad)
        FAST_GRAD="$2"
        shift # past argument
        shift # past value
        ;;
    --ignore_label)
        IGNORE_LABEL="$2"
        shift # past argument
        shift # past value
        ;;
    --embedding_dimension)
        EMBEDDING_DIMENSION="$2"
        shift # past argument
        shift # past value
        ;;
    --location)
        LOCATION="$2"
        shift # past argument
        shift # past value
        ;;
    --base_atrous_rate)
        BASE_ATROUS_RATE="$2"
        shift # past argument
        shift # past value
        ;;
    --tf_initial_checkpoint)
        TF_INITIAL_CHECKPOINT="$2"
        shift # past argument
        shift # past value
        ;;
    --ncpu)
        NCPU="$2"
        shift # past argument
        shift # past value
        ;;
    --gputype)
        GPUTYPE="$2"
        shift # past argument
        shift # past value
        ;;
    --time)
        TIME="$2"
        shift # past argument
        shift # past value
        ;;
    *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

## Create output directories  ##
if [ "$GPUTYPE" == "k80" ]
then
  GPUS_AVAILABLE_PER_NODE=4
elif [ "$GPUTYPE" == "p100" ]
then
  GPUS_AVAILABLE_PER_NODE=2
else
  echo "Invalid GPU type"
  exit 1
fi
NODES=$(((NUM_CLONES+GPUS_AVAILABLE_PER_NODE-1)/GPUS_AVAILABLE_PER_NODE))
ATROUS1=${BASE_ATROUS_RATE}
ATROUS2=$((2*BASE_ATROUS_RATE))
ATROUS3=$((3*BASE_ATROUS_RATE))


DATESTAMP=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%H%M%S")
DATETIME=${DATESTAMP}"_"${TIMESTAMP}
EXP_NAME=${EXP_NAME}"_"${DATETIME}
BASE_DIR=${SCRATCH_PROJECT}"/exp/cityscapes/train_on_train_set"
TRAIN_LOGDIR=${BASE_DIR}"/train/"${USER}"/"${EXP_NAME}
EMB_LOGDIR=${BASE_DIR}"/emb/"${USER}"/"${EXP_NAME}
ROUND_LOGDIR=${BASE_DIR}"/round/"${USER}"/"${EXP_NAME}
ROUND_TENSORBOARDDIR=${BASE_DIR}"/round_tensorboard/"${USER}"/"${EXP_NAME}
mkdir ${TRAIN_LOGDIR}
mkdir ${EMB_LOGDIR}
mkdir ${ROUND_LOGDIR}
mkdir ${EMB_LOGDIR}"/raw_segmentation_results"
mkdir ${ROUND_TENSORBOARDDIR}

echo "Distributed training:
- Train batch size                      ${TRAIN_BATCH_SIZE}
- GPU(s)                                ${NUM_CLONES} $GPUTYPE
- Node(s)                               ${NODES}
- GPUs available per node type          ${GPUS_AVAILABLE_PER_NODE}
- CPU(s) per node (i.e. per 2 GPUs)     ${NCPU}
- Training number of steps              ${TRAINING_NUMBER_OF_STEPS}
- Base learning rate                    ${BASE_LEARNING_RATE}
- Adam Optimizer                        ${ADAM}
- Fine tune batch norm                  ${FINE_TUNE_BATCH_NORM}
- Atrous rate 1                         ${ATROUS1}
- Atrous rate 2                         ${ATROUS2}
- Atrous rate 3                         ${ATROUS3}
- Output stride                         ${OUTPUT_STRIDE}
- Reader(s)                             ${NUM_READERS}
- Thread(s)                             ${NUM_THREADS}
- Spectral loss                         ${SPECTRAL}
- Semantic blocking                     ${SEMANTIC_BLOCKING}
- Packing Radius                        ${PACKING_RADIUS}
- Fast grad                             ${FAST_GRAD}
- Ignore the ignore label               ${IGNORE_LABEL}
- Embedding dimension                   ${EMBEDDING_DIMENSION}
- Location                              ${LOCATION}
- Max time                              ${TIME}
- Initial checkpoint                    ${TF_INITIAL_CHECKPOINT}
- Train log directory                   ${TRAIN_LOGDIR}
- Embedding log directory               ${EMB_LOGDIR}
- Rounding log directory                ${ROUND_LOGDIR}
- Rounding tensorboard directory        ${ROUND_TENSORBOARDDIR}
- Ground truth instance directory       ${VAL_GT_DIR}
- Semantic labels directory             ${VAL_SEMANTIC_DIR}"

## Training ##

# Write training shell file
echo "#!/bin/bash
#SBATCH --nodes=${NODES}
#SBATCH --partition=GPU-shared
#SBATCH --ntasks-per-node ${NCPU}
#SBATCH --gres=gpu:${GPUTYPE}:${NUM_CLONES}
#SBATCH -t ${TIME}

echo \"Using ${NUM_CLONES} GPU's on ${NODES} nodes for training.\"

set -x  # echo commands to stdout
#set -u  # throw an error if unset variable referenced
set -e  # exit on error
export PGI_ACC_TIME=1  # profiling on

# Setup environment
module purge
module load tensorflow/1.5_gpu
source activate
#module load python2/2.7.14_gcc5_np1.13    cuda/9.0   mpi/gcc_openmpi
#export LD_LIBRARY_PATH=/opt/packages/mkldnn/external/mklml_lnx_2018.0.1.20171227/lib
#source  /opt/packages/TensorFlow/gnu_openmpi/tf1.5_gpu/bin/activate

cd ${HOME_PROJECT}/deep_spectral_clustering/models/research/
export PYTHONPATH=\$PYTHONPATH:${HOME_PROJECT}/deep_spectral_clustering/models/research/
export PYTHONPATH=\$PYTHONPATH:${HOME_PROJECT}/deep_spectral_clustering/models/research/slim/

# Run job
python deeplab/train.py --train_batch_size=${TRAIN_BATCH_SIZE} --num_clones=${NUM_CLONES} --logtostdout=True --training_number_of_steps=${TRAINING_NUMBER_OF_STEPS} --train_split=\"${TRAIN_SPLIT}\" --model_variant=\"${MODEL_VARIANT}\" --atrous_rates=${ATROUS1} --atrous_rates=${ATROUS2} --atrous_rates=${ATROUS3} --output_stride=${OUTPUT_STRIDE} --decoder_output_stride=${DECODER_OUTPUT_STRIDE} --train_crop_size=769 --train_crop_size=769 --tf_initial_checkpoint=${TF_INITIAL_CHECKPOINT} --train_logdir=${TRAIN_LOGDIR} --dataset_dir=${DATASET_DIR} --fine_tune_batch_norm=${FINE_TUNE_BATCH_NORM} --dataset=\"${DATASET}\" --num_readers=${NUM_READERS} --num_threads=${NUM_THREADS} --spectral=${SPECTRAL} --semantic_blocking=${SEMANTIC_BLOCKING} --packing_radius=${PACKING_RADIUS} --base_learning_rate=${BASE_LEARNING_RATE} --fast_grad=${FAST_GRAD} --ignore_label=${IGNORE_LABEL} --timestamp=False --embedding_dimension=${EMBEDDING_DIMENSION} --location=${LOCATION} --adam=${ADAM}
" > ${TRAIN_LOGDIR}/train.sh

## Evaluation ##

# Write evaluation shell file
echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --ntasks-per-node 7
#SBATCH --gres=gpu:k80:1
#SBATCH -t ${TIME}

echo \"Using 1 k80 GPU on 1 node for evaluation.\"

set -x  # echo commands to stdout
#set -u  # throw an error if unset variable referenced
set -e  # exit on error
export PGI_ACC_TIME=1  # profiling on

# Setup environment
module purge
module load tensorflow/1.5_gpu
source activate

cd $HOME_PROJECT/deep_spectral_clustering/models/research/
export PYTHONPATH=\$PYTHONPATH:${HOME_PROJECT}/deep_spectral_clustering/models/research/
export PYTHONPATH=\$PYTHONPATH:${HOME_PROJECT}/deep_spectral_clustering/models/research/slim/

# Run job
python deeplab/vis.py  --logtostderr=${LOGTOSTDERR} --vis_split=\"val\" --model_variant=\"${MODEL_VARIANT}\" --atrous_rates=${ATROUS1} --atrous_rates=${ATROUS2} --atrous_rates=${ATROUS3} --output_stride=${OUTPUT_STRIDE} --decoder_output_stride=${DECODER_OUTPUT_STRIDE} --vis_crop_size=1025 --vis_crop_size=2049 --dataset=\"${DATASET}\" --colormap_type=\"cityscapes\" --checkpoint_dir=${TRAIN_LOGDIR} --vis_logdir=${EMB_LOGDIR} --dataset_dir=${DATASET_DIR} --save_raw_logits=True --instance=True --max_number_of_iterations=0 --keep_all_raw_logits=True --embedding_dimension=${EMBEDDING_DIMENSION} --final_train_iteration=${TRAINING_NUMBER_OF_STEPS} --location=${LOCATION}
" > ${EMB_LOGDIR}/emb.sh

## Rounding ##

# Write rounding shell file
echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=RM-shared
#SBATCH --ntasks-per-node ${NCPUS_ROUND}
#SBATCH -t ${TIME}

echo \"Launching rounding job.\"

set -x  # echo commands to stdout
#set -u  # throw an error if unset variable referenced
set -e  # exit on error
export PGI_ACC_TIME=1  # profiling on

# Setup environment
module purge
module load anaconda2/5.1.0  # this environment is missing tensorboardX
source activate $ANACONDA_HOME
pip install --user tensorboardX

cd $HOME_PROJECT/deep_spectral_clustering/models/research/deeplab/

# Run job
python round_embeddings.py ${VAL_GT_DIR} ${VAL_SEMANTIC_DIR} ${EMB_LOGDIR}/raw_segmentation_results ${ROUND_LOGDIR} ${ROUND_TENSORBOARDDIR} --num_processes=${NCPUS_ROUND} --mean_shift_iterations=3 --packing_radius=${PACKING_RADIUS} --evaluate_interval=1000 --max_number_of_iterations=0 --final_train_iteration=${TRAINING_NUMBER_OF_STEPS}
" > ${ROUND_LOGDIR}/round.sh

# Queue all jobs. Hold the training job, manually release it when resources for training and embedding are available.
# Release the job with "scontrol release jid_train"
# Use --nice with train job, to encourage that training and embedding jobs will run together.
jid_train=$(sbatch --nice=1000 --parsable --output ${TRAIN_LOGDIR}"/slurm.out" --job-name="tr"${TIMESTAMP} ${TRAIN_LOGDIR}/train.sh)
jid_emb=$(sbatch --dependency=after:$jid_train --parsable --output ${EMB_LOGDIR}"/slurm.out" --job-name="em"${TIMESTAMP} ${EMB_LOGDIR}/emb.sh)
jid_round=$(sbatch --dependency=after:$jid_emb --parsable --output ${ROUND_LOGDIR}"/slurm.out" --job-name="ro"${TIMESTAMP} ${ROUND_LOGDIR}/round.sh)
echo "Queued jobs:
Train     ${jid_train}
Embed     ${jid_emb}
Round     ${jid_round}"
