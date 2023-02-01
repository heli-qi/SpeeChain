### Author: Heli Qi
### Affiliation: NAIST
### Date: 2022.12

if [ -z "${SPEECHAIN_ROOT}" ] || [ -z "${SPEECHAIN_PYTHON}" ];then
  echo "Cannot find environmental variables SPEECHAIN_ROOT and SPEECHAIN_PYTHON.
  Please move to the root path of the toolkit and run envir_preparation.sh there!"
  exit 1
fi

function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given)
    [--dry_run false or true] \\                            # Whether to activate dry-running mode (default: false)
    [--no_optim false or true] \\                           # Whether to activate no-optimization mode (default: false)
    [--resume false or true] \\                             # Whether to activate resuming mode (default: false)
    [--train_result_path TRAIN_RESULT_PATH] \\              # The value of train_result_path given to runner.py (default: none)
    [--test_result_path TEST_RESULT_PATH] \\                # The value of train_result_path given to runner.py (default: none)
    [--test_model TEST_MODEL] \\                            # The value of test_model given to runner.py (default: none)
    --exp_cfg EXP_CFG \\                                    # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/{task}/{dataset}/{subset}/exp_cfg
    [--data_cfg DATA_CFG] \\                                # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/{task}/{dataset}/{subset}/data_cfg (default: none)
    [--train_cfg TRAIN_CFG] \\                              # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/{task}/{dataset}/{subset}/train_cfg (default: none)
    [--infer_cfg INFER_CFG] \\                              # The name of your specified configuration file in ${SPEECHAIN_ROOT}/config/{task}/ (default: none)
    [--ngpu NGPU] \\                                        # The value of 'ngpu' given to runner.py (default: none)
    [--gpus GPUS] \\                                        # The value of 'gpus' given to runner.py (default: none)
    [--num_workers NUM_WORKERS] \\                          # The value of 'num_workers' given to runner.py (default: none)
    [--accum_grad ACCUM_GRAD] \\                            # The value of 'accum_grad' given to runner.py (default: none)
    --task TASK \\                                          # The name of the task folder you want to run in ${SPEECHAIN_ROOT}/recipes/
    --dataset DATASET \\                                    # The name of the dataset folder you want to run in ${SPEECHAIN_ROOT}/recipes/{task}
    [--subset SUBSET] \\                                    # The name of the subset folder you want to run in ${SPEECHAIN_ROOT}/recipes/{task}/{subset} (default: none)
    --train false or true \\                                # Whether to activate training mode (default: false)
    --test false or true                                   # Whether to activate testing mode (default: false)" >&2
  exit 1
}

# --- Absolute Path References --- #
recipe_root=${SPEECHAIN_ROOT}/recipes
infer_root=${SPEECHAIN_ROOT}/config/infer
runner_path="${SPEECHAIN_ROOT}"/speechain/runner.py


# --- Arguments --- #
task=
dataset=
subset=

# For the first time you run a job using a new dataset, your job may suffer from long data loading time. It's probably
# because your target data haven't been accessed since your machine is turned on, hence it's difficult for your machine
# to read them into memory. If that happens, you could use the argument '--dry_run' to only perform the data loading for
# one epoch with '--num_epochs 1'. This will help your machine better access your target dataset.
#
# If the data loading speed of your job is still very slow even after the pre-heating, it's probably because your machine
# doesn't have enough memory for your target dataset. The lack of memory could be caused by either the limited equippment
# of your machine or occupation by the jobs of other members in your team.
dry_run=false
no_optim=false

# The training can be resumed from an existing checkpoint by giving the argument '--resume true'.
# Note that if you give a new data_cfg explicitly by '--data_cfg', the new data loading configuration will be adopted
# for resuming the model training. Otherwise, the existing data loading configuration in the exp folder will be used.
# The testing stage can also be resumed in our toolkit. The configuration is the same with training resuming.
# But one thing should be noted is that you must use the identical data loading configuration for resuming.
# It means that you should not give a new configuration by '--data_cfg'.
resume=false
train=false
test=false
# If '--train_result_path' is not given, the experimental files will be automatically saved to /exp under the same
# directory of your given '--config'.
train_result_path=
test_result_path=
test_model=

exp_cfg=
data_cfg=
train_cfg=
infer_cfg=

# The GPUs can be specified by either 'CUDA_VISIBLE_DEVICES' or '--gpus'. They are identical to the backbone.
ngpu=
gpus=
num_workers=
accum_grad=


### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        task)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          task=${val}
          ;;
        dataset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          dataset=${val}
          ;;
        subset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          subset=${val}
          ;;
        dry_run)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          dry_run=${val}
          ;;
        no_optim)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          no_optim=${val}
          ;;
        resume)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          resume=${val}
          ;;
        train)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          train=${val}
          ;;
        test)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          test=${val}
          ;;
        train_result_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          train_result_path=${val}
          ;;
        test_result_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          test_result_path=${val}
          ;;
        test_model)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          test_model=${val}
          ;;
        exp_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          exp_cfg=${val}
          ;;
        data_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          data_cfg=${val}
          ;;
        train_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          train_cfg=${val}
          ;;
        infer_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          infer_cfg=${val}
          ;;
        ngpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ngpu=${val}
          ;;
        gpus)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          gpus=${val}
          ;;
        num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          num_workers=${val}
          ;;
        accum_grad)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          accum_grad=${val}
          ;;
        help)
          print_help_message
          ;;
        ?)
          echo "Unknown variable $OPTARG"
          ;;
      esac
      ;;
    h)
      print_help_message
      ;;
    *)
      echo "Please refer to an argument by '--'."
      ;;
  esac
done


# --- 0. Argument Checking --- #
if [ -z ${task} ] || [ -z ${dataset} ];then
   echo "Please enter the name of your target task by '--task' and target dataset by '--dataset'!"
   exit 1
fi

if [ -z ${exp_cfg} ];then
   echo "Please give an experimental configuration by '--exp_cfg'!"
   exit 1
fi

if ${dry_run} && ${no_optim};then
  echo "'--dry_run' and '--no_optim' can not be true at the same time! Please specify one of them to true."
  exit 1
fi

if ${train} && ${test};then
  echo "'--train' and '--test' can not be true at the same time! Please specify one of them to true."
  exit 1
fi

if ! ${train} && ! ${test};then
  echo "'--train' and '--test' can not be false at the same time! Please specify one of them to true."
  exit 1
fi


# --- 1. Path Initialization --- #
infer_root=${infer_root}/${task}
task_root=${recipe_root}/${task}
dataset_root=${task_root}/${dataset}

if [ -n "${subset}" ];then
  subset_root=${dataset_root}/${subset}
else
  subset_root=${dataset_root}
fi


# --- 2. Argument Initialization --- #
args=""
#
if [ -n "${gpus}" ];then
  args="${args} --gpus ${gpus}"
fi
#
if [ -n "${ngpu}" ];then
  args="${args} --ngpu ${ngpu}"
fi
#
if [ -n "${num_workers}" ];then
  args="${args} --num_workers ${num_workers}"
fi
#
if [ -n "${accum_grad}" ];then
  args="${args} --accum_grad ${accum_grad}"
fi

#
if ${dry_run};then
  args="${args} --dry_run True --num_epochs 1"
fi
#
if ${no_optim};then
  args="${args} --no_optim True --num_epochs 1"
fi

#
if [ -n "${exp_cfg}" ];then
  # attach .yaml suffix if needed
  if [[ "${exp_cfg}" != *".yaml" ]];then
    exp_cfg="${exp_cfg}.yaml"
  fi
  # convert the relative path in ${subset_root}/exp_cfg if no slash inside
  if ! grep -q '/' <<< "${exp_cfg}";then
    exp_cfg="${subset_root}/exp_cfg/${exp_cfg}"
  fi
  args="${args} --config ${exp_cfg}"
fi
#
if [ -n "${data_cfg}" ];then
  # attach .yaml suffix if needed
  if [[ "${data_cfg}" != *".yaml" ]];then
    data_cfg="${data_cfg}.yaml"
  fi
  # convert the relative path in ${subset_root}/data_cfg if no slash inside
  if ! grep -q '/' <<< "${data_cfg}";then
    data_cfg="${subset_root}/data_cfg/${data_cfg}"
  fi
  args="${args} --data_cfg ${data_cfg}"
fi
#
if [ -n "${train_cfg}" ];then
  # attach .yaml suffix if needed
  if [[ "${train_cfg}" != *".yaml" ]];then
    train_cfg="${train_cfg}.yaml"
  fi
  # convert the relative path in ${subset_root}/train_cfg if no slash inside
  if ! grep -q '/' <<< "${train_cfg}";then
    train_cfg="${subset_root}/train_cfg/${train_cfg}"
  fi
  args="${args} --train_cfg ${train_cfg}"
fi

#
if ${resume};then
  args="${args} --resume True"
else
  args="${args} --resume False"
fi

#
if ${train};then
  args="${args} --train True"
else
  args="${args} --train False"
fi
#
if [ -n "${train_result_path}" ];then
  args="${args} --train_result_path ${train_result_path}"
fi

#
if ${test};then
  args="${args} --test True"
  #
  if [ -n "${infer_cfg}" ];then
    # do sth when infer_cfg is the name of a configuration file
    if ! grep -q ':' <<< "${infer_cfg}";then
      # attach .yaml suffix if needed
      if [[ "${infer_cfg}" != *".yaml" ]];then
        infer_cfg="${infer_cfg}.yaml"
      fi
      # convert the relative path in ${infer_root}/${task} if no slash inside
      if ! grep -q '/' <<< "${infer_cfg}";then
        if [ ${task} == 'offline_tts2asr' ];then
          folder='asr'
        elif [ ${task} == 'offline_asr2tts' ]; then
          folder='asr'
        else
          folder=${task}
        fi
        infer_cfg="${infer_root}/${folder}/${infer_cfg}"
      fi
    fi
    args="${args} --infer_cfg ${infer_cfg}"
  fi
  #
  if [ -n "${test_result_path}" ];then
    args="${args} --test_result_path ${test_result_path}"
  fi
  #
  if [ -n "${test_model}" ];then
    args="${args} --test_model ${test_model}"
  fi
else
  args="${args} --test False"
fi


# --- 3. Execute the Job --- #
# ${args} should not be surrounded by double-quote
# shellcheck disable=SC2086
${SPEECHAIN_PYTHON} "${runner_path}" ${args}
