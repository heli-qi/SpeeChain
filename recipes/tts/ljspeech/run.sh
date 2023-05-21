### Author: Heli Qi
### Affiliation: NAIST
### Date: 2022.12


# --- Global References --- #
if [ -z "${SPEECHAIN_ROOT}" ] || [ -z "${SPEECHAIN_PYTHON}" ];then
  echo "Cannot find environmental variables SPEECHAIN_ROOT and SPEECHAIN_PYTHON.
  Please move to the root path of the toolkit and run envir_preparation.sh there!"
  exit 1
fi
recipe_run_root=${SPEECHAIN_ROOT}/recipes/run.sh

# please manually change the following arguments everytime you dump a new dataset
task=tts
dataset=ljspeech
subset=


function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given)
    [--dry_run false or true] \\                            # Whether to activate dry-running mode (default: false)
    [--no_optim false or true] \\                           # Whether to activate no-optimization mode (default: false)
    [--resume false or true] \\                             # Whether to activate resuming mode (default: false)
    [--train_result_path TRAIN_RESULT_PATH] \\              # The value of train_result_path given to runner.py (default: none)
    [--test_result_path TEST_RESULT_PATH] \\                # The value of train_result_path given to runner.py (default: none)
    [--test_model TEST_MODEL] \\                            # The value of test_model given to runner.py (default: none)
    --exp_cfg EXP_CFG \\                                    # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/${task}/${dataset}/${subset}/exp_cfg
    [--data_cfg DATA_CFG] \\                                # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/${task}/${dataset}/${subset}/data_cfg (default: none)
    [--infer_cfg INFER_CFG] \\                              # The name of your specified configuration file in ${SPEECHAIN_ROOT}/config/${task}/ (default: none)
    [--train_num_workers TRAIN_NUM_WORKERS] \\              # The value of 'train_num_workers' given to runner.py (default: none)
    [--valid_num_workers VALID_NUM_WORKERS] \\              # The value of 'valid_num_workers' given to runner.py (default: none)
    [--test_num_workers TEST_NUM_WORKERS] \\                # The value of 'test_num_workers' given to runner.py (default: none)
    [--accum_grad ACCUM_GRAD] \\                            # The value of 'accum_grad' given to runner.py (default: none)
    [--ngpu NGPU] \\                                        # The value of 'ngpu' given to runner.py (default: none)
    [--gpus GPUS] \\                                        # The value of 'gpus' given to runner.py (default: none)
    --train TRAIN \\                                        # Whether to activate training mode (default: true)
    --test TEST                                            # Whether to activate testing mode (default: true)" >&2
  exit 1
}


# --- Arguments --- #
dry_run=false
no_optim=false

resume=false
train=true
test=true
train_result_path=
test_result_path=
test_model=

exp_cfg=
data_cfg=
infer_cfg=

train_num_workers=
valid_num_workers=
test_num_workers=
accum_grad=
ngpu=
gpus=


### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
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
        infer_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          infer_cfg=${val}
          ;;
        train_num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          train_num_workers=${val}
          ;;
        valid_num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          valid_num_workers=${val}
          ;;
        test_num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          test_num_workers=${val}
          ;;
        accum_grad)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          accum_grad=${val}
          ;;
        ngpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ngpu=${val}
          ;;
        gpus)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          gpus=${val}
          ;;
        help)
          print_help_message
          ;;
        *)
          echo "Unknown variable --$OPTARG"
          exit 1
          ;;
      esac
      ;;
    h)
      print_help_message
      ;;
    *)
      echo "Please refer to an argument by '--'."
      exit 1
      ;;
  esac
done


# --- 1. Argument Initialization --- #
#
args="
  --task ${task} \
  --dataset ${dataset} \
  --dry_run ${dry_run} \
  --no_optim ${no_optim} \
  --resume ${resume} \
  --train ${train} \
  --test ${test}"

#
if [ -n "${subset}" ];then
  args="${args} --subset ${subset}"
fi

#
if [ -z ${exp_cfg} ];then
   echo "Please give an experimental configuration by '--exp_cfg'!"
   exit 1
else
  args="${args} --exp_cfg ${exp_cfg}"
fi

#
if [ -n "${train_num_workers}" ];then
  args="${args} --train_num_workers ${train_num_workers}"
fi
#
if [ -n "${valid_num_workers}" ];then
  args="${args} --valid_num_workers ${valid_num_workers}"
fi
#
if [ -n "${test_num_workers}" ];then
  args="${args} --test_num_workers ${test_num_workers}"
fi
#
if [ -n "${accum_grad}" ];then
  args="${args} --accum_grad ${accum_grad}"
fi
#
if [ -n "${gpus}" ];then
  args="${args} --gpus ${gpus}"
fi
#
if [ -n "${ngpu}" ];then
  args="${args} --ngpu ${ngpu}"
fi

#
if [ -n "${data_cfg}" ];then
  args="${args} --data_cfg ${data_cfg}"
fi

#
if [ -n "${train_result_path}" ];then
  args="${args} --train_result_path ${train_result_path}"
fi

#
if ${test};then
  #
  if [ -n "${infer_cfg}" ];then
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
fi


# --- 2. Execute the Job --- #
# ${args} should not be surrounded by double-quote
# shellcheck disable=SC2086
bash "${recipe_run_root}" ${args}
if $? != 0;then
  echo "The command 'bash ""${recipe_run_root}"" ${args}' failed to be executed and returns $?!"
  exit $?
fi