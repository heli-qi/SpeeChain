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
    --exp_cfg EXP_CFG \\                                    # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/asr/libritts/train-960/exp_cfg
    [--data_cfg DATA_CFG] \\                                # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/asr/libritts/train-960/data_cfg (default: none)
    [--train_cfg TRAIN_CFG] \\                              # The name of your specified configuration file in ${SPEECHAIN_ROOT}/recipes/asr/libritts/train-960/train_cfg (default: none)
    [--infer_cfg INFER_CFG] \\                              # The name of your specified configuration file in ${SPEECHAIN_ROOT}/config/asr/ (default: none)
    [--num_workers NUM_WORKERS] \\                          # The value of 'num_workers' given to runner.py (default: 1)
    [--accum_grad ACCUM_GRAD] \\                            # The value of 'accum_grad' given to runner.py (default: 1)
    [--ngpu NGPU] \\                                        # The value of 'ngpu' given to runner.py (default: 1)
    [--gpus GPUS] \\                                        # The value of 'gpus' given to runner.py (default: none)
    --train false or true \\                                # Whether to activate training mode (default: false)
    --test false or true                                   # Whether to activate testing mode (default: false)" >&2
  exit 1
}

# --- Absolute Path References --- #
recipe_run_root=${SPEECHAIN_ROOT}/recipes/run.sh


# --- Arguments --- #
dry_run=false
no_optim=false

resume=false
train=false
test=false
train_result_path=
test_result_path=

exp_cfg=
data_cfg=
train_cfg=
infer_cfg=

num_workers=
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
        num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          num_workers=${val}
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


# --- 1. Argument Initialization --- #
#
args="
  --task asr \
  --dataset libritts \
  --subset train-960 \
  --dry_run ${dry_run} \
  --no_optim ${no_optim} \
  --resume ${resume} \
  --train ${train} \
  --test ${test}"

#
if [ -z ${exp_cfg} ];then
   echo "Please give an experimental configuration by '--exp_cfg'!"
   exit 1
else
  args="${args} --exp_cfg ${exp_cfg}"
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
if [ -n "${train_cfg}" ];then
  args="${args} --train_cfg ${train_cfg}"
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
fi


# --- 2. Execute the Job --- #
# ${args} should not be surrounded by double-quote
# shellcheck disable=SC2086
bash "${recipe_run_root}" ${args}
if $? != 0;then
  # ${args} should not be surrounded by double-quote
  # shellcheck disable=SC2086
  # shellcheck disable=SC1090
  source "${recipe_run_root}" ${args}
fi