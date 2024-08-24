from utils.logger import logger, clean_false_log

from utils.args import args

from utils.utils import pformat_dict

import wandb

import os

"""
parse all the arguments, generate the logger, check gpus to be used and wandb
"""
logger.info("Running with parameters: " + pformat_dict(args, indent=1))

# this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
if args.gpus is not None:
    logger.debug('Using only these GPUs: {}'.format(args.gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

# wanbd logging configuration
if args.wandb_name is not None:
    wandb.init(group=args.wandb_name, dir=args.wandb_dir)
    wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]

clean_false_log()