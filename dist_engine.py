import argparse
from functools import partial
import traceback
import os
import signal
from typing import List

import torch
import torch.optim
import torch.multiprocessing as mp

from cv_lib.logger import MultiProcessLoggerListener
from cv_lib.utils import get_eval_logger, get_train_logger, get_cfg, to_json_str

from schema_inference.utils import DistLaunchArgs, LogArgs
from schema_inference.tasks.worker import backbone_worker
from schema_inference.tasks.worker_eval import eval_worker
from schema_inference.tasks.worker_schema_net import schema_net_worker


def train_logger(args):
    logger_constructor = partial(
        get_train_logger,
        logdir=args.log_dir,
        filename=args.file_name_cfg,
        mode="a" if os.path.isfile(args.resume) else "w"
    )
    return logger_constructor


def val_logger(args):
    logger_constructor = partial(
        get_eval_logger,
        logdir=args.log_dir
    )
    return logger_constructor


__REGISTERED_TASKS__ = {
    "backbone_worker": (backbone_worker, train_logger),
    "eval_worker": (eval_worker, val_logger),
    "schema_net_worker": (schema_net_worker, train_logger)
}

START_METHOD = "spawn"


def get_args():
    parser = argparse.ArgumentParser(description="Dist Engine")
    parser.add_argument("--num-nodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--master-url", default="tcp://localhost:9876", type=str)
    parser.add_argument("--backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--file-name-cfg", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--cfg-filepath", type=str)
    parser.add_argument("--worker", type=str, default="worker", choices=__REGISTERED_TASKS__)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--preserve-gpu", action="store_true")
    parser.add_argument("--preserve-percent", type=float, default=0.99)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def main():
    # get arguments
    args = get_args()
    global_cfg = get_cfg(args.cfg_filepath)

    ckpt_path = None
    if args.worker != "val":
        ckpt_path = os.path.join(args.log_dir, "ckpt")
        os.makedirs(ckpt_path, exist_ok=True)

    # get root logger constructure
    worker, logger_constructor = __REGISTERED_TASKS__[args.worker]
    # multi-process logger
    logger_listener = MultiProcessLoggerListener(logger_constructor(args), START_METHOD)
    logger = logger_listener.get_logger()

    process_pool: List[mp.Process] = list()

    def kill_handler(signum, frame):
        logger.warning("Got kill signal %d, frame:\n%s\nExiting...", signum, frame)
        for process in process_pool:
            try:
                logger.info("Killing subprocess: %d-%s...", process.pid, process.name)
                process.kill()
            except:
                pass
        logger.info("Stopping multiprocess logger...")
        logger_listener.stop()
        exit(1)

    logger.info("Registering kill handler")
    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGHUP, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)
    logger.info("Registered kill handler")

    # launch arguments
    ngpus_per_node = torch.cuda.device_count()
    world_size = args.num_nodes
    if args.multiprocessing:
        world_size = ngpus_per_node * args.num_nodes
    distributed = world_size > 1
    launch_args = {
        "ngpus_per_node": ngpus_per_node,
        "world_size": world_size,
        "distributed": distributed,
        "multiprocessing": args.multiprocessing,
        "rank": args.rank,
        "seed": args.seed,
        "backend": args.backend,
        "master_url": args.master_url,
        "preserve_gpu": args.preserve_gpu,
        "preserve_percent": args.preserve_percent,
        "use_amp": args.use_amp,
        "debug": args.debug
    }
    logger.info("Starting distributed runner with arguments:\n%s", to_json_str(launch_args))
    launch_args = DistLaunchArgs(**launch_args)
    log_args = LogArgs(logger_listener.queue, args.log_dir, args.file_name_cfg, ckpt_path)

    try:
        if distributed:
            logger.info("Start from multiprocessing")
            process_context = mp.spawn(
                worker,
                nprocs=ngpus_per_node,
                join=False,
                start_method=START_METHOD,
                args=(launch_args, log_args, global_cfg, args.resume)
            )
            process_pool = process_context.processes

            # Loop on join until it returns True or raises an exception.
            while not process_context.join():
                pass
        else:
            logger.info("Start from direct call")
            worker(0, launch_args, log_args, global_cfg, args.resume)
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical("While running, exception:\n%s\nTraceback:\n%s", str(e), str(tb))
    finally:
        # make sure listener is stopped
        logger_listener.stop()


if __name__ == "__main__":
    main()
