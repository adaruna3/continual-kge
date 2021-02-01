import numpy as np
from copy import copy
import torch
from os.path import basename
import __main__  # used to get the original execute module

from models import model_utils
from logger.terminal_utils import ExperimentArgParse, logout, log_train, log_test, InteractiveTerminal
from logger.viz_utils import ProcessorViz

import pdb


def setup_experiment(args):
    # init batch processors for training and validation
    train_args = copy(args)
    train_args.set_name = "train2id"
    tr_bp = model_utils.TrainBatchProcessor(train_args)
    dev_args = copy(args)
    dev_args.set_name = "valid2id"
    dev_args.neg_ratio = 0
    dev_args.dataset_fps = None
    de_bp = model_utils.DevBatchProcessor(dev_args)

    # generate training visualization logging
    viz_args = copy(args)
    viz_args.tag = basename(__main__.__file__).split(".")[0]
    viz = ProcessorViz(viz_args)

    # initializes a single model and optimizer used by all batch processors
    model_optim_args = copy(args)
    model_optim_args.num_ents = len(tr_bp.dataset.e2i)
    model_optim_args.num_rels = len(tr_bp.dataset.r2i)
    model = model_utils.init_model(model_optim_args)
    model.to(model_optim_args.device, non_blocking=True)
    optimizer = model_utils.init_optimizer(model_optim_args, model)

    tracker_args = copy(args)
    tracker_args.tag = basename(__main__.__file__).split(".")[0]
    tracker_args.sess = str(0)
    tracker = model_utils.EarlyStopTracker(tracker_args)

    return tr_bp, de_bp, viz, model, optimizer, tracker


def setup_test_session(sess, args, model):
    """
    performs pre-testing session operation to load the model
    """
    # loads best model for session
    load_args = copy(args)
    load_args.tag = basename(__main__.__file__).split(".")[0]
    load_args.sess = str(sess)
    model = model_utils.load_model(load_args, model)

    return model


if __name__ == "__main__":
    exp_parser = ExperimentArgParse("Standard setting experiment")
    exp_args = exp_parser.parse()

    # select hardware to use
    if exp_args.cuda and torch.cuda.is_available():
        logout("Running with CUDA")
        exp_args.device = torch.device('cuda')
    else:
        logout("Running with CPU, experiments will be slow", "w")
        exp_args.device = torch.device('cpu')

    if exp_args.sess_mode == "TRAIN":
        exp_tr_bp, exp_de_bp, exp_viz, exp_model, exp_optim, exp_tracker = setup_experiment(exp_args)

        while exp_tracker.continue_training():
            # validate
            if exp_tracker.validate():
                inf_metrics = np.asarray([exp_de_bp.process_epoch(exp_model)])
                # log inference metrics
                exp_viz.add_de_sample(inf_metrics)
                log_label = "i" if exp_tracker.get_epoch() == 0 else "s"
                log_train(inf_metrics, exp_tracker.get_epoch(),
                          0, exp_args.num_sess, log_label,
                          None, None,
                          exp_viz.log_fp, exp_args.log_num)
                # update tracker for early stopping & model saving
                exp_tracker.update_best(0, inf_metrics, exp_model)
            
            # train
            exp_viz.add_tr_sample(0, exp_tr_bp.process_epoch(exp_model, exp_optim))
            exp_tracker.step_epoch()

        # logs the final performance for session (i.e. best)
        best_performance, best_epoch = exp_tracker.get_best()
        log_train(best_performance, best_epoch, 0,
                  exp_args.num_sess, "f", None, None,
                  exp_viz.log_fp, exp_args.log_num)

    elif exp_args.sess_mode == "TEST":
        logout("Testing running...", "i")
        exp_tr_bp, exp_de_bp, exp_viz, exp_model, exp_optim, exp_tracker = setup_experiment(exp_args)

        exp_model = setup_test_session(0, exp_args, exp_model)
        inf_metrics = np.asarray([exp_de_bp.process_epoch(exp_model)])
        log_train(inf_metrics, 0, 0,
                  exp_args.num_sess, "f", None, None,
                  exp_viz.log_fp, exp_args.log_num)

    else:
        logout("Mode not recognized for this setting.", "f")
