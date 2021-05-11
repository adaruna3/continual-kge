import numpy as np
from copy import copy
import torch
import os
import __main__  # used to get the original execute module

from models import model_utils
from models.pytorch_modelsize import SizeEstimator
from logger.terminal_utils import ExperimentArgParse, logout, log_train, log_test, InteractiveTerminal
from logger.viz_utils import ProcessorViz, AbstractProcessorViz


def setup_experiment(args):
    """
    initializes the batch processors for training and validation,
    the model and optimizer, training tracker for early stopping,
    and tensorboard for training visualization used throughout
    the entire experiment
    """
    # initializes batch processors for training and validation
    tr_bps = []
    de_bps = []
    for session in range(args.num_sess):
        # create training batch processor for session
        train_args = copy(args)
        train_args.dataset += "_" + str(session)
        train_args.set_name = "train2id"
        train_args.session = session
        if train_args.cl_method == "PNN":
            tr_bps.append(model_utils.PNNTrainBatchProcessor(train_args))
        elif train_args.cl_method == "CWR":
            tr_bps.append(model_utils.CWRTrainBatchProcessor(train_args))
        elif train_args.cl_method == "SI":
            tr_bps.append(model_utils.SITrainBatchProcessor(train_args))
        elif train_args.cl_method == "L2":
            tr_bps.append(model_utils.L2TrainBatchProcessor(train_args))
        elif train_args.cl_method == "DGR":
            tr_bps.append(model_utils.DGRTrainBatchProcessor(train_args))
        elif train_args.cl_method == "offline":
            tr_bps.append(model_utils.TrainBatchProcessor(train_args))
        elif train_args.cl_method == "finetune":
            tr_bps.append(model_utils.TrainBatchProcessor(train_args))
        else:
            logout("Training batch processor for CL method '" + str(args.cl_method) + "' to be used is not implemented.", "f")
            exit()
        # create evaluation batch processor for session
        dev_args = copy(args)
        dev_args.dataset += "_" + str(session)
        if dev_args.sess_mode == "TEST":
            dev_args.set_name = "test2id"
        else:
            dev_args.set_name = "valid2id"
        dev_args.neg_ratio = 0
        if session:
            dev_args.dataset_fps = [tr_bp.dataset.fp for tr_bp in tr_bps]
        else:
            dev_args.dataset_fps = None
        de_bps.append(model_utils.DevBatchProcessor(dev_args))

    # initializes a single model and optimizer used across sessions
    model_optim_args = copy(args)
    model_optim_args.num_ents = len(tr_bps[0].dataset.e2i)
    model_optim_args.num_rels = len(tr_bps[0].dataset.r2i)
    model = model_utils.init_model(model_optim_args)
    optimizer = model_utils.init_optimizer(model_optim_args, model)

    # generate training visualization logging
    if args.sess_mode == "TRAIN":
        viz_args = copy(args)
        viz_args.tag = os.path.basename(__main__.__file__).split(".")[0]
        viz = ProcessorViz(viz_args)
    else:
        viz_args = copy(args)
        viz_args.tag = os.path.basename(__main__.__file__).split(".")[0]
        viz = AbstractProcessorViz(viz_args)

    return tr_bps, de_bps, viz, model, optimizer


def setup_train_session(sess, args, model, optim, tr_bps, viz):
    """
    performs pre-training session operations required by each
    conitnual learning algorithm
    """
    tr_bp = tr_bps[sess]
    prev_tr_bp = tr_bps[sess - 1]

    initial_num_train_triples = copy(tr_bp.dataset.triples.shape[0])
    num_generated_train_triples = 0

    if sess:
        # load best model from prior session
        load_args = copy(args)
        load_args.tag = os.path.basename(__main__.__file__).split(".")[0]
        load_args.sess = str(sess - 1)
        model = model_utils.load_model(load_args, model)

    if sess and args.cl_method == "PNN":
        # freezes prior weights and initializes new weights using Xavier
        model.freeze_embeddings(prev_tr_bp.dataset.known_ents,
                                prev_tr_bp.dataset.known_rels)
        optim = model_utils.init_optimizer(copy(args), model)
    elif sess and args.cl_method == "CWR":
        # copies tw weights to cw and re-inits
        model = prev_tr_bp.copyweights_tw_2_cw(model)
        model = tr_bp.reinit_tw(model)
        optim = model_utils.init_optimizer(copy(args), model)
    elif sess and args.cl_method == "SI":
        # sets weights to be regularized and updates importance values
        model.set_regularize_ents_rels(prev_tr_bp.dataset.known_ents,
                                       prev_tr_bp.dataset.known_rels)
        model.update_omega()
        model.initialize_W()
        num_new_ents = np.sum(np.isin(tr_bp.dataset.triple_ents,
                                      prev_tr_bp.dataset.known_ents,
                                      invert=True))
        regul_strength = 1.0 - (num_new_ents / float(len(tr_bp.dataset.known_ents)))
        model.set_task_weight(args.regul_scaling * regul_strength)
        optim = model_utils.init_optimizer(copy(args), model)
    elif sess and args.cl_method == "L2":
        # sets weights to be regularized
        model.set_regularize_ents_rels(prev_tr_bp.dataset.known_ents,
                                       prev_tr_bp.dataset.known_rels)
        model.update_og_params()
        num_new_ents = np.sum(np.isin(tr_bp.dataset.triple_ents,
                                      prev_tr_bp.dataset.known_ents,
                                      invert=True))
        regul_strength = 1.0 - (num_new_ents / float(len(tr_bp.dataset.known_ents)))
        model.set_task_weight(args.regul_scaling * regul_strength)
        optim = model_utils.init_optimizer(copy(args), model)
    elif args.cl_method == "DGR":
        triples = None
        if sess:
            # uses the prior generative model to extend the triple set
            prev_dgr_args = copy(args)
            prev_dgr_args.dataset += "_" + str(sess-1)
            prev_dgr_args.tag = os.path.basename(__main__.__file__).split(".")[0]
            prev_dgr_args.sess = sess-1
            gruvae = model_utils.GRUVAETrainBatchProcessor(prev_dgr_args)
            gruvae.load_model()
            num_samples = prev_tr_bp.dataset.triples.shape[0]
            samples, triples = gruvae.get_samples(num_samples, "samples")
            stats = gruvae.get_sample_stats(samples)
            num_generated_train_triples += triples.shape[0]
            logout(stats)
            tr_bp.dataset.triples = np.append(tr_bp.dataset.triples, triples, axis=0)
            tr_bp.dataset.triples = np.unique(tr_bp.dataset.triples, axis=0)
            tr_bp.dataset.load_bernouli_sampling_stats()
            tr_bp.dataset.load_current_ents_rels()
            tr_bp.reset_data_loader()
            del gruvae
        dgr_args = copy(args)
        dgr_args.dataset += "_" + str(sess)
        dgr_args.tag = os.path.basename(__main__.__file__).split(".")[0]
        dgr_args.sess = sess
        gruvae = model_utils.GRUVAETrainBatchProcessor(dgr_args)
        if triples is not None:
            gruvae.extend_dataset(triples)
        # trains the new generative model
        gruvae.reset_model()
        gruvae.reset_data_loader()
        gruvae.reset_triple_set()
        gruvae.train_model(viz)
        del gruvae
        # prepares discriminative model for training
        model.init_weights()
        optim = model_utils.init_optimizer(copy(args), model)
    elif sess and args.cl_method == "offline":
        # combines all prior training sets with current for "offline" training
        model.init_weights()
        optim = model_utils.init_optimizer(copy(args), model)
        if args.enable_offline:  # allows disabling of offline for robot experiments
            tr_bp.dataset.triples = np.unique(np.concatenate((prev_tr_bp.dataset.triples,
                                                              tr_bp.dataset.triples), axis=0), axis=0)
            tr_bp.dataset.load_bernouli_sampling_stats()
            tr_bp.dataset.load_current_ents_rels()
            tr_bp.reset_data_loader()
    elif sess and args.cl_method == "finetune":
        # just resets optimizer to training
        optim = model_utils.init_optimizer(copy(args), model)

    # generates early stop tracker for training
    tracker_args = copy(args)
    tracker_args.tag = os.path.basename(__main__.__file__).split(".")[0]
    tracker_args.sess = str(sess)
    tracker = model_utils.EarlyStopTracker(tracker_args)

    # calculates model and sample memory sizes for results
    final_num_train_triples = copy(tr_bp.dataset.triples.shape[0])
    stored_sample_size = min(0, final_num_train_triples - initial_num_train_triples - num_generated_train_triples)
    se = SizeEstimator(copy(args))
    model_params_size = se.estimate_size(model)[0]
    model_mem_stats = (stored_sample_size, model_params_size)
    del se
    logout("Mem stats:" + str(model_mem_stats))

    return model, optim, tr_bp, tracker, model_mem_stats


def setup_test_session(sess, args, model):
    """
    performs pre-testing session operation to load the model
    """
    # loads best model for session
    load_args = copy(args)
    load_args.tag = os.path.basename(__main__.__file__).split(".")[0]
    load_args.sess = str(sess)
    model = model_utils.load_model(load_args, model)

    return model


if __name__ == "__main__":
    exp_parser = ExperimentArgParse("Continual setting experiment")
    exp_args = exp_parser.parse()

    # selects hardware to use
    if exp_args.cuda and torch.cuda.is_available():
        logout("Running with CUDA")
        exp_args.device = torch.device('cuda')
    else:
        logout("Running with CPU, experiments will be slow", "w")
        exp_args.device = torch.device('cpu')

    if exp_args.sess_mode == "TRAIN":
        logout("Training running...", "i")
        exp_tr_bps, exp_de_bps, exp_viz, exp_model, exp_optim = setup_experiment(exp_args)

        for exp_sess in range(exp_args.num_sess):
            exp_model, exp_optim, exp_tr_bp, exp_tracker, model_stats = \
                setup_train_session(exp_sess, exp_args, exp_model,
                                       exp_optim, exp_tr_bps, exp_viz)

            while exp_tracker.continue_training():
                # validate
                if exp_tracker.validate():
                    inf_metrics = model_utils.evaluate_model(exp_args, exp_sess, exp_de_bps, exp_model)
                    # log inference metrics
                    exp_viz.add_de_sample(inf_metrics)
                    log_label = "i" if exp_tracker.get_epoch() == 0 else "s"
                    log_train(inf_metrics, exp_tracker.get_epoch(),
                              exp_sess, exp_args.num_sess, log_label,
                              model_stats[0], model_stats[1],
                              exp_viz.log_fp, exp_args.log_num)
                    # update tracker for early stopping & model saving
                    exp_tracker.update_best(exp_sess, inf_metrics, exp_model)

                # train
                exp_viz.add_tr_sample(exp_sess, exp_tr_bp.process_epoch(exp_model, exp_optim))
                exp_tracker.step_epoch()

            # logs the final performance for session (i.e. best)
            best_metrics, best_epoch = exp_tracker.get_best()
            log_train(best_metrics, best_epoch, exp_sess, exp_args.num_sess,
                      "f", model_stats[0], model_stats[1], exp_viz.log_fp,
                      exp_args.log_num)

    elif exp_args.sess_mode == "TEST":
        logout("Testing running...", "i")
        exp_tr_bps, exp_de_bps, exp_viz, exp_model, exp_optim = setup_experiment(exp_args)

        for exp_sess in range(exp_args.num_sess):
            exp_model = setup_test_session(exp_sess, exp_args, exp_model)
            inf_metrics = model_utils.evaluate_model(exp_args, exp_sess, exp_de_bps, exp_model)
            log_test(inf_metrics, 0, exp_sess, exp_args.num_sess, "t", 0, 0,
                     exp_viz.log_fp, exp_args.log_num)

    else:
        logout("Mode not recognized for this setting.", "f")
