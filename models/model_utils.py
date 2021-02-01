from os.path import abspath, dirname
import numpy as np
from copy import copy, deepcopy
# torch imports
import torch
from torch.utils.data import DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable
import torch.optim as optim

# user module imports
from logger.terminal_utils import logout, log_train
import datasets.data_utils as data_utils
from models.pytorch_modelsize import SizeEstimator
import models.standard_models as std_models
import models.l2_models as l2_models
import models.pnn_models as pnn_models
import models.cwr_models as cwr_models
import models.si_models as si_models
import models.dgr_models as dgr_models

import pdb
import time

#######################################################
#  Standard Processors (finetune/offline)
#######################################################
class TrainBatchProcessor:
    def __init__(self, cmd_args):
        self.args = copy(cmd_args)
        self.dataset = data_utils.TripleDataset(self.args.dataset, self.args.neg_ratio)
        self.dataset.load_triple_set(self.args.set_name)
        self.dataset.load_known_ent_set()
        self.dataset.load_known_rel_set()
        self.dataset.load_current_ents_rels()
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=self.args.batch_size,
                                      num_workers=self.args.num_workers,
                                      collate_fn=collate_batch,
                                      pin_memory=True)

    def reset_data_loader(self):
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=self.args.batch_size,
                                      num_workers=self.args.num_workers,
                                      collate_fn=collate_batch,
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model_was_training = model.training
        if not model_was_training:
            model.train()

        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.forward(bh.contiguous().to(self.args.device),
                                       br.contiguous().to(self.args.device),
                                       bt.contiguous().to(self.args.device),
                                       by.contiguous().to(self.args.device))
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


class DevBatchProcessor:
    def __init__(self, cmd_args):
        self.args = copy(cmd_args)
        self.dataset = data_utils.TripleDataset(self.args.dataset, self.args.neg_ratio)
        self.dataset.load_triple_set(self.args.set_name)
        self.dataset.load_mask(cmd_args.dataset_fps)
        self.dataset.load_known_ent_set()
        self.dataset.load_known_rel_set()
        self.batch_size = 10
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      num_workers=self.args.num_workers,
                                      collate_fn=collate_batch,
                                      pin_memory=True)
        self.cutoff = int(self.args.valid_cutoff / self.batch_size) if self.args.valid_cutoff is not None else None

    def process_epoch(self, model):
        model_was_training = model.training
        if model_was_training:
            model.eval()

        h_ranks = np.ndarray(shape=0, dtype=np.float64)
        t_ranks = np.ndarray(shape=0, dtype=np.float64)
        with no_grad():
            for idx_b, batch in enumerate(self.data_loader):
                if self.cutoff is not None:  # validate on less triples for large datasets
                    if idx_b > self.cutoff:
                        break

                if self.args.cuda and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # get ranks for each triple in the batch
                bh, br, bt, by = batch
                h_ranks = np.append(h_ranks, self._rank_head(model, bh, br, bt), axis=0)
                t_ranks = np.append(t_ranks, self._rank_tail(model, bh, br, bt), axis=0)

        # calculate hits & mrr
        hits10_h = np.count_nonzero(h_ranks <= 10) / len(h_ranks)
        hits10_t = np.count_nonzero(t_ranks <= 10) / len(t_ranks)
        hits10 = (hits10_h + hits10_t) / 2.0
        mrr = np.mean(np.concatenate((1 / h_ranks, 1 / t_ranks), axis=0))

        return hits10, mrr

    def _rank_head(self, model, h, r, t):
        rank_heads = Variable(from_numpy(np.arange(len(self.dataset.e2i)))).repeat(h.shape[0], 1)
        scores = model.predict(rank_heads.contiguous().to(self.args.device),
                               r.unsqueeze(-1).contiguous().to(self.args.device),
                               t.unsqueeze(-1).contiguous().to(self.args.device))
        ranks = []
        known_ents = np.asarray(self.dataset.known_ents, dtype=np.int64)
        for i in range(scores.shape[0]):
            scores_ = copy(scores[i, :])
            scores_ = np.stack((scores_, np.arange(len(self.dataset.e2i))), axis=-1)
            if (int(r[i].numpy()), int(t[i].numpy())) in self.dataset.h_mask:
                h_mask = copy(self.dataset.h_mask[(int(r[i].numpy()), int(t[i].numpy()))])
                h_mask.remove(int(h[i].numpy()))
                ents = known_ents[np.isin(known_ents, h_mask, True, True)]
            else:
                ents = known_ents
            filtered_scores = scores_[np.isin(scores_[:, -1], ents, True), :]
            filtered_ent_idx = int(np.where(filtered_scores[:, -1] == int(h[i].numpy()))[0])
            ranks_ = np.argsort(filtered_scores[:, 0], 0)
            ranks.append(int(np.where(ranks_ == filtered_ent_idx)[0])+1)
        return ranks

    def _rank_tail(self, model, h, r, t):
        rank_tails = Variable(from_numpy(np.arange(len(self.dataset.e2i)))).repeat(t.shape[0], 1)
        scores = model.predict(h.unsqueeze(-1).contiguous().to(self.args.device),
                               r.unsqueeze(-1).contiguous().to(self.args.device),
                               rank_tails.contiguous().to(self.args.device))
        ranks = []
        known_ents = np.asarray(self.dataset.known_ents, dtype=np.int64)
        for i in range(scores.shape[0]):
            scores_ = copy(scores[i, :])
            scores_ = np.stack((scores_, np.arange(len(self.dataset.e2i))), axis=-1)
            if (int(h[i].numpy()), int(r[i].numpy())) in self.dataset.t_mask:
                t_mask = copy(self.dataset.t_mask[(int(h[i].numpy()), int(r[i].numpy()))])
                t_mask.remove(int(t[i].numpy()))
                ents = known_ents[np.isin(known_ents, t_mask, True, True)]
            else:
                ents = known_ents
            filtered_scores = scores_[np.isin(scores_[:, -1], ents, True), :]
            filtered_ent_idx = int(np.where(filtered_scores[:, -1] == int(t[i].numpy()))[0])
            ranks_ = np.argsort(filtered_scores[:, 0], 0)
            ranks.append(int(np.where(ranks_ == filtered_ent_idx)[0])+1)
        return ranks


#######################################################
#  Architecture Modification Processors
#######################################################
class PNNTrainBatchProcessor(TrainBatchProcessor):
    def __init__(self, cmd_args):
        super(PNNTrainBatchProcessor, self).__init__(cmd_args)

    def process_epoch(self, model, optimizer):
        model_was_training = model.training
        if not model_was_training:
            model.train()

        if self.args.session != 0:
            frozen_ent_indices = model.prev_ents
            frozen_rel_indices = model.prev_rels

        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.forward(bh.contiguous().to(self.args.device),
                                       br.contiguous().to(self.args.device),
                                       bt.contiguous().to(self.args.device),
                                       by.contiguous().to(self.args.device))
            batch_loss.backward()

            if self.args.session != 0:
                # zero-out grads for all frozen params as PNN calls for
                for param_group in model.named_parameters():
                    name, param_tensor = param_group
                    if "ent" in name:
                        if param_tensor.grad is not None:
                            param_tensor.grad[frozen_ent_indices] = 0
                    else:
                        if param_tensor.grad is not None:
                            param_tensor.grad[frozen_rel_indices] = 0

            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


class CWRTrainBatchProcessor(TrainBatchProcessor):
    def __init__(self, cmd_args):
        super(CWRTrainBatchProcessor, self).__init__(cmd_args)

    def reinit_tw(self, model):
        # performs the re-init of "TW"
        tw_model, cw_model = model
        tw_model.init_weights()
        model = tw_model, cw_model
        return model

    def copyweights_tw_2_cw(self, model):
        # performs copying of "TW" weights to "CW"
        tw_model, cw_model = model
        tw_ents = torch.tensor(self.dataset.triple_ents, dtype=torch.long)
        tw_rels = torch.tensor(self.dataset.triple_rels, dtype=torch.long)
        tw_params = tw_model.state_dict()

        for param_group in cw_model.named_parameters():
            name, param_tensor = param_group
            if "ent" in name:
                param_tensor.data[tw_ents] = deepcopy((param_tensor.data[tw_ents] * cw_model.cw_ent_updates[tw_ents, None] + tw_params[name].data[tw_ents]) / (1 + cw_model.cw_ent_updates[tw_ents])[:, None])
                cw_model.cw_ent_updates[tw_ents] += 1
            else:
                param_tensor.data[tw_rels] = deepcopy((param_tensor.data[tw_rels] * cw_model.cw_rel_updates[tw_rels, None] + tw_params[name].data[tw_rels]) / (1 + cw_model.cw_rel_updates[tw_rels])[:, None])
                cw_model.cw_ent_updates[tw_rels] += 1

        model = tw_model, cw_model
        return model

    def process_epoch(self, model, optimizer):
        tw_model, cw_model = model

        model_was_training = tw_model.training
        if not model_was_training:
            tw_model.train()

        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = tw_model.forward(bh.contiguous().to(self.args.device),
                                          br.contiguous().to(self.args.device),
                                          bt.contiguous().to(self.args.device),
                                          by.contiguous().to(self.args.device))
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


#######################################################
#  Regularization Processors
#######################################################
class SITrainBatchProcessor(TrainBatchProcessor):
    def __init__(self, cmd_args):
        super(SITrainBatchProcessor, self).__init__(cmd_args)

    def process_epoch(self, model, optimizer):
        model_was_training = model.training
        if not model_was_training:
            model.train()

        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.forward(bh.contiguous().to(self.args.device),
                                       br.contiguous().to(self.args.device),
                                       bt.contiguous().to(self.args.device),
                                       by.contiguous().to(self.args.device))
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

            # update SI variables
            model.update_W()

        return total_loss


class L2TrainBatchProcessor(TrainBatchProcessor):
    def __init__(self, cmd_args):
        super(L2TrainBatchProcessor, self).__init__(cmd_args)


#######################################################
#  Replay Processors
#######################################################
class DGRTrainBatchProcessor(TrainBatchProcessor):
    def __init__(self, cmd_args):
        super(DGRTrainBatchProcessor, self).__init__(cmd_args)


class GRUVAETrainBatchProcessor:
    def __init__(self, cmd_args):
        self.args = copy(cmd_args)
        self.batch_size = int(self.args.gruvae_args[3])
        # dataset and data loader
        self.args.set_name = "train2id"
        self.dataset = data_utils.TripleSequenceDataset(self.args.dataset)
        self.dataset.load_triple_set(self.args.set_name)
        self.data_loader = None
        self.reset_data_loader()
        self.triple_set = None
        self.reset_triple_set()
        # model and optim
        self.args.model += "_gruvae"
        self.args.opt_method = "adam"
        self.args.opt_params = [float(self.args.gruvae_args[4])]
        self.args.num_ents = len(self.dataset.e2i)
        self.args.num_rels = len(self.dataset.r2i)
        self.args.sot = copy(self.dataset.sot)
        self.args.eot = copy(self.dataset.eot)
        self.model = init_model(self.args)
        self.model.to(self.args.device, non_blocking=True)
        self.optim = init_optimizer(self.args, self.model)

    def load_model(self):
        self.model = load_model(self.args, self.model)

    def save_model(self):
        save_model(self.args, self.model)

    def extend_dataset(self, triples):
        samples = np.ndarray(shape=(0, 3), dtype=int)
        for row in range(triples.shape[0]):
            sample = [[self.dataset.w2i[self.dataset.i2r[triples[row, 1]]],
                       self.dataset.w2i[self.dataset.i2e[triples[row, 0]]],
                       self.dataset.w2i[self.dataset.i2e[triples[row, 2]]]]]
            samples = np.append(samples, sample, axis=0)
        self.dataset.triples = np.append(self.dataset.triples, samples, axis=0)
        self.dataset.triples = np.unique(self.dataset.triples, axis=0)
        self.dataset.load_bernouli_sampling_stats()

    def reset_model(self):
        self.model.init_weights()

    def reset_data_loader(self):
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      num_workers=self.args.num_workers)

    def reset_triple_set(self):
        triple_set = []
        for triple_idx in range(self.dataset.triples.shape[0]):
            triple_set.append(tuple(self.dataset.triples[triple_idx, :]))
        self.triple_set = set(triple_set)

    def train_model(self, viz):
        best_epoch = None
        best_performance = np.asarray([[0.0, 0.0, 0.0]])
        valid_freq = float(self.args.gruvae_args[0])
        patience = float(self.args.gruvae_args[1])
        early_stop_trigger = -int(patience / valid_freq)
        num_valid_sampling_steps = int(self.dataset.triples.shape[0] / self.batch_size)
        num_valid_samples = self.dataset.triples.shape[0]
        for epoch in range(int(self.args.gruvae_args[2])):
            # validate vae
            if epoch % valid_freq == 0:
                dev_performance = self.valid_epoch(num_valid_sampling_steps, "steps")
                viz.add_gruvae_de_sample(dev_performance)

                if dev_performance[0, 2] * dev_performance[0, 1] > best_performance[0, 2] * best_performance[0, 1]:
                    best_performance = copy(dev_performance)
                    best_epoch = copy(epoch)
                    self.save_model()
                    early_stop_trigger = -int(patience / valid_freq)
                elif self.model.compute_anneal() > self.model.anneal_max - 0.01:
                    early_stop_trigger += 1

                if early_stop_trigger > 0:
                    break

            # train vae
            viz.add_gruvae_tr_sample(self.train_epoch())

        self.load_model()
        best_performance = self.valid_epoch(num_valid_samples, "samples")
        viz.add_gruvae_de_sample(best_performance)
        se = SizeEstimator(copy(self.args))
        model_params_size = se.estimate_size(self.model)[0]
        log_train(best_performance, best_epoch, self.args.sess,
                  self.args.num_sess, "g", 0, model_params_size,
                  viz.log_fp, self.args.log_num)
        del se

        return best_performance, best_epoch

    def train_epoch(self):
        model_was_training = self.model.training
        if not model_was_training:
            self.model.train()

        total_loss = 0.0
        total_rc_loss = 0.0
        total_kl_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.optim.zero_grad()
            batch_loss, rc_loss, kl_loss = \
                self.model.forward(batch["input"].to(self.args.device),
                                   batch["target"].to(self.args.device))
            batch_loss.backward()
            self.optim.step()
            total_loss += batch_loss.item()
            total_rc_loss += np.mean(np.sum(rc_loss, axis=1), axis=0)
            total_kl_loss += np.mean(kl_loss, axis=0)

        anneal_weight = self.model.compute_anneal()
        self.model.step_anneal()

        return total_loss, total_rc_loss, total_kl_loss, anneal_weight

    def valid_epoch(self, n, mode):
        model_was_training = self.model.training
        if model_was_training:
            self.model.eval()

        if self.args.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            samples, _ = self.get_samples(n, mode)
            precision,  u_precision, coverage = self.get_sample_stats(samples)

        return np.asarray([[precision, u_precision, coverage]])

    def get_sample_stats(self, samples):
        tp = 0.0
        fp = 0.0
        tp_set = []
        fp_set = []
        for sample_idx in range(samples.shape[0]):
            triple = tuple(samples[sample_idx, :].astype(int))
            if triple in self.triple_set:
                tp += 1.0
                tp_set.append(triple)
            else:
                fp += 1.0
                fp_set.append(triple)
        triple_set = copy(tp_set + fp_set)
        triple_set = set(triple_set)
        tp_set = set(tp_set)
        if bool(triple_set):
            tpu = 0.0
            fpu = 0.0
            for triple in triple_set:
                if triple in self.triple_set:
                    tpu += 1.0
                else:
                    fpu += 1.0
        else:
            tpu = tp
            fpu = fp
        precision = tp / (tp + fp)
        precisionu = tpu / (tpu + fpu)
        coverage = float(len(tp_set)) / float(len(self.triple_set))

        return precision, precisionu, coverage

    def get_samples(self, n, mode):
        samples = np.ndarray(shape=(0, 3), dtype=int)
        triples = np.ndarray(shape=(0, 3), dtype=int)
        if mode == "steps":
            for sampling_step in range(n):
                samples = np.append(samples,
                                    self.model.sample(self.batch_size)[:, :-1],
                                    axis=0)
        elif mode == "samples":
            while samples.shape[0] < n * 95.0 / 100.0:
                samples = np.append(samples,
                                    self.model.sample(self.batch_size)[:, :-1].astype(int),
                                    axis=0)
                samples = np.unique(samples, axis=0)

            for row in range(samples.shape[0]):
                try:
                    triple = [[self.dataset.e2i[self.dataset.i2w[samples[row, 1]]],
                               self.dataset.r2i[self.dataset.i2w[samples[row, 0]]],
                               self.dataset.e2i[self.dataset.i2w[samples[row, 2]]]]]
                except KeyError as exc:
                    continue
                triples = np.append(triples, triple, axis=0)

            logout(str(samples.shape[0] - triples.shape[0]) + " samples did not exist in the vocabulary.")
        else:
            logout("Sample mode does not exist, no samples generated.", "e")

        return samples, triples


def collate_batch(batch):
    batch = tensor(batch)
    batch_h = batch[:, :, 0].flatten()
    batch_r = batch[:, :, 1].flatten()
    batch_t = batch[:, :, 2].flatten()
    batch_y = batch[:, :, 3].flatten()
    return batch_h, batch_r, batch_t, batch_y


def init_model(args):
    model = None
    if args.model == "transe":
        if args.cl_method == "PNN":
            model = pnn_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                      args.neg_ratio, args.batch_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "CWR":
            cw_model = cwr_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                         args.neg_ratio, args.batch_size, args.device)
            tw_model = cwr_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                         args.neg_ratio, args.batch_size, args.device)
            cw_model.to(args.device, non_blocking=True)
            tw_model.to(args.device, non_blocking=True)
            model = [tw_model, cw_model]
        elif args.cl_method == "SI":
            model = si_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                     args.neg_ratio, args.batch_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "L2":
            model = l2_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                     args.neg_ratio, args.batch_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "DGR":
            model = dgr_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                      args.neg_ratio, args.batch_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "finetune" or args.cl_method == "offline":
            model = std_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                      args.neg_ratio, args.batch_size, args.device)
            model.to(args.device, non_blocking=True)
        else:
            logout("The CL method '" + str(args.cl_method) + "' to be used is not implemented for TransE.", "f")
            exit()
    elif args.model == "analogy":
        if args.cl_method == "PNN":
            model = pnn_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "CWR":
            cw_model = cwr_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
            tw_model = cwr_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
            cw_model.to(args.device, non_blocking=True)
            tw_model.to(args.device, non_blocking=True)
            model = [tw_model, cw_model]
        elif args.cl_method == "SI":
            model = si_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "L2":
            model = l2_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "DGR":
            model = dgr_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
            model.to(args.device, non_blocking=True)
        elif args.cl_method == "finetune" or args.cl_method == "offline":
            model = std_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
            model.to(args.device, non_blocking=True)
        else:
            logout("The CL method '" + str(args.cl_method) + "' to be used is not implemented.", "f")
            exit()
    elif "gruvae" in args.model:
        e_dim, h_dim, z_dim, a_slope, a_pos, a_max = args.gruvae_args[5:]
        encoder = dgr_models.GRUEncoder(int(e_dim), int(h_dim), int(z_dim),
                                        args.num_ents + args.num_rels + 2, args.device)
        decoder = dgr_models.GRUDecoder(int(z_dim), int(h_dim), int(e_dim),
                                        args.num_ents + args.num_rels + 2, args.device)
        model = dgr_models.TripleGRUVAE(encoder, decoder, args.sot, args.eot, args.device,
                                        float(a_slope), float(a_pos), float(a_max))
        model.to(args.device, non_blocking=True)
    else:
        logout("The model '" + str(args.model) + "' to be used is not implemented.", "f")
        exit()
    return model


def init_optimizer(args, model):
    if args.cl_method == "CWR":
        tw_model, cw_model = model
        optim_model = tw_model
    else:
        optim_model = model
    optimizer = None
    if args.opt_method == "adagrad":
        try:
            lr = args.opt_params[0]
            optimizer = optim.Adagrad(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for adagrad are [-op lr]", "f")
            exit()
    elif args.opt_method == "adadelta":
        try:
            lr = args.opt_params[0]
            optimizer = optim.Adadelta(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for adadelta are [-op lr]", "f")
            exit()
    elif args.opt_method == "adam":
        try:
            lr = args.opt_params[0]
            optimizer = optim.Adam(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for adam are [-op lr]", "f")
            exit()
    elif args.opt_method == "sgd":
        try:
            lr = args.opt_params[0]
            optimizer = optim.SGD(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for sgd are [-op lr]", "f")
            exit()
    else:
        logout("Optimization options are 'adagrad','adadelta','adam','sgd'", "f")
        exit()

    return optimizer


def save_model(args, model):
    checkpoints_fp = abspath(dirname(__file__)) + "/checkpoints/"
    checkpoint_name = str(args.tag) + "__"
    checkpoint_name += "sess" + str(args.sess) + "_"
    checkpoint_name += str(args.dataset) + "_"
    checkpoint_name += "mt" + str(args.model) + "_"
    checkpoint_name += "clm" + str(args.cl_method) + "_"
    checkpoint_name += "ln" + str(args.log_num)

    if args.cl_method == "CWR":
        tw_model, cw_model = model
        save_checkpoint(tw_model.state_dict(), checkpoints_fp + checkpoint_name + "_tw")
        save_checkpoint(cw_model.state_dict(), checkpoints_fp + checkpoint_name + "_cw")
    else:
        save_checkpoint(model.state_dict(), checkpoints_fp + checkpoint_name)


def save_checkpoint(params, filename):
    try:
        torch.save(params, filename)
        # logout('Written to: ' + filename)
    except Exception as e:
        logout("Could not save: " + filename, "w")
        raise e


def load_model(args, model):
    checkpoints_fp = abspath(dirname(__file__)) + "/checkpoints/"
    checkpoint_name = str(args.tag) + "__"
    checkpoint_name += "sess" + str(args.sess) + "_"
    checkpoint_name += str(args.dataset) + "_"
    checkpoint_name += "mt" + str(args.model) + "_"
    checkpoint_name += "clm" + str(args.cl_method) + "_"
    checkpoint_name += "ln" + str(args.log_num)

    if args.cl_method == "CWR":
        tw_model, cw_model = model
        tw_model = load_checkpoint(tw_model, checkpoints_fp + checkpoint_name + "_tw")
        cw_model = load_checkpoint(cw_model, checkpoints_fp + checkpoint_name + "_cw")
        model = tw_model, cw_model
    else:
        model = load_checkpoint(model, checkpoints_fp + checkpoint_name)
    return model


def load_checkpoint(model, filename):
    try:
        model.load_state_dict(load(filename), strict=False)
    except Exception as e:
        logout("Could not load: " + filename, "w")
        raise e
    return model


def evaluate_model(args, sess, batch_processors, model):
    performances = np.ndarray(shape=(0, 2))
    for valid_sess in range(args.num_sess):
        eval_bp = batch_processors[valid_sess]
        if args.cl_method == "CWR":
            tw_model, cw_model = model
            if valid_sess == sess:
                performance = eval_bp.process_epoch(tw_model)
            else:
                performance = eval_bp.process_epoch(cw_model)
        else:
            performance = eval_bp.process_epoch(model)
        performances = np.append(performances, [performance], axis=0)
    return performances


class EarlyStopTracker:
    def __init__(self, args):
        self.args = args
        self.num_epoch = args.num_epochs
        self.epoch = 0
        self.valid_freq = args.valid_freq
        self.patience = args.patience
        self.early_stop_trigger = -int(self.patience / self.valid_freq)
        self.last_early_stop_value = 0.0
        self.best_performances = None
        self.best_measure = 0.0
        self.best_epoch = None

    def continue_training(self):
        return not bool(self.epoch > self.num_epoch or self.early_stop_trigger > 0)

    def get_epoch(self):
        return self.epoch

    def validate(self):
        return bool(self.epoch % self.valid_freq == 0)

    def update_best(self, sess, performances, model):
        measure = performances[sess, 1]
        # checks for new best model and saves if so
        if measure > self.best_measure:
            self.best_measure = copy(measure)
            self.best_epoch = copy(self.epoch)
            self.best_performances = np.copy(performances)
            save_model(self.args, model)
        # checks for reset of early stop trigger
        if measure - 0.01 > self.last_early_stop_value:
            self.last_early_stop_value = copy(measure)
            self.early_stop_trigger = -int(self.patience / self.valid_freq)
        else:
            self.early_stop_trigger += 1
        # adjusts valid frequency throughout training
        if self.epoch >= 400:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 50.0
            self.valid_freq = 50
        elif self.epoch >= 200:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 25.0
            self.valid_freq = 25
        elif self.epoch >= 50:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 10.0
            self.valid_freq = 10

    def step_epoch(self):
        self.epoch += 1

    def get_best(self):
        return self.best_performances, self.best_epoch



if __name__ == "__main__":
    # TODO add unit tests below
    pass
