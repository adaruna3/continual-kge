from os.path import abspath,dirname,exists
from os import makedirs
from copy import copy
import numpy as np
from argparse import ArgumentParser
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from logger.terminal_utils import logout
from datasets import data_utils

import pdb


class CLDatasetEntitySampler:
    def __init__(self, dataset_name, entity_sample_rate=0.5, max_sessions=5):
        original_dataset = data_utils.TripleDataset(dataset_name)
        datasets_fp = abspath(dirname(__file__)) + '/'
        self.prev_fp = datasets_fp + dataset_name
        self.dataset_name = dataset_name
        self.fps = []
        for sess_idx in range(max_sessions):
            self.fps.append(self.prev_fp + "_" + str(sess_idx) + "/")
            if not exists(self.fps[sess_idx]):
                makedirs(self.fps[sess_idx])
        original_dataset.load_triple_set("train2id")
        self.sros_tr = copy(original_dataset.triples)
        original_dataset.load_triple_set("valid2id")
        self.sros_de = copy(original_dataset.triples)
        original_dataset.load_triple_set("test2id")
        self.sros_te = copy(original_dataset.triples)
        self.e2i = original_dataset.e2i
        self.i2e = original_dataset.i2e
        self.r2i = original_dataset.r2i
        self.i2r = original_dataset.i2r
        self.esr = entity_sample_rate
        self.ms = max_sessions
        self.sess_ents = []
        self.unk_ents = set()
        self.unk_rels = set()
        self.num_triples = self.sros_tr.shape[0] + self.sros_de.shape[0] + self.sros_te.shape[0]

    def generate_splits(self):
        np.random.seed(0)  # for benchmark dataset repeatability
        sampled_count = np.zeros(shape=(len(self.e2i)))
        for sess_idx in range(self.ms):
            tops = 1 / (1 + sampled_count)
            probs = tops / np.sum(tops)
            num_sess_ents = int(len(self.e2i) * self.esr)
            sess_ents = np.random.choice(np.arange(len(self.e2i)), num_sess_ents, False, probs)
            self.sess_ents.append(sess_ents)
            for idx in  range(len(sampled_count)):
                if idx in sess_ents:
                    sampled_count[idx] += 1

    def get_sess_stats(self):
        seen_ents = set()
        seen_rels = set()
        seen_tr_triples = set()
        seen_de_triples = set()
        seen_te_triples = set()
        stats = np.ndarray(shape=(20, self.ms))

        for sess_idx in range(self.ms):
            # get data
            sess_ents = set(copy(self.sess_ents[sess_idx]))
            sess_tr, sess_de, sess_te = self.filter_triples(sess_ents)
            sess_rels = self.get_rels(sess_tr)
            # get stats
            sess_tr_triples = set()
            sess_de_triples = set()
            sess_te_triples = set()
            for triple_idx in range(sess_tr.shape[0]):
                triple = tuple(sess_tr[triple_idx, :].astype(int).tolist())
                sess_tr_triples = sess_tr_triples.union({triple})
            for triple_idx in range(sess_de.shape[0]):
                triple = tuple(sess_de[triple_idx, :].astype(int).tolist())
                sess_de_triples = sess_de_triples.union({triple})
            for triple_idx in range(sess_te.shape[0]):
                triple = tuple(sess_te[triple_idx, :].astype(int).tolist())
                sess_te_triples = sess_te_triples.union({triple})

            new_tr_triple_rate = 100.0 * float(len(sess_tr_triples.difference(seen_tr_triples))) / float(len(sess_tr_triples))
            new_de_triple_rate = 100.0 * float(len(sess_de_triples.difference(seen_de_triples))) / float(len(sess_de_triples))
            new_te_triple_rate = 100.0 * float(len(sess_te_triples.difference(seen_te_triples))) / float(len(sess_te_triples))
            new_ent_rate = 100.0 * float(len(sess_ents.difference(seen_ents))) / float(len(sess_ents))
            new_rel_rate = 100.0 * float(len(sess_rels.difference(seen_rels))) / float(len(sess_rels))

            seen_tr_triples = seen_tr_triples.union(sess_tr_triples)
            seen_de_triples = seen_de_triples.union(sess_de_triples)
            seen_te_triples = seen_te_triples.union(sess_te_triples)
            seen_ents = seen_ents.union(sess_ents)
            seen_rels = seen_rels.union(sess_rels)
            sess_num_tr_triples = str(len(sess_tr_triples))
            accu_num_tr_triples = str(len(seen_tr_triples))
            sess_num_de_triples = str(len(sess_de_triples))
            accu_num_de_triples = str(len(seen_de_triples))
            sess_num_te_triples = str(len(sess_te_triples))
            accu_num_te_triples = str(len(seen_te_triples))
            sess_num_ents = str(len(sess_ents))
            accu_num_ents = str(len(seen_ents))
            sess_num_rels = str(len(sess_rels))
            accu_num_rels = str(len(seen_rels))
            tr_triple_coverage = 100.0 * len(seen_tr_triples) / self.sros_tr.shape[0]
            de_triple_coverage = 100.0 * len(seen_de_triples) / self.sros_de.shape[0]
            te_triple_coverage = 100.0 * len(seen_te_triples) / self.sros_te.shape[0]
            ent_coverage = 100.0 * len(seen_ents) / len(self.e2i)
            rel_coverage = 100.0 * len(seen_rels) / len(self.r2i)

            # train triples
            logout("There are " + str(len(sess_num_tr_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[0, sess_idx] = sess_num_tr_triples
            logout("There are " + "{:.2f}".format(new_tr_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[1, sess_idx] = "{:.2f}".format(new_tr_triple_rate)
            logout("There are " + accu_num_tr_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[2, sess_idx] = accu_num_tr_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(tr_triple_coverage), "i")
            stats[3, sess_idx] = "{:.2f}".format(tr_triple_coverage)
            # dev triples
            logout("There are " + str(len(sess_num_de_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[4, sess_idx] = sess_num_de_triples
            logout("There are " + "{:.2f}".format(new_de_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[5, sess_idx] = "{:.2f}".format(new_de_triple_rate)
            logout("There are " + accu_num_de_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[6, sess_idx] = accu_num_de_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(de_triple_coverage), "i")
            stats[7, sess_idx] = "{:.2f}".format(de_triple_coverage)
            # test triples
            logout("There are " + str(len(sess_num_te_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[8, sess_idx] = sess_num_te_triples
            logout("There are " + "{:.2f}".format(new_te_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[9, sess_idx] = "{:.2f}".format(new_te_triple_rate)
            logout("There are " + accu_num_te_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[10, sess_idx] = accu_num_te_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(te_triple_coverage), "i")
            stats[11, sess_idx] = "{:.2f}".format(te_triple_coverage)
            # ents
            logout("There are " + sess_num_ents + " entities for session " + str(sess_idx) + ".", "i")
            stats[12, sess_idx] = sess_num_ents
            logout("There are " + "{:.2f}".format(new_ent_rate) + "% new ents for session " + str(sess_idx) + ".", "i")
            stats[13, sess_idx] = "{:.2f}".format(new_ent_rate)
            logout("There are " + accu_num_ents + " accumulated entities after session " + str(sess_idx) + ".", "i")
            stats[14, sess_idx] = accu_num_ents
            logout("Ent coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(ent_coverage), "i")
            stats[15, sess_idx] = "{:.2f}".format(ent_coverage)
            # relations
            logout("There are " + sess_num_rels + " relations for session " + str(sess_idx) + ".", "i")
            stats[16, sess_idx] = sess_num_rels
            logout("There are " + "{:.2f}".format(new_rel_rate) + "% new rels for session " + str(sess_idx) + ".", "i")
            stats[17, sess_idx] = "{:.2f}".format(new_rel_rate)
            logout("There are " + accu_num_rels + " accumulated relations after session " + str(sess_idx) + ".", "i")
            stats[18, sess_idx] = accu_num_rels
            logout("Rel coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(rel_coverage), "i")
            stats[19, sess_idx] = "{:.2f}".format(rel_coverage)

        row_labels = ["Num Train Triples",
                      "% New Train Triples",
                      "Num Accu. Train Triples",
                      "Train Triple Coverage",
                      "Num Dev Triples",
                      "% New Dev Triples",
                      "Num Accu. Dev Triples",
                      "Dev Triple Coverage",
                      "Num Test Triples",
                      "% New Test Triples",
                      "Num Accu. Test Triples",
                      "Test Triple Coverage",
                      "Num Ents",
                      "% New Ents",
                      "Num Accu. Ents",
                      "Ent Coverage",
                      "Num Rels",
                      "% New Rels",
                      "Num Accu. Rels",
                      "Rel Coverage"]
        col_labels = ["S" + str(i + 1) for i in range(self.ms)]

        fig = plt.figure(figsize=(10, 6))
        axs = fig.add_subplot(1, 1, 1)
        fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        plt.grid('off')
        the_table = axs.table(cellText=stats, rowLabels=row_labels, colLabels=col_labels, loc='center')
        fig.tight_layout()
        plt.savefig(str(self.dataset_name + "_" + str(self.ms) + "_ES.pdf"), bbox_inches="tight")

    def filter_triples(self, fitler_ents):
        tr_tr = np.ndarray(shape=(0, 3), dtype=np.int32)
        de_tr = np.ndarray(shape=(0, 3), dtype=np.int32)
        te_tr = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(len(self.sros_tr)):
            if self.sros_tr[idx][0] not in fitler_ents or self.sros_tr[idx][2] not in fitler_ents:
                continue
            else:
                tr_tr = np.append(tr_tr, [self.sros_tr[idx]], axis=0)
        for idx in range(len(self.sros_de)):
            if self.sros_de[idx][0] not in fitler_ents or self.sros_de[idx][2] not in fitler_ents:
                continue
            else:
                de_tr = np.append(de_tr, [self.sros_de[idx]], axis=0)
        for idx in range(len(self.sros_te)):
            if self.sros_te[idx][0] not in fitler_ents or self.sros_te[idx][2] not in fitler_ents:
                continue
            else:
                te_tr = np.append(te_tr, [self.sros_te[idx]], axis=0)
        return tr_tr, de_tr, te_tr

    def get_rels(self, triples):
        rels = set()
        for triple in triples:
            rels.add(triple[1])
        return rels

    def save(self, dkge=False):
        known_ents = set()
        known_rels = set()
        for sess_idx in range(self.ms):
            filter_ents = self.sess_ents[sess_idx]
            known_ents = known_ents.union(set(copy(filter_ents)))
            sess_tr, sess_de, sess_te = self.filter_triples(filter_ents)
            filter_rels = self.get_rels(np.concatenate((sess_tr, sess_de, sess_te), axis=0))
            known_rels = known_rels.union(set(copy(filter_rels)))
            self.write_triples_to_file(self.fps[sess_idx] + "train2id.txt", sess_tr)
            self.write_triples_to_file(self.fps[sess_idx] + "valid2id.txt", sess_de)
            self.write_triples_to_file(self.fps[sess_idx] + "test2id.txt", sess_te)
            copyfile(self.prev_fp + "/relation2id.txt", self.fps[sess_idx] + "relation2id.txt")
            copyfile(self.prev_fp + "/entity2id.txt", self.fps[sess_idx] + "entity2id.txt")
            self.write_known_to_file(self.fps[sess_idx] + "known_ents.txt", known_ents, self.i2e)
            self.write_known_to_file(self.fps[sess_idx] + "known_rels.txt", known_rels, self.i2r)
            self.write_sess_to_file(self.fps[sess_idx] + "sess_ents.txt", filter_ents, self.i2e)
            self.write_sess_to_file(self.fps[sess_idx] + "sess_rels.txt", filter_rels, self.i2r)
            logout("Dataset session " + str(sess_idx) + " written to file", "s")
            if dkge:
                self.save_dkge_dataset(sess_tr, sess_de, sess_te, filter_ents, filter_rels, self.fps[sess_idx])
            logout("DKGE Dataset session " + str(sess_idx) + " written to file", "s")

    def write_triples_to_file(self, filename, triples):
        with open(filename, "w") as f:
            f.write(str(triples.shape[0]) + "\n")
            for triple in triples:
                f.write(str(triple[0]) + " " + str(triple[2]) + " " + str(triple[1]) + "\n")

    def write_known_to_file(self, filename, ids, lookup):
        with open(filename, "w") as f:
            for id in ids:
                f.write(str(lookup[id]) + "\n")

    def write_sess_to_file(self, filename, ids, lookup):
        with open(filename, "w") as f:
            f.write(str(len(ids)) + "\n")
            for id in ids:
                f.write(str(lookup[id]) + "\t" + str(id) + "\n")

    def save_dkge_dataset(self, trtr, detr, tetr, ents, rels, fp):
        # remaps the ids of entities and relations to be compatible with the dkge code
        e2i_ = {}
        i2e_ = {}
        r2i_ = {}
        i2r_ = {}
        ents = list(ents)
        rels = list(rels)

        for i in range(len(ents)):
            e2i_[self.i2e[ents[i]]] = i
            i2e_[i] = self.i2e[ents[i]]

        for i in range(len(rels)):
            r2i_[self.i2r[rels[i]]] = i
            i2r_[i] = self.i2r[rels[i]]

        trtr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(trtr.shape[0]):
            h, r, t = trtr[idx]
            try:
                tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            except:
                pdb.set_trace()
            trtr_ = np.append(trtr_, [tr], axis=0)

        detr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(detr.shape[0]):
            h, r, t = detr[idx]
            tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            detr_ = np.append(detr_, [tr], axis=0)

        tetr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(tetr.shape[0]):
            h, r, t = tetr[idx]
            tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            tetr_ = np.append(tetr_, [tr], axis=0)

        fp += "dkge/"
        if not exists(fp):
            makedirs(fp)

        self.write_triples_to_file(fp + "train2id.txt", trtr_)
        self.write_triples_to_file(fp + "valid2id.txt", detr_)
        self.write_triples_to_file(fp + "test2id.txt", tetr_)
        self.write_sess_to_file(fp + "entity2id.txt", sorted(list(e2i_.values())), i2e_)
        self.write_sess_to_file(fp + "relation2id.txt", sorted(list(r2i_.values())), i2r_)


class CLDatasetRelationSampler:
    def __init__(self, dataset_name, relation_sample_rate=0.5, max_sessions=5):
        original_dataset = data_utils.TripleDataset(dataset_name)
        datasets_fp = abspath(dirname(__file__)) + '/'
        self.prev_fp = datasets_fp + dataset_name
        self.dataset_name = dataset_name
        self.fps = []
        for sess_idx in range(max_sessions):
            self.fps.append(self.prev_fp + "_" + str(sess_idx) + "/")
            if not exists(self.fps[sess_idx]):
                makedirs(self.fps[sess_idx])
        original_dataset.load_triple_set("train2id")
        self.sros_tr = copy(original_dataset.triples)
        original_dataset.load_triple_set("valid2id")
        self.sros_de = copy(original_dataset.triples)
        original_dataset.load_triple_set("test2id")
        self.sros_te = copy(original_dataset.triples)
        self.e2i = original_dataset.e2i
        self.i2e = original_dataset.i2e
        self.r2i = original_dataset.r2i
        self.i2r = original_dataset.i2r
        self.rsr = relation_sample_rate
        self.ms = max_sessions
        self.sess_rels = []
        self.unk_ents = set()
        self.unk_rels = set()
        self.num_triples = self.sros_tr.shape[0] + self.sros_de.shape[0] + self.sros_te.shape[0]

    def generate_splits(self):
        np.random.seed(0)  # for benchmark dataset repeatability
        sampled_count = np.zeros(shape=(len(self.r2i)))
        for sess_idx in range(self.ms):
            tops = 1 / (1 + sampled_count)
            probs = tops / np.sum(tops)
            num_sess_rels = int(len(self.r2i) * self.rsr)
            sess_rels = np.random.choice(np.arange(len(self.r2i)), num_sess_rels, False, probs)
            self.sess_rels.append(sess_rels)
            for idx in range(len(sampled_count)):
                if idx in sess_rels:
                    sampled_count[idx] += 1

    def get_sess_stats(self):
        seen_ents = set()
        seen_rels = set()
        seen_tr_triples = set()
        seen_de_triples = set()
        seen_te_triples = set()
        stats = np.ndarray(shape=(20, self.ms))

        for sess_idx in range(self.ms):
            # get data
            sess_rels = set(copy(self.sess_rels[sess_idx]))
            sess_tr, sess_de, sess_te, sess_ents = self.filter_triples(sess_rels)
            # get stats
            sess_tr_triples = set()
            sess_de_triples = set()
            sess_te_triples = set()
            for triple_idx in range(sess_tr.shape[0]):
                triple = tuple(sess_tr[triple_idx, :].astype(int).tolist())
                sess_tr_triples = sess_tr_triples.union({triple})
            for triple_idx in range(sess_de.shape[0]):
                triple = tuple(sess_de[triple_idx, :].astype(int).tolist())
                sess_de_triples = sess_de_triples.union({triple})
            for triple_idx in range(sess_te.shape[0]):
                triple = tuple(sess_te[triple_idx, :].astype(int).tolist())
                sess_te_triples = sess_te_triples.union({triple})

            new_tr_triple_rate = 100.0 * float(len(sess_tr_triples.difference(seen_tr_triples))) / float(len(sess_tr_triples))
            new_de_triple_rate = 100.0 * float(len(sess_de_triples.difference(seen_de_triples))) / float(len(sess_de_triples))
            new_te_triple_rate = 100.0 * float(len(sess_te_triples.difference(seen_te_triples))) / float(len(sess_te_triples))
            new_ent_rate = 100.0 * float(len(sess_ents.difference(seen_ents))) / float(len(sess_ents))
            new_rel_rate = 100.0 * float(len(sess_rels.difference(seen_rels))) / float(len(sess_rels))

            seen_tr_triples = seen_tr_triples.union(sess_tr_triples)
            seen_de_triples = seen_de_triples.union(sess_de_triples)
            seen_te_triples = seen_te_triples.union(sess_te_triples)
            seen_ents = seen_ents.union(sess_ents)
            seen_rels = seen_rels.union(sess_rels)
            sess_num_tr_triples = str(len(sess_tr_triples))
            accu_num_tr_triples = str(len(seen_tr_triples))
            sess_num_de_triples = str(len(sess_de_triples))
            accu_num_de_triples = str(len(seen_de_triples))
            sess_num_te_triples = str(len(sess_te_triples))
            accu_num_te_triples = str(len(seen_te_triples))
            sess_num_ents = str(len(sess_ents))
            accu_num_ents = str(len(seen_ents))
            sess_num_rels = str(len(sess_rels))
            accu_num_rels = str(len(seen_rels))
            tr_triple_coverage = 100.0 * len(seen_tr_triples) / self.sros_tr.shape[0]
            de_triple_coverage = 100.0 * len(seen_de_triples) / self.sros_de.shape[0]
            te_triple_coverage = 100.0 * len(seen_te_triples) / self.sros_te.shape[0]
            ent_coverage = 100.0 * len(seen_ents) / len(self.e2i)
            rel_coverage = 100.0 * len(seen_rels) / len(self.r2i)

            # train triples
            logout("There are " + str(len(sess_num_tr_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[0, sess_idx] = sess_num_tr_triples
            logout("There are " + "{:.2f}".format(new_tr_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[1, sess_idx] = "{:.2f}".format(new_tr_triple_rate)
            logout("There are " + accu_num_tr_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[2, sess_idx] = accu_num_tr_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(tr_triple_coverage), "i")
            stats[3, sess_idx] = "{:.2f}".format(tr_triple_coverage)
            # dev triples
            logout("There are " + str(len(sess_num_de_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[4, sess_idx] = sess_num_de_triples
            logout("There are " + "{:.2f}".format(new_de_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[5, sess_idx] = "{:.2f}".format(new_de_triple_rate)
            logout("There are " + accu_num_de_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[6, sess_idx] = accu_num_de_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(de_triple_coverage), "i")
            stats[7, sess_idx] = "{:.2f}".format(de_triple_coverage)
            # test triples
            logout("There are " + str(len(sess_num_te_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[8, sess_idx] = sess_num_te_triples
            logout("There are " + "{:.2f}".format(new_te_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[9, sess_idx] = "{:.2f}".format(new_te_triple_rate)
            logout("There are " + accu_num_te_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[10, sess_idx] = accu_num_te_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(te_triple_coverage), "i")
            stats[11, sess_idx] = "{:.2f}".format(te_triple_coverage)
            # ents
            logout("There are " + sess_num_ents + " entities for session " + str(sess_idx) + ".", "i")
            stats[12, sess_idx] = sess_num_ents
            logout("There are " + "{:.2f}".format(new_ent_rate) + "% new ents for session " + str(sess_idx) + ".", "i")
            stats[13, sess_idx] = "{:.2f}".format(new_ent_rate)
            logout("There are " + accu_num_ents + " accumulated entities after session " + str(sess_idx) + ".", "i")
            stats[14, sess_idx] = accu_num_ents
            logout("Ent coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(ent_coverage), "i")
            stats[15, sess_idx] = "{:.2f}".format(ent_coverage)
            # relations
            logout("There are " + sess_num_rels + " relations for session " + str(sess_idx) + ".", "i")
            stats[16, sess_idx] = sess_num_rels
            logout("There are " + "{:.2f}".format(new_rel_rate) + "% new rels for session " + str(sess_idx) + ".", "i")
            stats[17, sess_idx] = "{:.2f}".format(new_rel_rate)
            logout("There are " + accu_num_rels + " accumulated relations after session " + str(sess_idx) + ".", "i")
            stats[18, sess_idx] = accu_num_rels
            logout("Rel coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(rel_coverage), "i")
            stats[19, sess_idx] = "{:.2f}".format(rel_coverage)

        row_labels = ["Num Train Triples",
                      "% New Train Triples",
                      "Num Accu. Train Triples",
                      "Train Triple Coverage",
                      "Num Dev Triples",
                      "% New Dev Triples",
                      "Num Accu. Dev Triples",
                      "Dev Triple Coverage",
                      "Num Test Triples",
                      "% New Test Triples",
                      "Num Accu. Test Triples",
                      "Test Triple Coverage",
                      "Num Ents",
                      "% New Ents",
                      "Num Accu. Ents",
                      "Ent Coverage",
                      "Num Rels",
                      "% New Rels",
                      "Num Accu. Rels",
                      "Rel Coverage"]
        col_labels = ["S" + str(i + 1) for i in range(self.ms)]

        fig = plt.figure(figsize=(10, 6))
        axs = fig.add_subplot(1, 1, 1)
        fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        plt.grid('off')
        the_table = axs.table(cellText=stats, rowLabels=row_labels, colLabels=col_labels, loc='center')
        fig.tight_layout()
        plt.savefig(str(self.dataset_name + "_" + str(self.ms) + "_RS.pdf"), bbox_inches="tight")

    def filter_triples(self, fitler_rels):
        tr_tr = np.ndarray(shape=(0, 3), dtype=np.int32)
        de_tr = np.ndarray(shape=(0, 3), dtype=np.int32)
        te_tr = np.ndarray(shape=(0, 3), dtype=np.int32)
        sess_ents = set()
        for idx in range(len(self.sros_tr)):
            if self.sros_tr[idx][1] not in fitler_rels:
                continue
            else:
                sess_ents.add(self.sros_tr[idx][0])
                sess_ents.add(self.sros_tr[idx][2])
                tr_tr = np.append(tr_tr, [self.sros_tr[idx]], axis=0)
        for idx in range(len(self.sros_de)):
            if (self.sros_de[idx][1] not in fitler_rels
                    or self.sros_de[idx][0] not in sess_ents
                    or self.sros_de[idx][2] not in sess_ents):
                continue
            else:
                de_tr = np.append(de_tr, [self.sros_de[idx]], axis=0)
        for idx in range(len(self.sros_te)):
            if (self.sros_te[idx][1] not in fitler_rels
                    or self.sros_te[idx][0] not in sess_ents
                    or self.sros_te[idx][2] not in sess_ents):
                continue
            else:
                te_tr = np.append(te_tr, [self.sros_te[idx]], axis=0)
        return tr_tr, de_tr, te_tr, sess_ents

    def save(self, dkge=False):
        known_ents = set()
        known_rels = set()
        for sess_idx in range(self.ms):
            sess_rels = self.sess_rels[sess_idx]
            known_rels = known_rels.union(set(copy(sess_rels)))
            sess_tr, sess_de, sess_te, sess_ents = self.filter_triples(sess_rels)
            known_ents = known_ents.union(set(copy(sess_ents)))
            self.write_triples_to_file(self.fps[sess_idx] + "train2id.txt", sess_tr)
            self.write_triples_to_file(self.fps[sess_idx] + "valid2id.txt", sess_de)
            self.write_triples_to_file(self.fps[sess_idx] + "test2id.txt", sess_te)
            copyfile(self.prev_fp + "/relation2id.txt", self.fps[sess_idx] + "relation2id.txt")
            copyfile(self.prev_fp + "/entity2id.txt", self.fps[sess_idx] + "entity2id.txt")
            self.write_known_to_file(self.fps[sess_idx] + "known_ents.txt", known_ents, self.i2e)
            self.write_known_to_file(self.fps[sess_idx] + "known_rels.txt", known_rels, self.i2r)
            self.write_sess_to_file(self.fps[sess_idx] + "sess_ents.txt", sess_ents, self.i2e)
            self.write_sess_to_file(self.fps[sess_idx] + "sess_rels.txt", sess_rels, self.i2r)
            logout("Dataset session " +str(sess_idx) + " written to file", "s")
            if dkge:
                self.save_dkge_dataset(sess_tr, sess_de, sess_te, sess_ents, sess_rels, self.fps[sess_idx])
            logout("DKGE Dataset session " + str(sess_idx) + " written to file", "s")


    def write_triples_to_file(self, filename, triples):
        with open(filename, "w") as f:
            f.write(str(triples.shape[0]) + "\n")
            for triple in triples:
                f.write(str(triple[0]) + " " + str(triple[2]) + " " + str(triple[1]) + "\n")

    def write_known_to_file(self, filename, ids, lookup):
        with open(filename, "w") as f:
            for id in ids:
                f.write(str(lookup[id]) + "\n")

    def write_sess_to_file(self, filename, ids, lookup):
        with open(filename, "w") as f:
            f.write(str(len(ids)) + "\n")
            for id in ids:
                f.write(str(lookup[id]) + "\t" + str(id) + "\n")

    def save_dkge_dataset(self, trtr, detr, tetr, ents, rels, fp):
        # remaps the ids of entities and relations to be compatible with the dkge code
        e2i_ = {}
        i2e_ = {}
        r2i_ = {}
        i2r_ = {}
        ents = list(ents)
        rels = list(rels)

        for i in range(len(ents)):
            e2i_[self.i2e[ents[i]]] = i
            i2e_[i] = self.i2e[ents[i]]

        for i in range(len(rels)):
            r2i_[self.i2r[rels[i]]] = i
            i2r_[i] = self.i2r[rels[i]]

        trtr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(trtr.shape[0]):
            h, r, t = trtr[idx]
            try:
                tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            except:
                pdb.set_trace()
            trtr_ = np.append(trtr_, [tr], axis=0)

        detr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(detr.shape[0]):
            h, r, t = detr[idx]
            tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            detr_ = np.append(detr_, [tr], axis=0)

        tetr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(tetr.shape[0]):
            h, r, t = tetr[idx]
            tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            tetr_ = np.append(tetr_, [tr], axis=0)

        fp += "dkge/"
        if not exists(fp):
            makedirs(fp)

        self.write_triples_to_file(fp + "train2id.txt", trtr_)
        self.write_triples_to_file(fp + "valid2id.txt", detr_)
        self.write_triples_to_file(fp + "test2id.txt", tetr_)
        self.write_sess_to_file(fp + "entity2id.txt",
                                sorted(list(e2i_.values())), i2e_)
        self.write_sess_to_file(fp + "relation2id.txt",
                                sorted(list(r2i_.values())), i2r_)


class CLDatasetTripleSampler:
    def __init__(self, dataset_name, triple_sample_rate=0.5, max_sessions=5):
        original_dataset = data_utils.TripleDataset(dataset_name)
        datasets_fp = abspath(dirname(__file__)) + '/'
        self.prev_fp = datasets_fp + dataset_name
        self.dataset_name = dataset_name
        self.fps = []
        for sess_idx in range(max_sessions):
            self.fps.append(self.prev_fp + "_" + str(sess_idx) + "/")
            if not exists(self.fps[sess_idx]):
                makedirs(self.fps[sess_idx])
        original_dataset.load_triple_set("train2id")
        self.sros_tr = copy(original_dataset.triples)
        original_dataset.load_triple_set("valid2id")
        self.sros_de = copy(original_dataset.triples)
        original_dataset.load_triple_set("test2id")
        self.sros_te = copy(original_dataset.triples)
        self.e2i = original_dataset.e2i
        self.i2e = original_dataset.i2e
        self.r2i = original_dataset.r2i
        self.i2r = original_dataset.i2r
        self.tsr = triple_sample_rate
        self.ms = max_sessions
        self.sess_triples = []
        self.unk_ents = set()
        self.unk_rels = set()
        self.num_triples = self.sros_tr.shape[0] + self.sros_de.shape[0] + self.sros_te.shape[0]

    def generate_splits(self):
        np.random.seed(0)  # for benchmark dataset repeatability
        train_triples = copy(self.sros_tr)
        num_sess_triples = int(len(self.sros_tr) * self.tsr)
        for sess_idx in range(self.ms):
            sess_triple_idx = np.random.choice(np.arange(train_triples.shape[0]), num_sess_triples, False)
            self.sess_triples.append(train_triples[sess_triple_idx, :])
            train_triples = np.delete(train_triples, sess_triple_idx, axis=0)

    def get_sess_stats(self):
        seen_ents = set()
        seen_rels = set()
        seen_tr_triples = set()
        seen_de_triples = set()
        seen_te_triples = set()
        stats = np.ndarray(shape=(20, self.ms))

        for sess_idx in range(self.ms):
            # get data
            sess_tr = self.sess_triples[sess_idx]
            sess_ents = self.get_ents(sess_tr)
            sess_rels = self.get_rels(sess_tr)
            sess_de = self.filter_triples(sess_ents, sess_rels, self.sros_de)
            sess_te = self.filter_triples(sess_ents, sess_rels, self.sros_te)
            # sess_de = self.filter_triples(seen_ents.union(sess_ents), seen_rels.union(sess_rels), self.sros_de)
            # sess_te = self.filter_triples(seen_ents.union(sess_ents), seen_rels.union(sess_rels), self.sros_te)
            # get stats
            sess_tr_triples = set()
            sess_de_triples = set()
            sess_te_triples = set()
            for triple_idx in range(sess_tr.shape[0]):
                triple = tuple(sess_tr[triple_idx, :].astype(int).tolist())
                sess_tr_triples = sess_tr_triples.union({triple})
            for triple_idx in range(sess_de.shape[0]):
                triple = tuple(sess_de[triple_idx, :].astype(int).tolist())
                sess_de_triples = sess_de_triples.union({triple})
            for triple_idx in range(sess_te.shape[0]):
                triple = tuple(sess_te[triple_idx, :].astype(int).tolist())
                sess_te_triples = sess_te_triples.union({triple})

            new_tr_triple_rate = 100.0 * float(len(sess_tr_triples.difference(seen_tr_triples))) / float(len(sess_tr_triples))
            new_de_triple_rate = 100.0 * float(len(sess_de_triples.difference(seen_de_triples))) / float(len(sess_de_triples))
            new_te_triple_rate = 100.0 * float(len(sess_te_triples.difference(seen_te_triples))) / float(len(sess_te_triples))
            new_ent_rate = 100.0 * float(len(sess_ents.difference(seen_ents))) / float(len(sess_ents))
            new_rel_rate = 100.0 * float(len(sess_rels.difference(seen_rels))) / float(len(sess_rels))

            seen_tr_triples = seen_tr_triples.union(sess_tr_triples)
            seen_de_triples = seen_de_triples.union(sess_de_triples)
            seen_te_triples = seen_te_triples.union(sess_te_triples)
            seen_ents = seen_ents.union(sess_ents)
            seen_rels = seen_rels.union(sess_rels)
            sess_num_tr_triples = str(len(sess_tr_triples))
            accu_num_tr_triples = str(len(seen_tr_triples))
            sess_num_de_triples = str(len(sess_de_triples))
            accu_num_de_triples = str(len(seen_de_triples))
            sess_num_te_triples = str(len(sess_te_triples))
            accu_num_te_triples = str(len(seen_te_triples))
            sess_num_ents = str(len(sess_ents))
            accu_num_ents = str(len(seen_ents))
            sess_num_rels = str(len(sess_rels))
            accu_num_rels = str(len(seen_rels))
            tr_triple_coverage = 100.0 * len(seen_tr_triples) / self.sros_tr.shape[0]
            de_triple_coverage = 100.0 * len(seen_de_triples) / self.sros_de.shape[0]
            te_triple_coverage = 100.0 * len(seen_te_triples) / self.sros_te.shape[0]
            ent_coverage = 100.0 * len(seen_ents) / len(self.e2i)
            rel_coverage = 100.0 * len(seen_rels) / len(self.r2i)

            # train triples
            logout("There are " + str(len(sess_num_tr_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[0, sess_idx] = sess_num_tr_triples
            logout("There are " + "{:.2f}".format(new_tr_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[1, sess_idx] = "{:.2f}".format(new_tr_triple_rate)
            logout("There are " + accu_num_tr_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[2, sess_idx] = accu_num_tr_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(tr_triple_coverage), "i")
            stats[3, sess_idx] = "{:.2f}".format(tr_triple_coverage)
            # dev triples
            logout("There are " + str(len(sess_num_de_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[4, sess_idx] = sess_num_de_triples
            logout("There are " + "{:.2f}".format(new_de_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[5, sess_idx] = "{:.2f}".format(new_de_triple_rate)
            logout("There are " + accu_num_de_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[6, sess_idx] = accu_num_de_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(de_triple_coverage), "i")
            stats[7, sess_idx] = "{:.2f}".format(de_triple_coverage)
            # test triples
            logout("There are " + str(len(sess_num_te_triples)) + " train triples for session " + str(sess_idx) + ".", "i")
            stats[8, sess_idx] = sess_num_te_triples
            logout("There are " + "{:.2f}".format(new_te_triple_rate) + "% new triples for session " + str(sess_idx) + ".", "i")
            stats[9, sess_idx] = "{:.2f}".format(new_te_triple_rate)
            logout("There are " + accu_num_te_triples + " accumulated triples after session " + str(sess_idx) + ".", "i")
            stats[10, sess_idx] = accu_num_te_triples
            logout("Triple coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(te_triple_coverage), "i")
            stats[11, sess_idx] = "{:.2f}".format(te_triple_coverage)
            # ents
            logout("There are " + sess_num_ents + " entities for session " + str(sess_idx) + ".", "i")
            stats[12, sess_idx] = sess_num_ents
            logout("There are " + "{:.2f}".format(new_ent_rate) + "% new ents for session " + str(sess_idx) + ".", "i")
            stats[13, sess_idx] = "{:.2f}".format(new_ent_rate)
            logout("There are " + accu_num_ents + " accumulated entities after session " + str(sess_idx) + ".", "i")
            stats[14, sess_idx] = accu_num_ents
            logout("Ent coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(ent_coverage), "i")
            stats[15, sess_idx] = "{:.2f}".format(ent_coverage)
            # relations
            logout("There are " + sess_num_rels + " relations for session " + str(sess_idx) + ".", "i")
            stats[16, sess_idx] = sess_num_rels
            logout("There are " + "{:.2f}".format(new_rel_rate) + "% new rels for session " + str(sess_idx) + ".", "i")
            stats[17, sess_idx] = "{:.2f}".format(new_rel_rate)
            logout("There are " + accu_num_rels + " accumulated relations after session " + str(sess_idx) + ".", "i")
            stats[18, sess_idx] = accu_num_rels
            logout("Rel coverage for session " + str(sess_idx) + ": " + "{:.2f}".format(rel_coverage), "i")
            stats[19, sess_idx] = "{:.2f}".format(rel_coverage)

        row_labels = ["Num Train Triples",
                      "% New Train Triples",
                      "Num Accu. Train Triples",
                      "Train Triple Coverage",
                      "Num Dev Triples",
                      "% New Dev Triples",
                      "Num Accu. Dev Triples",
                      "Dev Triple Coverage",
                      "Num Test Triples",
                      "% New Test Triples",
                      "Num Accu. Test Triples",
                      "Test Triple Coverage",
                      "Num Ents",
                      "% New Ents",
                      "Num Accu. Ents",
                      "Ent Coverage",
                      "Num Rels",
                      "% New Rels",
                      "Num Accu. Rels",
                      "Rel Coverage"]
        col_labels = ["S" + str(i+1) for i in range(self.ms)]

        fig = plt.figure(figsize=(10, 6))
        axs = fig.add_subplot(1, 1, 1)
        fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        plt.grid('off')
        the_table = axs.table(cellText=stats, rowLabels=row_labels, colLabels=col_labels, loc='center')
        fig.tight_layout()
        plt.savefig(str(self.dataset_name + "_" + str(self.ms) + "_TS.pdf"), bbox_inches="tight")

    def filter_triples(self, fitler_ents, filter_rels, triples):
        filtered_triples = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(len(triples)):
            h, r, t = triples[idx]
            if h not in fitler_ents or t not in fitler_ents or r not in filter_rels:
                continue
            else:
                filtered_triples = np.append(filtered_triples, [triples[idx]], axis=0)
        return filtered_triples

    def get_rels(self, triples):
        rels = set()
        for triple in triples:
            rels.add(triple[1])
        return rels

    def get_ents(self, triples):
        ents = set()
        for triple in triples:
            ents.add(triple[0])
            ents.add(triple[2])
        return ents

    def save(self, dkge=False):
        known_ents = set()
        known_rels = set()
        for sess_idx in range(self.ms):
            sess_tr = self.sess_triples[sess_idx]
            sess_ents = self.get_ents(sess_tr)
            sess_rels = self.get_rels(sess_tr)
            sess_de = self.filter_triples(sess_ents, sess_rels, self.sros_de)
            sess_te = self.filter_triples(sess_ents, sess_rels, self.sros_te)
            known_ents = known_ents.union(set(copy(sess_ents)))
            known_rels = known_rels.union(set(copy(sess_rels)))
            self.write_triples_to_file(self.fps[sess_idx] + "train2id.txt", sess_tr)
            self.write_triples_to_file(self.fps[sess_idx] + "valid2id.txt", sess_de)
            self.write_triples_to_file(self.fps[sess_idx] + "test2id.txt", sess_te)
            copyfile(self.prev_fp + "/relation2id.txt", self.fps[sess_idx] + "relation2id.txt")
            copyfile(self.prev_fp + "/entity2id.txt", self.fps[sess_idx] + "entity2id.txt")
            self.write_known_to_file(self.fps[sess_idx] + "known_ents.txt", known_ents, self.i2e)
            self.write_known_to_file(self.fps[sess_idx] + "known_rels.txt", known_rels, self.i2r)
            self.write_sess_to_file(self.fps[sess_idx] + "sess_ents.txt", sess_ents, self.i2e)
            self.write_sess_to_file(self.fps[sess_idx] + "sess_rels.txt", sess_rels, self.i2r)
            logout("Dataset session " + str(sess_idx) + " written to file", "s")
            if dkge:
                self.save_dkge_dataset(sess_tr, sess_de, sess_te, sess_ents, sess_rels, self.fps[sess_idx])
            logout("DKGE Dataset session " + str(sess_idx) + " written to file", "s")

    def write_triples_to_file(self, filename, triples):
        with open(filename, "w") as f:
            f.write(str(triples.shape[0]) + "\n")
            for triple in triples:
                f.write(str(triple[0]) + " " + str(triple[2]) + " " + str(triple[1]) + "\n")

    def write_known_to_file(self, filename, ids, lookup):
        with open(filename, "w") as f:
            for id in ids:
                f.write(str(lookup[id]) + "\n")

    def write_sess_to_file(self, filename, ids, lookup):
        with open(filename, "w") as f:
            f.write(str(len(ids)) + "\n")
            for id in ids:
                f.write(str(lookup[id]) + "\t" + str(id) + "\n")

    def save_dkge_dataset(self, trtr, detr, tetr, ents, rels, fp):
        # remaps the ids of entities and relations to be compatible with the dkge code
        e2i_ = {}
        i2e_ = {}
        r2i_ = {}
        i2r_ = {}
        ents = list(ents)
        rels = list(rels)

        for i in range(len(ents)):
            e2i_[self.i2e[ents[i]]] = i
            i2e_[i] = self.i2e[ents[i]]

        for i in range(len(rels)):
            r2i_[self.i2r[rels[i]]] = i
            i2r_[i] = self.i2r[rels[i]]

        trtr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(trtr.shape[0]):
            h,r,t = trtr[idx]
            try:
                tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            except:
                pdb.set_trace()
            trtr_ = np.append(trtr_, [tr], axis=0)

        detr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(detr.shape[0]):
            h, r, t = detr[idx]
            tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            detr_ = np.append(detr_, [tr], axis=0)

        tetr_ = np.ndarray(shape=(0, 3), dtype=np.int32)
        for idx in range(tetr.shape[0]):
            h, r, t = tetr[idx]
            tr = [e2i_[self.i2e[h]], r2i_[self.i2r[r]], e2i_[self.i2e[t]]]
            tetr_ = np.append(tetr_, [tr], axis=0)

        fp += "dkge/"
        if not exists(fp):
            makedirs(fp)

        self.write_triples_to_file(fp + "train2id.txt", trtr_)
        self.write_triples_to_file(fp + "valid2id.txt", detr_)
        self.write_triples_to_file(fp + "test2id.txt", tetr_)
        self.write_sess_to_file(fp + "entity2id.txt", sorted(list(e2i_.values())), i2e_)
        self.write_sess_to_file(fp + "relation2id.txt", sorted(list(r2i_.values())), i2r_)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generates benchmark continual-learning knowledge graph embedding dataset")
    parser.add_argument("-d", dest="dataset", type=str, help="Dataset name (WN18, FB15K, etc.)")
    parser.add_argument("-r", dest="ent_sample_rate", type=float, help="Sampling rate for entities")
    parser.add_argument("-n", dest="num_sess", type=int, help="Number of learning sessions")
    parser.add_argument("-s", dest="sampling_type", type=str, help="ent or triple")
    parser.add_argument("-y", dest="dkge", type=bool, help="Whether to generate the DKGE dataset")
    args = parser.parse_args()
    # selects the sampling strategy to use (i.e. entity or triple sampling)
    if args.sampling_type == "ent":
        generator = CLDatasetEntitySampler(args.dataset, args.ent_sample_rate, args.num_sess)
    elif args.sampling_type == "rel":
        generator = CLDatasetRelationSampler(args.dataset, args.ent_sample_rate, args.num_sess)
    elif args.sampling_type == "triple":
        generator = CLDatasetTripleSampler(args.dataset, args.ent_sample_rate, args.num_sess)
    else:
        logout("Sampling strategy not recognized. Aborted.", "f")
        exit()
    # generates and saves the datasets along with dataset statistics
    generator.generate_splits()
    generator.get_sess_stats()
    generator.save(args.dkge)
