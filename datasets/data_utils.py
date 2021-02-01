from os.path import abspath, dirname, exists
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from logger.terminal_utils import logout

import pdb


class TripleDataset(Dataset):
    def __init__(self, dataset_name, neg_ratio=0):
        """
        Represents a triples dataset
        :param dataset_name: dataset folder name
        """
        super(TripleDataset, self).__init__()
        datasets_fp = abspath(dirname(__file__)) + "/"
        self.fp = datasets_fp + dataset_name + "/"
        self.neg_ratio = neg_ratio
        self.e2i, self.i2e = self.load_id_map("entity2id.txt")
        self.r2i, self.i2r = self.load_id_map("relation2id.txt")
        self.triple_ents = []
        self.triple_rels = []
        self.known_ents = []
        self.known_rels = []
        self.triples = None
        self.berns = None
        self.h_mask = {}
        self.t_mask = {}
        self.counts = None

    def load_id_map(self, label_file):
        """
        loads a mapping between triples/strings and IDs
        :param label_file: filename of labels
        :return: ID mapping(s) for the set of labels in a file
        """
        try:
            labels = pd.read_csv(self.fp + label_file, sep="\t", skiprows=1, header=None,
                                 dtype={0: np.str, 1: np.int32})
        except IOError as e:
            logout("Could not load " + str(label_file), "f")
            raise IOError

        label2index = {labels.iloc[idx, 0]: labels.iloc[idx, 1] for idx in range(len(labels))}
        index2label = {labels.iloc[idx, 1]: labels.iloc[idx, 0] for idx in range(len(labels))}
        return label2index, index2label

    def load_triple_set(self, names):
        """
        Loads the dataset object with triples in set `name` of the dataset
        :param name: `name` of the set to load (i.e. train2id, test2id, valid2id)
        :return: None
        """
        if type(names) == str:
            names = [names]
        self.triples = self.load_triples([name + ".txt" for name in names])
        self.load_bernouli_sampling_stats()

    def load_triples(self, triples_files):
        """
        loads all triples in the triples file
        :param triples_file: contains triples for train, valid, or test
        :return:
        """
        triples = np.ndarray(shape=(0, 3), dtype=int)
        for triples_file in triples_files:
            try:
                file_triples = pd.read_csv(self.fp + triples_file, sep=" |,", skiprows=1, header=None,
                                     dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
                file_triples[:, [1, 2]] = file_triples[:, [2, 1]]
                triples = np.append(triples, file_triples, axis=0)
            except IOError as e:
                logout('Could not load ' + str(triples_file), "f")
                raise IOError
        return triples

    def load_known_ent_set(self):
        """
        loads the known ents array used during negative sampling and regularization
        :return:
        """
        known_ents_file = self.fp + "known_ents.txt"
        if exists(known_ents_file):
            with open(known_ents_file, "r") as f:
                for line in f:
                    ent = line.strip()
                    self.known_ents.append(self.e2i[ent])
        else:
            self.known_ents = list(self.e2i.values())
        self.known_ents.sort()

    def load_known_rel_set(self):
        """
        loads the known rels array used for regularization
        unknown entities
        :return:
        """
        known_rels_file = self.fp + "known_rels.txt"
        if exists(known_rels_file):
            with open(known_rels_file, "r") as f:
                for line in f:
                    rel = line.strip()
                    self.known_rels.append(self.r2i[rel])
        else:
            self.known_rels = list(self.r2i.values())
        self.known_rels.sort()

    def load_current_ents_rels(self):
        for triple in self.triples:
            h, r, t = triple.tolist()
            if h not in self.triple_ents:
                self.triple_ents.append(int(h))
            if t not in self.triple_ents:
                self.triple_ents.append(int(t))
            if r not in self.triple_rels:
                self.triple_rels.append(int(r))
        self.triple_ents.sort()
        self.triple_rels.sort()

    def load_bernouli_sampling_stats(self):
        """
        calculates probabilities needed to do negative sampling based on Bernoulli method
        :return:
        """
        probs = {}
        for rel in self.r2i.values():
            hpt = {}
            tph = {}
            for idx in range(len(self.triples)):
                h, r, t = self.triples[idx, :].tolist()
                if r == rel:
                    if h not in tph:
                        tph[h] = {t}
                    else:
                        tph[h].add(t)
                    if t not in hpt:
                        hpt[t] = {h}
                    else:
                        hpt[t].add(h)
            if len(tph) > 0 and len(hpt) > 0:
                avg_tph = np.average([float(len(tph[h])) for h in tph])
                avg_hpt = np.average([float(len(hpt[t])) for t in hpt])
                probs[rel] = avg_tph / (avg_tph + avg_hpt)
            else:
                probs[rel] = 0.0
        self.berns = probs

    def __len__(self):
        """
        Used by dataloader, returns set size
        :return: triples set size
        """
        return len(self.triples)

    def __getitem__(self, idx):
        """
        :param idx: index of triple to return
        :return: training triples sample
        """
        samples = np.asarray([self.triples[idx, :].tolist()+[1]])
        samples = np.concatenate([samples, self.corrupt(self.triples[idx, :], self.neg_ratio)])
        return samples

    def corrupt(self, triple, num):
        """
        uses Bernoulli method to make corrupted triples
        :param triple: triple used for generating negative samples
        :param num: number of negative samples
        :return: np.ndarray of negative samples
        """
        h, r, t = triple.tolist()
        corrupted_triples = np.ndarray(shape=(0, 4), dtype=np.int32)
        try:
            prob = self.berns[r]
        except KeyError as e: # for dealing with UNK relations...
            prob = 0.5
        for i in range(num):
            if np.random.uniform() < prob:
                hh = self.known_ents[np.random.randint(len(self.known_ents), dtype=np.int32)]
                corrupted_triples = np.append(corrupted_triples, [[hh, r, t, -1]], axis=0)
            else:
                tt = self.known_ents[np.random.randint(len(self.known_ents), dtype=np.int32)]
                corrupted_triples = np.append(corrupted_triples, [[h, r, tt, -1]], axis=0)
        return corrupted_triples

    def load_mask(self, dataset_fps=None):
        """
        loads the hr -> o & rt -> h vocab used for "filtering" during evaluation
        """
        t_mask = {}
        h_mask = {}
        all_triples = np.ndarray(shape=(0, 3))

        if dataset_fps is None:
            dataset_fps = [self.fp]
        else:
            dataset_fps += [self.fp]
        dataset_fps = list(set(dataset_fps))

        # loads all train, valid, and test triples
        triple_file_names = ["train2id", "valid2id", "test2id"]
        for dataset_fp in dataset_fps:
            for filename in triple_file_names:
                triples_file = dataset_fp + filename + ".txt"
                try:
                    new_triples = pd.read_csv(triples_file, sep=" |,", skiprows=1, header=None,
                                         dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
                    new_triples[:, [1, 2]] = new_triples[:, [2, 1]]
                    all_triples = np.append(all_triples, new_triples, axis=0)
                except IOError as e:
                    logout('Could not load ' + str(triples_file), "f")
                    exit()
        all_triples = np.unique(all_triples, axis=0)

        # sets the hr -> t & rt -> h vocabs
        for triple in all_triples:
            h, r, t = triple
            if (r, t) in h_mask:
                if h not in h_mask[(r, t)]:
                    h_mask[(r, t)].append(h)
            else:
                h_mask[(r, t)] = [h]

            if (h, r) in t_mask:
                if t not in t_mask[(h, r)]:
                    t_mask[(h, r)].append(t)
            else:
                t_mask[(h, r)] = [t]

        self.h_mask = h_mask
        self.t_mask = t_mask

    def load_counts(self, ground_truth_file, filtering_file=None):
        # loads the ground truth triples from the full dataset
        gt_triples = pd.read_csv(self.fp + ground_truth_file, sep=" |,", skiprows=1, header=None,
                                 dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
        gt_triples[:, [1, 2]] = gt_triples[:, [2, 1]]

        # populates the counts matrix
        self.counts = np.zeros(shape=(len(self.r2i), len(self.e2i), len(self.e2i)), dtype=np.int64)
        for idx in range(gt_triples.shape[0]):
            h, r, t = gt_triples[idx, :]
            self.counts[r, h, t] += 1.0

        if filtering_file is not None:  # TODO consider further what SHOULD be filtered...
            # loads the train triples from the full dataset
            train_triples = pd.read_csv(self.fp + filtering_file, sep=" |,", skiprows=1, header=None,
                                        dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
            train_triples[:, [1, 2]] = train_triples[:, [2, 1]]

            # removes training triples from counts matrix
            for idx in range(train_triples.shape[0]):
                h, r, t = train_triples[idx, :]
                self.counts[r, h, t] = 0.0

    def predict(self, h, r, t):
        return -self.counts[r.cpu().data.numpy(), h.cpu().data.numpy(), t.cpu().data.numpy()]

    def substitute_unks(self):
        # sets the UNK id
        ent_unk_id = len(self.e2i)
        rel_unk_id = len(self.r2i)

        # loads the UNK entities and relations
        unk_ents = set()
        unk_rels = set()
        with open(self.fp+"UNK_ents.txt","r") as f:
            next(f)
            for line in f:
                ent, id = line.strip().split("\t")
                unk_ents.add(int(id))
        with open(self.fp+"UNK_rels.txt","r") as f:
            next(f)
            for line in f:
                rel, id = line.strip().split("\t")
                unk_rels.add(int(id))

        # substitutes UNKs into triples
        for idx in range(self.triples.shape[0]):
            h,r,t = self.triples[idx,:]
            self.triples[idx, 0] = ent_unk_id if h in unk_ents else h
            self.triples[idx, 2] = ent_unk_id if t in unk_ents else t
            self.triples[idx, 1] = rel_unk_id if r in unk_rels else r


class TripleSequenceDataset(TripleDataset):
    def __init__(self, dataset_name):
        super(TripleSequenceDataset, self).__init__(dataset_name)
        self.vocab, self.w2i, self.i2w, self.sot, self.eot = self.load_vocab_map()

    def load_vocab_map(self):
        vocab = list(self.e2i.keys()) + list(self.r2i.keys()) + ["<sot>", "<eot>"]
        w2i = {vocab[idx]: idx for idx in range(len(vocab))}
        i2w = {idx: vocab[idx] for idx in range(len(vocab))}
        sot = w2i["<sot>"]
        eot = w2i["<eot>"]
        return vocab, w2i, i2w, sot, eot

    def load_triple_set(self, names):
        if type(names) == str:
            names = [names]
        self.load_triples([name + ".txt" for name in names])
        self.load_bernouli_sampling_stats()

    def load_triples(self, triples_file):
        self.triples = super().load_triples(triples_file)
        triples = np.zeros_like(self.triples, dtype=int)
        for row in range(self.triples.shape[0]):
            triples[row, 0] = self.w2i[self.i2r[self.triples[row, 1]]]
            triples[row, 1] = self.w2i[self.i2e[self.triples[row, 0]]]
            triples[row, 2] = self.w2i[self.i2e[self.triples[row, 2]]]
        self.triples = triples

    def load_bernouli_sampling_stats(self):
        probs = {}
        for rel in [self.w2i[rel] for rel in self.r2i.keys()]:
            hpt = {}
            tph = {}
            for idx in range(len(self.triples)):
                r, h, t = self.triples[idx, :].tolist()
                if r == rel:
                    if h not in tph:
                        tph[h] = {t}
                    else:
                        tph[h].add(t)
                    if t not in hpt:
                        hpt[t] = {h}
                    else:
                        hpt[t].add(h)
            if len(tph) > 0 and len(hpt) > 0:
                avg_tph = np.average([float(len(tph[h])) for h in tph])
                avg_hpt = np.average([float(len(hpt[t])) for t in hpt])
                probs[rel] = avg_tph / (avg_tph + avg_hpt)
            else:
                probs[rel] = 0.0
        self.berns = probs

    def __getitem__(self, idx):
        return {"input": np.concatenate(([self.w2i["<sot>"]], self.triples[idx, :])),
                "target": np.concatenate((self.triples[idx, :], [self.w2i["<eot>"]]))}

    def corrupt(self, triple, num):
        raise NotImplementedError

    def load_mask(self, dataset_fps=None):
        raise NotImplementedError


if __name__ == "__main__":
    # TODO add unit tests
    pass
