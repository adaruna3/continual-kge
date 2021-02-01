import torch
import torch.nn as nn
from torch.autograd import Variable

from logger.terminal_utils import logout


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def update_og_params(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')
                # update the stored values
                self.register_buffer('{}_l2_og_params'.format(n), p.detach().clone())

    def l2_regularization(self):
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '_')
                    p_prev = getattr(self, '{}_l2_og_params'.format(n))
                    if "rel" in n:
                        losses.append(((p - p_prev)[self.prev_rels] ** 2).sum())
                    elif "ent" in n:
                        losses.append(((p - p_prev)[self.prev_ents] ** 2).sum())
                    else:
                        logout("Unknown model params", "f")
                        exit()
            return sum(losses)
        except AttributeError:
            # default L2 loss when no prior task
            return torch.tensor(0., device=self.device)


class Analogy(L2):
    def __init__(self, num_ents, num_rels, hidden_size, device):
        super(Analogy, self).__init__()
        self.ent_re_embeddings = nn.Embedding(num_ents, int(hidden_size / 2.0)).to(device)
        self.ent_im_embeddings = nn.Embedding(num_ents, int(hidden_size / 2.0)).to(device)
        self.rel_re_embeddings = nn.Embedding(num_rels, int(hidden_size / 2.0)).to(device)
        self.rel_im_embeddings = nn.Embedding(num_rels, int(hidden_size / 2.0)).to(device)
        self.ent_embeddings = nn.Embedding(num_ents, int(hidden_size / 2.0)).to(device)
        self.rel_embeddings = nn.Embedding(num_rels, int(hidden_size / 2.0)).to(device)
        self.criterion = nn.Sigmoid().to(device)
        self.device = device
        self.init_weights()

        # L2 regularization related
        self.l2_task_weight = Variable(torch.Tensor([1.0])).to(self.device)
        self.prev_ents = []
        self.prev_rels = []

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def set_regularize_ents_rels(self, prev_ents, prev_rels):
        self.prev_ents = torch.tensor(prev_ents, dtype=torch.long)
        self.prev_rels = torch.tensor(prev_rels, dtype=torch.long)

    def set_task_weight(self, weight):
        self.l2_task_weight = Variable(torch.Tensor([weight])).to(self.device)

    def _calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
        return torch.sum(r_re * h_re * t_re + r_re * h_im * t_im + r_im * h_re * t_im - r_im * h_im * t_re, -1) + \
               torch.sum(h * t * r, -1)

    def loss(self, score, batch_y):
        return torch.sum(-torch.log(self.criterion(score * batch_y.float())))

    def forward(self, batch_h, batch_r, batch_t, batch_y):
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        h = self.ent_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        t = self.ent_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
        return self.loss(score, batch_y) + self.l2_task_weight * self.l2_regularization()

    def predict(self, batch_h, batch_r, batch_t):
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        h = self.ent_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        t = self.ent_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
        return -score.cpu().data.numpy()


class TransE(L2):
    def __init__(self, num_ents, num_rels, hidden_size, margin, neg_ratio, batch_size, device):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(num_ents, hidden_size).to(device)
        self.rel_embeddings = nn.Embedding(num_rels, hidden_size).to(device)
        self.criterion = nn.MarginRankingLoss(margin, reduction="sum").to(device)
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.device = device
        self.init_weights()

        # L2 regularization related
        self.l2_task_weight = Variable(torch.Tensor([1.0])).to(self.device)
        self.prev_ents = []
        self.prev_rels = []

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def set_regularize_ents_rels(self, prev_ents, prev_rels):
        self.prev_ents = torch.tensor(prev_ents, dtype=torch.long)
        self.prev_rels = torch.tensor(prev_rels, dtype=torch.long)

    def set_task_weight(self, weight):
        self.l2_task_weight = Variable(torch.Tensor([weight])).to(self.device)

    def _calc(self, h, r, t):
        h = nn.functional.normalize(h, 2, -1)
        r = nn.functional.normalize(r, 2, -1)
        t = nn.functional.normalize(t, 2, -1)
        return torch.norm(h + r - t, 1, -1)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1])).to(self.device)
        return self.criterion(p_score, n_score, y)

    def forward(self, batch_h, batch_r, batch_t, batch_y):
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        score = self._calc(h, r, t)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score) + self.l2_task_weight * self.l2_regularization()

    def predict(self, batch_h, batch_r, batch_t):
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        score = self._calc(h, r, t)
        return score.cpu().data.numpy()

    def get_positive_score(self, score):
        return score[0:len(score):self.neg_ratio+1]

    def get_negative_score(self, score):
        negs = torch.tensor([], dtype=torch.float32).to(self.device)
        for idx in range(0, len(score), self.neg_ratio + 1):
            batch_negs = score[idx + 1:idx + self.neg_ratio + 1]
            negs = torch.cat((negs, torch.mean(batch_negs,0,keepdim=True)))
        return negs