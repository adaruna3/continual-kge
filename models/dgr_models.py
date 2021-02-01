from copy import copy
import numpy as np

import torch
import torch.nn as nn


import pdb


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, num_embeddings, device):
        super(GRUEncoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.device = device
        # Architecture: embedding -> GRU -> hidden -> linear -> latent
        self.embeddings = nn.Embedding(num_embeddings, input_dim)
        self.gru_enc = nn.GRU(input_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.hidden2mu = nn.Linear(hidden_dim, z_dim)
        self.hidden2var = nn.Linear(hidden_dim, z_dim)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if len(p.shape) >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, a=-1.0/self.hidden_dim, b=1.0/self.hidden_dim)

    def forward(self, input_sequences):
        input_embedding = self.embeddings(input_sequences)
        _, hidden = self.gru_enc(input_embedding)
        hidden = hidden.squeeze()
        # get mean / var for latent
        z_mu = self.hidden2mu(hidden)
        z_var = self.hidden2var(hidden)
        return z_mu, z_var, input_embedding


class GRUDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, num_embeddings, device):
        super(GRUDecoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        self.device = device
        # Architecture: latent -> linear -> hidden -> GRU -> linear -> vocab
        self.latent2hidden = nn.Linear(z_dim, hidden_dim)
        self.gru_dec = nn.GRU(output_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.output2logits = nn.Linear(hidden_dim, num_embeddings)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if len(p.shape) >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, a=-1.0 / self.hidden_dim, b=1.0 / self.hidden_dim)

    def forward(self, latent, input_embeddings):
        # take the latent embedding and put through fully connected to hidden
        hidden = self.latent2hidden(latent)
        if len(hidden.shape) < 2:
            # handles cases where batch is of size 1 and 2D vector collapses
            hidden = hidden.unsqueeze(0).unsqueeze(0)
        else:
            hidden = hidden.unsqueeze(0)
        outputs, _ = self.gru_dec(input_embeddings, hidden)
        logits = torch.nn.functional.log_softmax(self.output2logits(outputs.reshape(-1, outputs.size(2))), dim=1)
        return logits.view(-1, 4, self.num_embeddings)


class TripleGRUVAE(nn.Module):
    def __init__(self, encoder, decoder, sot_idx, eot_idx, device,
                 anneal_slope=0.06, anneal_position=233.0, anneal_max=0.8):
        super(TripleGRUVAE, self).__init__()
        self.encoder = encoder
        self.z_dim = copy(encoder.z_dim)
        self.decoder = decoder
        self.sot_idx = sot_idx
        self.eot_idx = eot_idx
        self.device = device
        self.anneal_slope = anneal_slope
        self.anneal_position = anneal_position
        self.anneal_max = anneal_max
        self.anneal_step = 0.0
        self.nll_loss = torch.nn.NLLLoss(reduction="none")

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, input_sequences, target_sequences):
        z_mu, z_var, input_embeddings = self.encoder(input_sequences)
        x_sample = self.reparameterize(z_mu, z_var)
        predicted = self.decoder(x_sample, input_embeddings)
        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var, dim=1)
        kl_loss = kl_loss.unsqueeze(-1)
        kl_weight = self.compute_anneal()
        # reconstruction loss
        rc_loss = self.nll_loss(predicted.view(-1, predicted.size(2)),
                                target_sequences.view(-1)).view(-1, 4)
        # complete loss combines kl and rc losses
        loss = torch.mean(torch.sum(torch.cat((rc_loss, kl_loss * kl_weight), dim=1), dim=1))
        return loss, rc_loss.cpu().data.numpy(), kl_loss.cpu().data.numpy()

    def compute_anneal(self):
        return float(self.anneal_max / (1 + np.exp(-self.anneal_slope * (self.anneal_step - self.anneal_position))))

    def step_anneal(self):
        self.anneal_step += 1.0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, n=1):
        output_sequences = torch.empty(n, 4).to(self.device)
        z = torch.randn((n, self.z_dim)).to(self.device)
        input_sequences = torch.tensor(()).to(self.device)
        input_sequences = input_sequences.new_full((n,), self.sot_idx, dtype=torch.long)
        hidden = self.decoder.latent2hidden(z)
        hidden = hidden.unsqueeze(0)
        for t in range(4):
            input_sequences = input_sequences.unsqueeze(1)
            input_embedding = self.encoder.embeddings(input_sequences)
            output, hidden = self.decoder.gru_dec(input_embedding, hidden)
            logits = self.decoder.output2logits(output)
            _, output_token = torch.topk(logits, 1, dim=-1)
            input_sequences = output_token.squeeze()
            output_sequences[:, t] = output_token.squeeze()
        return output_sequences.cpu().data.numpy()


class Analogy(nn.Module):
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

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

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
        return self.loss(score, batch_y)

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


class TransE(nn.Module):
    def __init__(self, num_ents, num_rels, hidden_size, margin, neg_ratio, batch_size, device):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(num_ents, hidden_size).to(device)
        self.rel_embeddings = nn.Embedding(num_rels, hidden_size).to(device)
        self.criterion = nn.MarginRankingLoss(margin, reduction="sum").to(device)
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.device = device
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, r, t):
        h = nn.functional.normalize(h, 2, -1)
        r = nn.functional.normalize(r, 2, -1)
        t = nn.functional.normalize(t, 2, -1)
        return torch.norm(h + r - t, 1, -1)

    def loss(self, p_score, n_score):
        y = torch.autograd.Variable(torch.Tensor([-1])).to(self.device)
        return self.criterion(p_score, n_score, y)

    def forward(self, batch_h, batch_r, batch_t, batch_y):
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        score = self._calc(h, r, t)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

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
