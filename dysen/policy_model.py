from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch
from dysen.metrics import action_planning_reliability, triplet_recall
from scripts.fvd_utils.fvd_utils import get_fvd_logits, frechet_distance, load_fvd_model, polynomial_mmd
import numpy as np
from tqdm import tqdm
import time
import math


class PolicyNetwork(nn.Module):

    def __init__(self,
                 model_config='openai/clip-vit-base-patch32',
                 add_linear=False,
                 in_dim=512,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()

        assert freeze_encoder
        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(in_dim, embedding_size)
            # self.linear = nn.Sequential(nn.Linear(in_dim, embedding_size * 2), nn.ReLU(),
            #                             nn.Linear(embedding_size * 2, embedding_size))
        else:
            self.linear = None

    def forward(self, emb_inp):
        sentence_embedding = emb_inp

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding


class Reward(nn.Module):
    def __init__(self, n_sample, batch_size, args, device='cuda'):
        super().__init__()
        self.i3d = load_fvd_model(device)
        self.model.eval()
        self.device = device
        self.args = args
        self.n_sample = n_sample
        self.batch_size = batch_size

    @torch.no_grad()
    def forward(self, video_pred, video_gt, dsg_pred, dsg_gt, action_pred, action_gt):

        # reward of video quality
        s = time.time()
        gold_embeddings = []
        n_batch = video_gt.shape[0] // self.batch_size
        for i in tqdm(range(n_batch), desc="Extract Fake Embedding"):
            gold_embeddings.append(
                get_fvd_logits(video_gt[i * self.batch_size:(i + 1) * self.batch_size], i3d=self.i3d,
                               device=self.device,
                               batch_size=self.batch_size))
        gold_embeddings = torch.cat(gold_embeddings, 0)[:self.n_sample]
        t = time.time() - s
        print(f'cost time for extract gold video embedding: {t}')
        s = time.time()
        pred_embeddings = []
        n_batch = video_pred.shape[0] // self.batch_size
        for i in tqdm(range(n_batch), desc="Extract Fake Embedding"):
            pred_embeddings.append(
                get_fvd_logits(video_pred[i * self.batch_size:(i + 1) * self.batch_size], i3d=self.i3d, device=self.device,
                               batch_size=self.batch_size))
        pred_embeddings = torch.cat(pred_embeddings, 0)[:self.n_sample]
        t = time.time() - s
        print(f'cost time for extract pred video embedding: {t}')

        print('calculate fvd ...')
        fvd = frechet_distance(pred_embeddings, gold_embeddings)
        print(f'fvd: {fvd}')

        # reward of action planning reliability
        apr = action_planning_reliability(action_gt, action_pred)
        print(f'action planing reliability: {apr}')

        # reward of imagination rationality
        ir = triplet_recall(dsg_gt, dsg_pred)
        print(f'imagination rationality with TriRec.: {ir}')

        # reward = clip_reward + aes_reward * 0.1 + miou * 10
        reward = 1/fvd + apr + ir

        norm_reward = (reward - reward.mean()) / (reward.std() + 1e-6)
        return norm_reward, reward