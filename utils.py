import numpy as np

import torch
from torch.utils.data import Dataset
import random

def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


class SelCaDataset(Dataset):
    def __init__(self, train, num_neg_samples=3):
        self.users = train.sequences.user_ids
        self.sequences = train.sequences.sequences
        self.probs = train.sequences.probs
        self.targets = train.sequences.targets

        self.all_items = np.arange(train.num_items - 1) + 1  # 0 for padding
        interaction_matrix = train.tocsr()
        self.candidates = [r.indices for r in interaction_matrix]

        self.num_neg_samples = num_neg_samples
        self.num_items = len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        sequence = torch.LongTensor(self.sequences[index])
        prob = torch.FloatTensor(self.probs[index])
        target = torch.LongTensor(self.targets[index])

        # negative samples
        candidates = np.setdiff1d(self.all_items, self.candidates[user])
        neg_samples = np.random.choice(candidates, self.num_neg_samples, replace=False)

        user = torch.LongTensor([user])
        neg_samples = torch.LongTensor(neg_samples)

        return user, sequence, prob, neg_samples, target

    def __len__(self):
        return self.num_items
