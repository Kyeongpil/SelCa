import argparse
import os
import pickle
from collections import defaultdict
from math import ceil
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import ujson as json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.sparse import dok_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from evaluation import evaluate_ranking
from interactions import Interactions
from selca import SelCa
from utils import SelCaDataset, set_seed


class Recommender(object):
    def __init__(self, train, test, device, args, item_probs=None, doc2vec=None):
        self.args = args
        self.device = device

        self.train = train
        self.test = test

        self.test_sequence = train.test_sequences
        self._num_items = train.num_items
        self._num_users = train.num_users

        self._net = SelCa(self._num_users, self._num_items, args).to(self.device)

        self._optimizer = AdamW(self._net.parameters(), weight_decay=args.l2, lr=args.learning_rate)
        self.scheduler = StepLR(self._optimizer, step_size=args.decay_step, gamma=args.lr_decay)

        train_dataset = SelCaDataset(train, num_neg_samples=args.neg_samples)
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.n_jobs, pin_memory=True)

        # initialize category embedding vector
        if doc2vec is not None:
            item_vecs = []
            for i in range(1, self._num_items):
                item_vec = normalize(doc2vec.wv[f'i_{i}'].reshape(1, -1))
                item_vecs.append(item_vec.reshape(-1))
                self._net.item_embeddings.weight.data[i] = torch.FloatTensor(item_vec)

            for i in range(self._num_users):
                user_vec = normalize(doc2vec.docvecs[f'u_{i}'].reshape(1, -1))
                self._net.user_embeddings.weight.data[i] = torch.FloatTensor(user_vec)

            item_vecs = np.stack(item_vecs)
            category_vec = item_probs @ item_vecs
            category_vec = normalize(category_vec)
            self._net.category_embeddings.weight.data = torch.FloatTensor(category_vec).to(device)

    def train_one_epoch(self):
        self._net.train()
        avg_loss = 0.0
        for minibatch_num, (user, sequence, prob, neg_samples, target) in enumerate(self.train_dataloader):
            user = user.to(self.device)
            sequence = sequence.to(self.device)
            prob = prob.to(self.device)
            target = target.to(self.device)
            neg_samples = neg_samples.to(self.device)

            target_prediction = self._net(sequence, user, target, prob, self.device)
            negative_prediction = self._net(sequence, user, neg_samples, prob, self.device, use_cache=True)

            self._optimizer.zero_grad()

            positive_loss = -torch.mean(torch.log(torch.sigmoid(target_prediction) + 1e-8))
            negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negative_prediction) + 1e-8))
            loss = positive_loss + negative_loss
            loss.backward()
            self._optimizer.step()
            avg_loss += loss.item()

        avg_loss /= minibatch_num + 1
        self.scheduler.step()
        return avg_loss
    
    def fit(self):        
        # train
        valid_aps = 0
        for e in range(args.n_epochs):
            t1 = time()
            avg_loss = self.train_one_epoch()
            t2 = time()
            if e % 5 == 0 or e == self.args.n_epochs - 1:
                precision, recall, mean_aps = evaluate_ranking(self, self.test, self.train, k=[1, 5, 10])
                precs = [np.mean(p) for p in precision]
                recalls = [np.mean(r) for r in recall]
                output_str = f"Epoch {e+1} [{t2-t1:.1f}s]\tloss={avg_loss:.4f}, map={mean_aps:.4f}, " \
                             f"prec@1={precs[0]:.4f}, prec@5={precs[1]:.4f}, prec@10={precs[2]:.4f}, " \
                             f"recall@1={recalls[0]:.4f}, recall@5={recalls[1]:.4f}, recall@10={recalls[2]:.4f}, [{time()-t2:.1f}s]"

                if mean_aps >= valid_aps:
                    mean_aps = valid_aps
                else:
                    break
                
        print(output_str)
        return {'epochs': e, 'loss': avg_loss, 'mAP': mean_aps, 
                'prec1': precs[0], 'prec5': precs[1], 'prec10': precs[2],
                'recall1': recalls[0], 'recall5': recalls[1], 'recall10': recalls[2]}

    def predict(self, user_id, item_ids=None):
        self._net.eval()

        sequence = self.test_sequence.sequences[user_id, :]
        sequence = np.atleast_2d(sequence)

        with torch.no_grad():
            sequences = torch.from_numpy(sequence.astype(np.int64).reshape(1, -1)).to(self.device)
            item_ids = torch.from_numpy(np.arange(self._num_items).reshape(-1, 1).astype(np.int64)).to(self.device)
            user_id = torch.from_numpy(np.array([[user_id]]).astype(np.int64)).to(self.device)
            probs = torch.from_numpy(self.test_sequence.probs[user_id, :]).to(self.device)
            out = self._net(sequences, user_id, item_ids, probs, self.device, for_pred=True)
        return out


def build_data(args):
    # load dataset
    train = Interactions(args.train_root)
    # transform triplets to sequence representation
    train.to_sequence(args.L, args.T)

    test = Interactions(args.test_root, user_map=train.user_map, item_map=train.item_map)

    if not os.path.exists('./topic_data'):
        os.mkdir('./topic_data')
    topic_path =  f'./topic_data/data_{args.train_data}_{args.L}_{args.topic_num}.pkl'
    if args.train_lda or not os.path.exists(topic_path):
        matrix = dok_matrix((train.num_users, train.num_items - 1))
        user_items = defaultdict(list)
        with open(args.train_root) as f:
            for line in f:
                line = line[:-1]
                if line == '':
                    break
                user, item, _ = line.split()
                matrix[train.user_map[user], train.item_map[item] - 1] += 1
                user_items[train.user_map[user]].append(f'i_{train.item_map[item]}')
                
        lda = LatentDirichletAllocation(n_components=args.topic_num, perp_tol=0.01, max_iter=200, max_doc_update_iter=500, 
                    n_jobs=args.n_jobs, random_state=args.seed, verbose=1, evaluate_every=10)
        lda.fit(matrix)
        item_probs = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        print("LDA training process finished!")
        
        ### calculate train_probs
        ### This process may takes a lot of memories. Therefore, we apply batch processing 
        n_sequences = train.sequences.sequences.shape[0]
        train_probs = np.zeros((n_sequences, args.topic_num))
        
        n_batches = ceil(n_sequences/args.lda_batch_size)
        for n in range(n_batches):
            sub_sequences = train.sequences.sequences[n*args.lda_batch_size: (n+1)*args.lda_batch_size]
            matrix = np.zeros((*sub_sequences.shape, train.num_items - 1), dtype=np.float32)

            i = np.arange(sub_sequences.shape[0]).reshape(-1, 1)
            i = i.repeat(sub_sequences.shape[1], axis=1).reshape(-1)
            j = np.arange(sub_sequences.shape[1]).repeat(sub_sequences.shape[0])
            k = sub_sequences.reshape(-1) - 1

            matrix[i, j, k] += 1
            matrix = matrix.sum(axis=1)

            probs = lda.transform(matrix)
            train_probs[n*args.lda_batch_size: (n+1)*args.lda_batch_size, :] = probs
        
        # calculate test_probs
        n_sequences = train.test_sequences.sequences.shape[0]
        test_probs = np.zeros((n_sequences, args.topic_num))
        n_batches = ceil(n_sequences/args.lda_batch_size)
        for n in range(n_batches):
            sub_sequences = train.test_sequences.sequences[n*args.lda_batch_size: (n+1)*args.lda_batch_size]
            matrix = np.zeros((*sub_sequences.shape, train.num_items - 1), dtype=np.float32)

            i = np.arange(sub_sequences.shape[0]).reshape(-1, 1)
            i = i.repeat(sub_sequences.shape[1], axis=1).reshape(-1)
            j = np.arange(sub_sequences.shape[1]).repeat(sub_sequences.shape[0])
            k = sub_sequences.reshape(-1) - 1

            matrix[i, j, k] += 1
            matrix = matrix.sum(axis=1)

            probs = lda.transform(matrix)
            test_probs[n*args.lda_batch_size: (n+1)*args.lda_batch_size, :] = probs

        train.sequences.probs = train_probs.astype(np.float32)
        train.test_sequences.probs = test_probs.astype(np.float32)

        with open(topic_path, 'wb') as f:
            pickle.dump((train, test, user_items, item_probs), f, protocol=pickle.HIGHEST_PROTOCOL)     
    else:
        with open(topic_path, 'rb') as f:
            train, test, user_items, item_probs = pickle.load(f)

    return train, test, user_items, item_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--use_cuda', action='store_true')
    
    parser.add_argument('--train_data', type=str, default='ml1m')
    parser.add_argument('--L', type=int, default=9) # ml: 9, gowalla: 1
    parser.add_argument('--T', type=int, default=3)
    
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3) # ml: 1e-3, gowalla: 1e-2
    parser.add_argument('--lr_decay', type=float, default=0.5) # ml: 0.5, gowalla: 0.25
    parser.add_argument('--decay_step', type=int, default=8) # ml: 8, gowalla: 10
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    
    parser.add_argument('--train_lda', action='store_true')
    parser.add_argument('--topic_num', type=int, default=30) # ml: 30, gowalla: 5
    parser.add_argument('--lda_batch_size', type=int, default=10000)
    
    parser.add_argument('--d', type=int, default=100) # ml: 100, gowalla: 150
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embedding_dropout', type=float, default=0.1)
    parser.add_argument('--n_attn', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=2)
    args = parser.parse_args()
    print(args)

    args.train_root = f'datasets/{args.train_data}/validation/train.txt'
    args.test_root = f'datasets/{args.train_data}/validation/test.txt'
    
    # set seed
    set_seed(args.seed, cuda=args.use_cuda)
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    train, test, user_items, item_probs = build_data(args)
    
    doc2vec = None
    if args.train_data == 'ml1m':
        user_items = [TaggedDocument(v, [f'u_{k}']) for k, v in user_items.items()]
        doc2vec = Doc2Vec(user_items, vector_size=args.d, min_count=1, workers=4)

    # fit model
    model = Recommender(train, test, device, args, doc2vec=doc2vec, item_probs=item_probs)
    output = model.fit()
    print(output)
