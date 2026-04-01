# coding: utf-8
# 2021/8/17 @ sone

import math
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm

from EduKTM import KTM
from LPKTNet import LPKTNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score
from LMEKT import LMEKT
from DLST import DLST


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def etl(*args, **kwargs) -> ...:  # pragma: no cover
    """
    extract - transform - load
    """
    pass


def train(*args, **kwargs) -> ...:  # pragma: no cover
    pass


def evaluate(*args, **kwargs) -> ...:  # pragma: no cover
    pass


class KTM(object):
    def __init__(self, *args, **kwargs) -> ...:
        pass

    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError



def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <=  0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def compute_rmse(all_target, all_pred):
    return np.sqrt(np.mean((all_target - all_pred)**2))

def train_one_epoch(net, optimizer, criterion, batch_size, k_data, a_data, e_data, it_data, at_data, al_data, df_data):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size))
    shuffled_ind = np.arange(e_data.shape[0])
    np.random.shuffle(shuffled_ind)
    k_data = k_data[shuffled_ind]
    e_data = e_data[shuffled_ind]
    at_data = at_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    al_data = al_data[shuffled_ind]
    df_data = df_data[shuffled_ind]
    
    pred_list = []
    target_list = []
    pred_rmse = []

    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()

        k_one_seq = k_data[idx * batch_size: (idx + 1) * batch_size, :]
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        al_one_seq = al_data[idx * batch_size: (idx + 1) * batch_size, :]
        df_one_seq = df_data[idx * batch_size: (idx + 1) * batch_size, :]
       

        input_k = torch.from_numpy(k_one_seq).long().to(device)
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_al = torch.from_numpy(al_one_seq).long().to(device)
        input_df = torch.from_numpy(df_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        

        pred = net(input_k, input_e, input_at, target, input_it, input_al, input_df)

        mask = input_e[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        loss = criterion(masked_pred, masked_truth).sum()

        loss.backward()
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()
        
        pred_rmse += list(masked_pred)

        if len(masked_pred) == 0 or len(masked_truth) == 0:
            print("Empty masked_pred or masked_truth, idx =", idx)

        pred_list.append(masked_pred)
        target_list.append(masked_truth)

        if len(pred_list) == 0 or len(target_list) == 0:
            print("Empty pred_list or target_list")


    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    all_pred_rmse = np.array(pred_rmse)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    r2 = r2_score(all_target, all_pred_rmse)
    rmse = compute_rmse(all_target,all_pred_rmse)

    return loss, auc, accuracy, rmse, r2


def test_one_epoch(net, batch_size, k_data, a_data, e_data, it_data, at_data, al_data, df_data):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []
    pred_rmse = []

    for idx in tqdm.tqdm(range(n), 'Testing'):
        k_one_seq = k_data[idx * batch_size: (idx + 1) * batch_size, :]
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        al_one_seq = al_data[idx * batch_size: (idx + 1) * batch_size, :]
        df_one_seq = df_data[idx * batch_size: (idx + 1) * batch_size, :]
       
        input_k = torch.from_numpy(k_one_seq).long().to(device)
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_al = torch.from_numpy(al_one_seq).long().to(device)
        input_df = torch.from_numpy(df_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        with torch.no_grad():
            pred = net(input_k, input_e, input_at, target, input_it, input_al, input_df)

            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)
            pred_rmse += list(masked_pred)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    all_pred_rmse = np.array(pred_rmse)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    """ r2 = calculate_r2(all_target, all_pred)
    rmse = calculate_rmse(all_target, all_pred) """
    rmse = compute_rmse(all_target,all_pred_rmse )
    r2 = r2_score(all_target, all_pred_rmse)

    return loss, auc, accuracy, rmse, r2


class LPKT(KTM):
    #def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout=0.2):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, batch_size, dropout=0.2):
        super(LPKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        p_matrix = torch.from_numpy(p_matrix).float().to(device)
        
        self.lpkt_net = LPKTNet(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, dropout).to(device)
        self.batch_size = batch_size

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002, lr_decay_step=15, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.lpkt_net.parameters(), lr=lr, eps=1e-8, betas=(0.1, 0.999), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')
        criterion_mse = nn.MSELoss(reduction='none')
        best_train_auc, best_test_auc = .0, .0

        for idx in range(epoch):
            train_loss, train_auc, train_accuracy,train_r2, train_rmse = train_one_epoch(self.lpkt_net, optimizer, criterion,
                                                                    self.batch_size, *train_data) 
            train_loss, train_auc, train_accuracy , train_rmse, r2 = train_one_epoch(self.lpkt_net, optimizer, criterion,
                                                                    self.batch_size, *train_data)
            print("[Epoch %d] R2: %.6f" % (idx, train_r2))
            print("[Epoch %d] RMSE: %.6f" % (idx, train_rmse)) 

            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            if train_auc > best_train_auc:
                best_train_auc = train_auc

            scheduler.step()

            if test_data is not None:
                test_loss, test_auc, test_accuracy ,test_rmse ,test_r2= self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, r2: %.6f" % (idx, test_auc, test_accuracy, test_rmse, test_r2))
                if test_auc > best_test_auc:
                    torch.save(self.lpkt_net.state_dict(), "/home/q22301203/original/baseline/EduKTM-main/examples/LPKT/2017params/lpkt3.params")
                    best_test_auc = test_auc
                    best_test_accuracy = test_accuracy
                    best_test_rmse = test_rmse
                    best_test_r2 = test_r2
                    
                    print(f"valida auc:{test_auc}")
                    print(f"The beat epoch is {idx}")
                    best_test_auc = test_auc

        return best_train_auc, best_test_auc, best_test_accuracy, best_test_rmse, best_test_r2


    def eval(self, test_data) -> ...:
        self.lpkt_net.eval()
        return test_one_epoch(self.lpkt_net, self.batch_size, *test_data)

    def save(self, filepath) -> ...:
        torch.save(self.lpkt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.lpkt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

