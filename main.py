import logging
import numpy as np
import pandas
from load_data import DATA
from LPKT import LPKT
import pandas as pd
import torch
from scipy import stats   # 记得 requirements 里装 scipy

def mean_confidence_interval(data, confidence=0.95):
    """
    计算均值和置信区间
    data: list 或 numpy array
    confidence: 置信水平（默认 95%）
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)   # 标准误差
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h


def generate_q_matrix(path, n_skill, n_problem, gamma=0.0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1
    return q_matrix


def generate_p_matrix(file_path, p_gamma):
    # 读取txt文件
    with open(file_path, 'r') as f:
        data_str = f.read()
    # 将字符串转换为字典
    data = eval(data_str)
    # 确定矩阵大小
    max_key = max(data.keys())
    max_len = max([len(data[key]) for key in data])
    matrix = np.zeros((max_key + 1, max_len), dtype=float)

    # 将字典转换为矩阵
    for key in data:
        for i, val in enumerate(data[key]):
            if val != -1:
                matrix[key][i] = val ** p_gamma # 使用p_gamma参数计算矩阵元素的值

    return matrix

 # k-fold cross validation
'''k, train_auc_sum, valid_auc_sum = 5, .0, .0
for i in range(k):
    lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix,  batch_size, dropout)
    train_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/train' + str(i) + '.txt')
    valid_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/valid' + str(i) + '.txt')
    best_train_auc, best_valid_auc = lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)
    print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
    train_auc_sum += best_train_auc
    valid_auc_sum += best_valid_auc
print('%d-fold validation: avg of best train auc %f, avg of best valid auc %f' % (k, train_auc_sum / k, valid_auc_sum / k)) '''

# train and pred

# #ednet数据集
batch_size = 128
n_at = 283453             
n_it = 13142             
n_question = 141
n_exercise =  11068
seqlen = 100 
d_k = 128
d_a = 50
d_e = 128
d_f = 11068
d_l = 708289
q_gamma = 0.03
dropout = 0.2
p_gamma = 0.02

q_matrix = generate_q_matrix(
    '/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1/problem2skill',
    n_question, n_exercise,
    q_gamma
)

p_matrix = generate_p_matrix('/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1/ednetmatrix.txt', p_gamma)

dat = DATA(seqlen=seqlen, separate_char=',')
logging.getLogger().setLevel(logging.INFO)

# train_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1/train.txt')
# valid_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1/valid.txt')
# test_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1/test.txt')

train_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1_train_val_tes/train.txt')
valid_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1_train_val_tes/valid.txt')
test_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/ednet_KT1_train_val_tes/test.txt')

lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, batch_size, dropout)
lpkt.train(train_data, valid_data, epoch=6, lr=0.003, lr_decay_step=10)
lpkt.load("/home/q22301203/original/baseline/EduKTM-main/examples/LPKT/ednetparams/lpkt2.params")
_, auc, accuracy, rmse, r2 = lpkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f,  rmse: %.6f,  r2: %.6f" % (auc, accuracy, rmse, r2))



# #assist2017
# batch_size = 32
# n_at = 1326
# n_it = 2839
# n_question = 102
# n_exercise = 3162
# seqlen = 500
# d_k = 128
# d_a = 50
# d_e = 128
# d_f = 3162
# d_l = 1709
# d_x = 3000
# q_gamma = 0.03
# p_gamma = 0.01
# dropout = 0.2 

# q_matrix = generate_q_matrix(
#     '/home/q22301203/original/baseline/EduKTM-main/data/anonymized_full_release_competition_dataset/problem2skill',
#     n_question, n_exercise,
#     q_gamma
# )

# p_matrix = generate_p_matrix('/home/q22301203/original/baseline/EduKTM-main/examples/LPKT/dependency_dict_extend_pp_assist2017.txt', p_gamma)

# dat = DATA(seqlen=seqlen, separate_char=',')
# logging.getLogger().setLevel(logging.INFO)

# # train_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/anonymized_full_release_competition_dataset/train0.txt')
# # valid_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/anonymized_full_release_competition_dataset/valid0.txt')
# # test_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/anonymized_full_release_competition_dataset/test.txt')

# train_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2017train_va_te/train0.txt')
# valid_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2017train_va_te/valid0.txt')
# test_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2017train_va_te/test.txt')

'''
# 加置信区间前
lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, batch_size, dropout)
lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)
lpkt.load("/home/q22301203/original/baseline/EduKTM-main/examples/LPKT/2017params/lpkt3.params")
_, auc, accuracy, rmse, r2 = lpkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f,  rmse: %.6f,  r2: %.6f" % (auc, accuracy, rmse, r2))'''

# # ============= 多次实验，保存最佳结果 =============
# num_runs = 10
# auc_list, acc_list, rmse_list, r2_list = [], [], [], []

# for run in range(num_runs):
#     print(f"\n===== Run {run+1}/{num_runs} =====")
#     torch.manual_seed(run)
#     np.random.seed(run)

#     lpkt = LPKT(n_at, n_it, n_exercise, n_question,
#                 d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, batch_size, dropout)

#     best_train_auc, best_test_auc, best_acc, best_rmse, best_r2 = lpkt.train(
#         train_data, valid_data, epoch=6, lr=0.003, lr_decay_step=10
#     )

#     # 保存最佳结果
#     auc_list.append(best_test_auc)
#     acc_list.append(best_acc)
#     rmse_list.append(best_rmse)
#     r2_list.append(best_r2)

# # ============= 统计平均值 + 95% CI =============
# auc_mean, auc_low, auc_high = mean_confidence_interval(auc_list)
# acc_mean, acc_low, acc_high = mean_confidence_interval(acc_list)
# rmse_mean, rmse_low, rmse_high = mean_confidence_interval(rmse_list)
# r2_mean, r2_low, r2_high = mean_confidence_interval(r2_list)

# print("\n===== Final Best Results with 95% CI =====")
# print(f"AUC:  {auc_mean:.6f}  (95% CI: {auc_low:.6f} ~ {auc_high:.6f})")
# print(f"ACC:  {acc_mean:.6f}  (95% CI: {acc_low:.6f} ~ {acc_high:.6f})")
# print(f"RMSE: {rmse_mean:.6f}  (95% CI: {rmse_low:.6f} ~ {rmse_high:.6f})")
# print(f"R²:   {r2_mean:.6f}  (95% CI: {r2_low:.6f} ~ {r2_high:.6f})")


# # #assist2012数据集
# batch_size = 128
# n_at = 26715
# n_it = 29651
# n_question = 265
# n_exercise = 53091
# seqlen = 100
# d_k = 128
# d_a = 50
# d_e = 128
# d_f = 53091
# d_l = 29018
# q_gamma = 0.1
# p_gamma = 0.03
# dropout = 0.2 


# q_matrix = generate_q_matrix(
#     '/home/q22301203/original/baseline/EduKTM-main/data/assist2012/problem2skill',
#     n_question, n_exercise,
#     q_gamma
# ) 

# p_matrix = generate_p_matrix('/home/q22301203/original/baseline/EduKTM-main/data/assist2012/matrix2012.txt', p_gamma)

# dat = DATA(seqlen=seqlen, separate_char=',')
# logging.getLogger().setLevel(logging.INFO)

# train_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2012train_val_tes/train0.txt')
# valid_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2012train_val_tes/valid0.txt')
# test_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2012train_val_tes/test.txt')

# # train_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2012/train0.txt')
# # valid_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2012/valid0.txt')
# # test_data = dat.load_data('/home/q22301203/original/baseline/EduKTM-main/data/assist2012/test.txt')

# lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, batch_size, dropout)
# #lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout)
# lpkt.train(train_data, valid_data, epoch=4, lr=0.003, lr_decay_step=10)
# lpkt.load("/home/q22301203/original/baseline/EduKTM-main/examples/LPKT/params2012/lpkt2.params")
# _, auc, accuracy, rmse, r2 = lpkt.eval(test_data)
# print("auc: %.6f, accuracy: %.6f,  rmse: %.6f,  r2: %.6f" % (auc, accuracy, rmse, r2))


# 加置信区间前
_, auc, accuracy, r2, rmse = lpkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
print("r2: %.6f, rmse: %.6f" % (r2, rmse)) 

