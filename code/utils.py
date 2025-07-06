import torch
import numpy as np
import pandas as pd
import random
from param import *
from torch_geometric.data import Data
import scipy.sparse as sp
import matplotlib.pyplot as plt
import math
#from scipy import interp

args = parameter_parser()  # 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_load(k):
    Adj = pd.read_csv('../data/association_matrix.csv', header=0) #2315*265
    Dis_simi = pd.read_csv('../data/diease_network_simi.csv', header=0) #265*265
    Meta_simi = pd.read_csv('../data/metabolite_ntework_simi.csv', header=0) #2315*2315
    Adj_next = pd.read_csv('../data/association_matrix.csv', header=0)


    one_matrix = np.mat(np.where(Adj_next == 1))  # 输出邻接矩阵中为“1”的关联关系，维度：2 X 4763
    association_num = one_matrix.shape[1]  # 关联关系数：4763
    random_one = one_matrix.T.tolist()  # list：4763 X 2
    random.seed(args.seed)  # random.seed(): 设定随机种子，使得random.shuffle随机打乱的顺序一致
    random.shuffle(random_one)  # random.shuffle将random_index列表中的元素打乱顺序
    k_folds = args.k_folds
    CV_size = int(association_num / k_folds)  # 每折的个数
    temp = np.array(random_one[:association_num - association_num % k_folds]).reshape(k_folds, CV_size, -1).tolist()  # %取余,每折分952个，结果存储在temp中
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_one[association_num - association_num % k_folds:]  # 将余下的元素加到最后一折里面
    random_index = temp
    metric = np.zeros((1, 7))

    train_matrix = np.matrix(Adj_next, copy=True)  # 将邻接矩阵转化为np矩阵
    train_matrix_root = np.matrix(Adj,copy=True)
    train_matrix_root[tuple(np.array(random_index[k]).T)] = 0  # tuple()转化为元组，将train_matrix中每一折中的测试集元素变为0
    train_matrix[tuple(np.array(random_index[k]).T)] = 0  # tuple()转化为元组，将train_matrix中每一折中的测试集元素变为0

    #节点特征矩阵与边索引、边权值
    x = constructHNet(train_matrix_root)
    adj = constructGNet(train_matrix, Meta_simi, Dis_simi)
    adj_list = adj.tolist()
    edge_index = np.array(np.where(adj > 0))#边索引
    edge_attr_list = []
    for i in range(len(edge_index[0])):
        row = edge_index[0][i]
        col = edge_index[1][i]
        edge_attr_list.append(adj_list[row][col])

    # 验证集正样本
    val_pos_edge_index = np.array(random_index[k]).T
    # 验证集负采样，采集与正样本相同数量的负样本
    val_neg_edge_index = np.mat(np.where(train_matrix_root < 1)).T.tolist()
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = val_neg_edge_index[:val_pos_edge_index.shape[1]]
    val_neg_edge_index = np.array(val_neg_edge_index).T

    #训练集正样本，训练集负样本在epoch内划分，保证更好的可解释性
    train_pos_edge_index = np.mat(np.where(train_matrix_root > 0))  # 训练集边索引，正样本

    data = Data(
        Meta_simi = Meta_simi,
        Adj = torch.tensor(adj, dtype=torch.float).to(device),
        Adj_next = Adj_next,
        train_matrix = torch.tensor(train_matrix, dtype=torch.float).to(device),
        x = torch.tensor(x, dtype=torch.float).to(device),
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device),
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(device),  # 边权值
        val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).to(device),  # tensor格式，验证集负样本
        val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long).to(device),  # tensor格式，验证集正样本
        train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long).to(device)  # tensor格式，训练集正样本
    )
    return data

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def constructGNet(met_dis_matrix,met_matrix,dis_matrix):
    mat1 = np.hstack((met_matrix, met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(met_dis_matrix):
    met_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[0],met_dis_matrix.shape[0]),dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[1],met_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((met_matrix,met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
    adj = np.vstack((mat1,mat2))
    return adj

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(predict_score.flatten()))))  # set只保留唯一值，并从小到大排序
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]  # 抽取999个作为阈值
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))  # 将predict_score复制hresholds_num（999）次
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)  # 正确预测为正样本的数量（真阳率）
    FP = predict_score_matrix.sum(axis=1) - TP  # 错误预测为正样本的数量  求和表示所有正样本个数
    FN = real_score.sum() - TP  # 错误预测为负样本的数量
    TN = len(real_score.T) - TP - FP - FN  # 正确预测为负样本的数量（真阴率）

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [accuracy, precision, recall, f1_score]


#  interaction_matrix原邻接矩阵4536个1  predict_matrix预测邻接矩阵  train_matrix 去掉测试集的训练矩阵4536-907=3629个1
def cv_model_evaluate(output, val_pos_edge_index, val_neg_edge_index):
    edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)
    val_scores = output[edge_index[0], edge_index[1]].to(device)
    val_labels = get_link_labels(val_pos_edge_index, val_pos_edge_index).to(device)  # 训练集中正样本标签
    return val_scores.cpu().numpy(), val_labels.cpu().numpy(), get_metrics(val_labels.cpu().numpy(), val_scores.cpu().numpy())


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i])) #原interp
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='#CF4D4F',  alpha=0.9, label='Mean AUC: %.4f' % mean_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

    plt.gcf().set_size_inches(5, 4)
    # plt.title('ROC curve')
    plt.legend(loc='lower right', fontsize=9.5, frameon=False)  # 修改处：添加 frameon=False
    plt.savefig(directory+'/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()
# def plot_auc_curves(fprs, tprs, auc, directory, name):
#     mean_fpr = np.linspace(0, 1, 20000)
#     tpr = []
#
#     for i in range(len(fprs)):
#         tpr.append(np.interp(mean_fpr, fprs[i], tprs[i])) #原interp
#         tpr[-1][0] = 0.0
#         plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))
#
#     mean_tpr = np.mean(tpr, axis=0)
#     mean_tpr[-1] = 1.0
#     # mean_auc = metrics.auc(mean_fpr, mean_tpr)
#     mean_auc = np.mean(auc)
#     auc_std = np.std(auc)
#     plt.plot(mean_fpr, mean_tpr, color='#cb0000',  alpha=0.9, label='Mean AUC: %.4f' % mean_auc)
#
#     plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)
#
#     # std_tpr = np.std(tpr, axis=0)
#     # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
#     # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
#     # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')
#
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC curve')
#     plt.legend(loc='lower right')
#     plt.savefig(directory+'/%s.pdf' % name, dpi=300, bbox_inches='tight')
#     plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))#原interp
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='#cb0000', alpha=0.9,
             label='Mean AUPR: %.4f' % mean_prc)  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)

    plt.gcf().set_size_inches(5, 4)
    # plt.title('PR curve')
    plt.legend(loc='lower left', fontsize=9.5, frameon=False)
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()


