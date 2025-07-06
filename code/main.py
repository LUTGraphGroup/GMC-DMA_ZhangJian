import torch as torch
from mpmath import degree
from model import *
from param import *
from utils import *
from sklearn import metrics
import numpy as np
import pandas as pd
import random
import time

args = parameter_parser()  # 超参数
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    # 训练集负样本
    train_neg_edge_index = np.mat(np.where(data.train_matrix.cpu().numpy() < 1)).T.tolist()
    random.shuffle(train_neg_edge_index)
    train_neg_edge_index = train_neg_edge_index[:data.train_pos_edge_index.shape[1]]
    train_neg_edge_index = np.array(train_neg_edge_index).T
    train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)  # tensor格式训练集负样本

    # output = model(data.x, data.Adj, data.edge_index, data.edge_attr)
    output, contrastive_loss = model(data.x, data.Adj, data.edge_index, data.edge_attr)
    edge_index = torch.cat([data.train_pos_edge_index, train_neg_edge_index], 1)#边索引
    trian_scores = output[edge_index[0], edge_index[1]].to(device)
    trian_labels = get_link_labels(data.train_pos_edge_index, train_neg_edge_index).to(device)  # 训练集中正样本标签

    # 总损失 = 主任务损失 + λ * 对比损失
    main_loss = criterion(trian_scores, trian_labels)
    loss = main_loss + args.Lambda * contrastive_loss  # λ可以调整
    # loss = main_loss

    loss.backward(retain_graph=True)
    optimizer.step()

    return loss, train_neg_edge_index, trian_labels, trian_scores, output

@torch.no_grad()
def mytest(data, trian_labels, trian_scores, model, output, train_neg_edge_index):
    model.eval()

    with torch.no_grad():  # 禁用梯度计算，以避免跟踪计算图中的梯度
        score_train_cpu = np.squeeze(trian_scores.detach().cpu().numpy())
        label_train_cpu = np.squeeze(trian_labels.detach().cpu().numpy())
        train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)

        predict_y_proba = output.reshape(data.Adj_next.shape[0], data.Adj_next.shape[1]).to(device)
        score_val, label_val, metric_tmp = cv_model_evaluate(predict_y_proba, data.val_pos_edge_index, data.val_neg_edge_index)

        fpr, tpr, thresholds = metrics.roc_curve(label_val, score_val)
        precision, recall, _ = metrics.precision_recall_curve(label_val, score_val)
        val_auc = metrics.auc(fpr, tpr)
        val_prc = metrics.auc(recall, precision)

        return score_val, metric_tmp, train_auc, val_auc, val_prc, tpr, fpr, recall, precision

auc_result = []
acc_result = []
pre_result = []
recall_result = []
f1_result = []
prc_result = []
fprs = []
tprs = []
precisions = []
recalls = []

print("seed=%d, evaluating met-disease...." % args.seed)
for k in range(args.k_folds):
    print("------this is %dth cross validation------" % (k + 1))
    data = data_load(k)

    model = GCMC(in_channels = data.x.shape[1],
                hidden_channels = args.hidden_channels,
                out_channels = args.out_channels,
                num_gcn_layers = args.num_gcn_layers,
                num_mamba_layers = args.num_mamba_layers,
                d_state = args.d_state,
                expand = args.expand,
                dropout = args.dropout,
                num_r = data.Meta_simi.shape[0],
                temperature = args.temperature).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) #余弦退火
    criterion =  F.binary_cross_entropy
    best_auc = best_prc = best_epoch = best_tpr = best_fpr = best_recall = best_precision = best_output = 0
    best_score = []
    for epoch in range(args.epochs):
        start_time = time.time()  # 记录epoch开始时间
        train_loss, train_neg_edge_index, trian_labels, trian_scores, output = train(data, model, optimizer, criterion)
        score_val, metric_tmp, train_auc, val_auc, val_prc, tpr, fpr, recall, precision = mytest(data, trian_labels,trian_scores, model,output, train_neg_edge_index)
        end_time = time.time()  # 记录epoch结束时间
        epoch_time = end_time - start_time  # 计算epoch所用时间
        print('Epoch:', epoch + 1, 'Train Loss: %.4f' % train_loss.item(),
              'Acc: %.4f' % metric_tmp[0], 'Pre: %.4f' % metric_tmp[1], 'Recall: %.4f' % metric_tmp[2],
              'F1: %.4f' % metric_tmp[3],
              'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Val PRC: %.4f' % val_prc,
              'Time: %.2f' % (end_time - start_time))

        if val_auc > best_auc:
            metric_tmp_best = metric_tmp
            best_auc = val_auc
            best_prc = val_prc
            best_epoch = epoch + 1
            best_tpr = tpr
            best_fpr = fpr
            best_recall = recall
            best_precision = precision
            best_output = output
            best_score = score_val

    print('Fold:', k + 1, 'Best Epoch:', best_epoch, 'Val acc: %.4f' % metric_tmp_best[0],
          'Val Pre: %.4f' % metric_tmp_best[1],
          'Val Recall: %.4f' % metric_tmp_best[2], 'Val F1: %.4f' % metric_tmp_best[3], 'Val AUC: %.4f' % best_auc,
          'Val PRC: %.4f' % best_prc, best_score,
          )
    acc_result.append(metric_tmp_best[0])
    pre_result.append(metric_tmp_best[1])
    recall_result.append(metric_tmp_best[2])
    f1_result.append(metric_tmp_best[3])
    auc_result.append(best_auc)
    prc_result.append(best_prc)
    fprs.append(best_fpr)
    tprs.append(best_tpr)
    recalls.append(best_recall)
    precisions.append(best_precision)

print('## Training Finished !')
print('-----------------------------------------------------------------------------------------------')
print('Acc', acc_result)
print('Pre', pre_result)
print('Recall', recall_result)
print('F1', f1_result)
print('Auc', auc_result)
print('Prc', prc_result)
print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
        'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
        'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
        'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
        'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
        'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))
# 保存到文件
# result_filename = f"../output/result_{time.strftime('%Y%m%d_%H%M')}.txt"
# with open(result_filename, 'w') as f:
#     f.write('## Training Finished !\n')
#     f.write('-' * 80 + '\n')
#     f.write(f'Acc {acc_result}\n')
#     f.write(f'Pre {pre_result}\n')
#     f.write(f'Recall {recall_result}\n')
#     f.write(f'F1 {f1_result}\n')
#     f.write(f'Auc {auc_result}\n')
#     f.write(f'Prc {prc_result}\n\n')
#     f.write(f'AUC mean: {np.mean(auc_result):.4f}, variance: {np.std(auc_result):.4f}\n')
#     f.write(f'Accuracy mean: {np.mean(acc_result):.4f}, variance: {np.std(acc_result):.4f}\n')
#     f.write(f'Precision mean: {np.mean(pre_result):.4f}, variance: {np.std(pre_result):.4f}\n')
#     f.write(f'Recall mean: {np.mean(recall_result):.4f}, variance: {np.std(recall_result):.4f}\n')
#     f.write(f'F1-score mean: {np.mean(f1_result):.4f}, variance: {np.std(f1_result):.4f}\n')
#     f.write(f'PRC mean: {np.mean(prc_result):.4f}, variance: {np.std(prc_result):.4f}\n')


pd.DataFrame(recalls).to_csv('../output/recalls.csv', index=False)
pd.DataFrame(precisions).to_csv('../output/precisions.csv', index=False)
pd.DataFrame(fprs).to_csv('../output/fprs.csv', index=False)
pd.DataFrame(tprs).to_csv('../output/tprs.csv', index=False)

# 画五折AUC和PR曲线
plot_auc_curves(fprs, tprs, auc_result, directory='../output', name='test_auc')
plot_prc_curves(precisions, recalls, prc_result, directory='../output', name='test_prc')
