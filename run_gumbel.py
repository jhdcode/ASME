#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SNS_model

from util import override_config,save_model,read_triple,set_logger,log_metrics,plot_config, read_triple2id
from dataloader import TrainDataset, TestDataset, BidirectionalOneShotIterator,Emb_MKG_WY,Emb_MMKB_DB15K,Emb_Kuai16K
from tqdm import tqdm
import os
import pickle
import numpy as np
from collections import defaultdict
import logging  # 确保已导入logging


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--perf_file', type=str, default=None)

    parser.add_argument('--data_path', type=str, default='data/MKG-W')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default='./models', type=str)

    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=3, type=int)
    parser.add_argument('-d', '--hidden_dim', default=200, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-sns_lr', '--sns_learning_rate', default=0.001, type=float)
    parser.add_argument('--sample_method', default='gumbel',choices=['uni','gumbel'],type=str)
    parser.add_argument('--pre_sample_num', default=1500,type=int)
    parser.add_argument('--loss_rate', default=100,type=int)
    parser.add_argument('--exploration_temp', default=10,type=int)

    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('--test_batch_size', default=2, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', default=False,type=bool,
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-cpu', '--cpu_num', default=3, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')


    return parser.parse_args(args)


def train_step(kge_model, optimizer, positive_sample, negative_sample, subsampling_weight,
         mode, args,device,simil_img, simil_text,simil_t,pre_sample,pos_neg_mask,neg_for_adv, neg_img_emb, neg_text_emb, neg_emb2=None, simil_t2=None):
    kge_model.train()
    optimizer.zero_grad()


    positive_sample = positive_sample.to(device)
    negative_sample = negative_sample.to(device)
    subsampling_weight = subsampling_weight.to(device)
    # img_neg_emb = img_neg_emb.to(device)
    # text_neg_emb = text_neg_emb.to(device)
    if args.sample_method == 'uni':
        negative_score = kge_model((positive_sample, negative_sample), mode, 'train')  # batch * neg
    else:
        negative_score = kge_model((positive_sample, negative_sample, neg_img_emb, neg_text_emb, neg_emb2), mode, 'train')  # batch * neg
    positive_score = kge_model(positive_sample,'single','train')
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)


    if args.sample_method=='uni':
        if args.model in ['RotatE','PairRE']:
            self_adversarial_weight = F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()  # batch * neg
            negative_score = (self_adversarial_weight * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
    elif args.sample_method=='gumbel':
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        self_adversarial_weight = F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()  # batch * neg
        negative_score = (self_adversarial_weight * F.logsigmoid(-negative_score)).sum(dim=1)

        # 确保 pos_neg_mask 在与 simil_img 相同的设备上
        pos_neg_mask = pos_neg_mask.to(device)

        # pre_sample 确保在正确的设备上
        pre_sample = pre_sample.to(device) if torch.is_tensor(pre_sample) else pre_sample

    ####  contrastive loss self_adversarial
        self_adversarial_weight_for_presample = torch.ones_like(simil_img,device=device,requires_grad=False) /simil_img.shape[-1]  # batchsize x pre_sample
        batch_index = torch.arange(args.batch_size)
        for n in range(neg_for_adv.shape[-1]):
            self_adversarial_weight_for_presample[batch_index,neg_for_adv[:,n]] *= self_adversarial_weight[:,n]

        simil_img = torch.mul( torch.exp(simil_img), self_adversarial_weight_for_presample )  # batchsize x pre_sample
        simil_text = torch.mul( torch.exp(simil_text) , self_adversarial_weight_for_presample )  # batchsize x pre_sample
        simil_t = torch.mul( torch.exp(simil_t) ,self_adversarial_weight_for_presample)   # batchsize x pre_sample
        simil_t2 = torch.mul( torch.exp(simil_t2), self_adversarial_weight_for_presample)  # batchsize x pre_sample

    # pre_sample
        contra_loss_img = - torch.log( torch.sum(simil_img[pos_neg_mask[:,pre_sample]])  / torch.sum(simil_img[~pos_neg_mask[:,pre_sample]])  )
        contra_loss_text = - torch.log( torch.sum(simil_text[pos_neg_mask[:,pre_sample]])  / torch.sum(simil_text[~pos_neg_mask[:,pre_sample]])  )
        contra_loss_t = - torch.log( torch.sum(simil_t[pos_neg_mask[:,pre_sample]])  / torch.sum(simil_t[~pos_neg_mask[:,pre_sample]]))
        contra_loss_t2 = - torch.log( torch.sum(simil_t2[pos_neg_mask[:,pre_sample]])  / torch.sum(simil_t2[~pos_neg_mask[:,pre_sample]]))







    if args.uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

    if args.sample_method=='uni':
        loss = (positive_sample_loss + negative_sample_loss) / 2
    else:
        loss = (positive_sample_loss + negative_sample_loss) / 2  + (contra_loss_img+ contra_loss_text+contra_loss_t+contra_loss_t2)  / (4 * args.loss_rate)

    if args.regularization != 0.0:
        # Use L3 regularization for ComplEx and DistMult
        regularization = args.regularization * (
            kge_model.entity_embedding.norm(p=3) ** 3 + kge_model.img_proj(kge_model.img_emb.weight).norm(p=3) ** 3 +
            kge_model.relation_embedding.norm(p=3).norm(p=3) ** 3
        )
        loss = loss + regularization
        regularization_log = {'regularization': regularization.item()}
    else:
        regularization_log = {}
    pos_heads = positive_sample[:, 0]          # (batch_size,)
    pos_tails = positive_sample[:, 2]          # (batch_size,)
    pos_ent_idx = torch.unique(torch.cat([pos_heads, pos_tails]))  # 去重

    # # 拿到对应行
    # emb1_pos = torch.index_select(
    #     kge_model.entity_embedding, dim=0, index=pos_ent_idx)      # (n_pos_ent, dim)
    # emb2_pos = torch.index_select(
    #     kge_model.entity_embedding2, dim=0, index=pos_ent_idx)     # (n_pos_ent, dim)
    #
    # # 计算余弦相似度的平方均值
    # cos_sim = F.cosine_similarity(emb1_pos, emb2_pos, dim=-1)      # (n_pos_ent,)
    # ortho_loss = torch.mean(cos_sim ** 2)                          # 标量

    # 加到总 loss，ortho_lambda 可在 args 里配置，默认 1e-2 先试
    # ortho_lambda = getattr(args, 'ortho_lambda', 1e-1)
    # loss = loss + ortho_lambda * ortho_loss
    loss.backward()
    optimizer.step()
    log = {
        **regularization_log,
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item()
    }

    return log


def test_step(kge_model, test_triples, all_true_triples, args,device):
    kge_model.eval()

    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'head-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'tail-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    logs = []

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                positive_sample = positive_sample.to(device)
                negative_sample = negative_sample.to(device)
                filter_bias = filter_bias.to(device)

                batch_size = positive_sample.size(0)

                score  = kge_model((positive_sample, negative_sample), mode,'test')
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    return metrics

# def test_step(kge_model, test_triples, all_true_triples, args, device, name="test"):
#     kge_model.eval()
#
#     test_dataloader_head = DataLoader(
#         TestDataset(
#             test_triples,
#             all_true_triples,
#             args.nentity,
#             args.nrelation,
#             'head-batch'
#         ),
#         batch_size=args.test_batch_size,
#         num_workers=max(1, args.cpu_num // 2),
#         collate_fn=TestDataset.collate_fn
#     )
#
#     test_dataloader_tail = DataLoader(
#         TestDataset(
#             test_triples,
#             all_true_triples,
#             args.nentity,
#             args.nrelation,
#             'tail-batch'
#         ),
#         batch_size=args.test_batch_size,
#         num_workers=max(1, args.cpu_num // 2),
#         collate_fn=TestDataset.collate_fn
#     )
#
#     test_dataset_list = [test_dataloader_head, test_dataloader_tail]
#
#     logs = []
#
#     # 收集用于绘图的数据
#     imbalance_scores = []  # (score_s_to_s + score_i_to_i) - (score_i_to_s + score_s_to_i)
#     ranks = []  # 原始排名值
#     mrr_values = []  # 1/rank
#
#     step = 0
#     total_steps = sum([len(dataset) for dataset in test_dataset_list])
#
#     with torch.no_grad():
#         for test_dataset in test_dataset_list:
#             for positive_sample, negative_sample, filter_bias, mode in test_dataset:
#
#                 positive_sample = positive_sample.to(device)
#                 negative_sample = negative_sample.to(device)
#                 filter_bias = filter_bias.to(device)
#
#                 batch_size = positive_sample.size(0)
#
#                 score, score_s_to_s, score_s_to_i, score_i_to_i, score_i_to_s = kge_model(
#                     (positive_sample, negative_sample), mode, 'test')
#                 score += filter_bias
#
#                 # Explicitly sort all the entities to ensure that there is no test exposure bias
#                 argsort = torch.argsort(score, dim=1, descending=True)
#
#                 if mode == 'head-batch':
#                     positive_arg = positive_sample[:, 0]
#                 elif mode == 'tail-batch':
#                     positive_arg = positive_sample[:, 2]
#                 else:
#                     raise ValueError('mode %s not supported' % mode)
#
#                 for i in range(batch_size):
#                     # Notice that argsort is not ranking
#                     ranking = (argsort[i, :] == positive_arg[i]).nonzero()
#                     assert ranking.size(0) == 1
#
#                     # ranking + 1 is the true ranking used in evaluation metrics
#                     ranking = 1 + ranking.item()
#                     mrr = 1.0 / ranking
#                     ranks.append(ranking)  # 添加原始排名值
#                     logs.append({
#                         'MRR': mrr,
#                         'MR': float(ranking),
#                         'HITS@1': 1.0 if ranking <= 1 else 0.0,
#                         'HITS@3': 1.0 if ranking <= 3 else 0.0,
#                         'HITS@10': 1.0 if ranking <= 10 else 0.0,
#                     })
#
#                     # 收集用于绘图的数据
#                     # 获取正样本的各个分数
#                     s_to_s_val = score_s_to_s[i, positive_arg[i]].item()
#                     s_to_i_val = score_s_to_i[i, positive_arg[i]].item()
#                     i_to_i_val = score_i_to_i[i, positive_arg[i]].item()
#                     i_to_s_val = score_i_to_s[i, positive_arg[i]].item()
#
#                     # 计算不平衡分数
#                     imbalance_score = (s_to_s_val + i_to_i_val) - (i_to_s_val + s_to_i_val)
#
#                     imbalance_scores.append(imbalance_score)
#                     mrr_values.append(mrr)
#
#                 if step % args.test_log_steps == 0:
#                     logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
#
#                 step += 1
#
#     metrics = {}
#     for metric in logs[0].keys():
#         metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
#
#         # 计算并添加 avg 指标：MRR、HITS@1、HITS@3、HITS@10 的平均值
#     avg_score = (metrics['MRR'] + metrics['HITS@1'] + metrics['HITS@3'] + metrics['HITS@10']) / 4.0
#     metrics['avg'] = avg_score
#
#     # 仅绘制散点图
#     if imbalance_scores and mrr_values and ranks:
#         # plot_mrr_vs_imbalance(imbalance_scores, mrr_values, name)
#         plot_rank_vs_imbalance(imbalance_scores, ranks, name)
#
#     return metrics
#
#
# def plot_mrr_vs_imbalance(imbalance_scores, mrr_values, name):
#     """MRR vs 不平衡分数的散点图"""
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     plt.figure(figsize=(12, 8))
#
#     # 1. 绘制散点图
#     plt.scatter(imbalance_scores, mrr_values, alpha=0.5, s=10, c=np.log1p(mrr_values),
#                 cmap='viridis', vmin=0, vmax=np.log1p(1.0))
#
#     # 2. 添加颜色条
#     cbar = plt.colorbar()
#     cbar.set_label('Log(MRR + 1)')
#
#     # 3. 设置Y轴范围
#     plt.ylim(0, 1.1)
#
#     # 4. 添加关键MRR参考线
#     key_mrr = [0.001, 0.01, 0.1, 0.3, 0.5, 1.0]
#     for y_pos in key_mrr:
#         plt.axhline(y=y_pos, color='gray', linestyle=':', alpha=0.4)
#
#     # 5. 添加标题和标签
#     plt.title(f"MRR vs Structural Imbalance Score ({name})", fontsize=16)
#     plt.xlabel('(s→s + i→i) - (i→s + s→i)', fontsize=14)
#     plt.ylabel('MRR (1/Rank)', fontsize=14)
#
#     # 6. 添加平衡点参考线
#     plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Balance Point')
#     plt.legend(loc='upper right')
#
#     # 7. 添加网格
#     plt.grid(True, linestyle='--', alpha=0.3)
#
#     # 8. 添加统计信息
#     stats_text = (f"Mean Imbalance: {np.mean(imbalance_scores):.2f}\n"
#                   f"Mean MRR: {np.mean(mrr_values):.4f}\n"
#                   f"Median MRR: {np.median(mrr_values):.4f}\n"
#                   f"Samples: {len(mrr_values)}")
#
#     plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
#              fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
#
#     # 9. 保存图像
#     plt.tight_layout()
#     plt.savefig(f"mrr_imbalance_{name}.png")
#     plt.close()
#     logging.info(f"Saved MRR vs Imbalance plot to mrr_imbalance_{name}.png")
#
#
# def plot_rank_vs_imbalance(imbalance_scores, ranks, name):
#     """排名 vs 不平衡分数的散点图"""
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     plt.figure(figsize=(12, 8))
#
#     # 1. 反转排名值（使得1在顶部）
#     # 因为排名1是最高位置（最佳排名），所以我们需要反转Y轴
#     # 但我们不在数据上反转，而是使用set_ylim反转坐标轴
#     max_rank = max(ranks)
#
#     # 2. 绘制散点图
#     plt.scatter(imbalance_scores, ranks, alpha=0.5, s=10,
#                 c=np.log10(ranks), cmap='viridis_r')
#
#     # 3. 设置Y轴反转和对数刻度
#     plt.yscale('log')  # 使用对数刻度压缩大排名值
#     plt.ylim(max_rank, 0.9)  # 反转Y轴：最大值在底部，最小值在顶部
#     plt.gca().invert_yaxis()  # 确保1在顶部
#
#     # 4. 添加关键排名参考线
#     key_ranks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
#     for rank in key_ranks:
#         if rank <= max_rank:
#             plt.axhline(y=rank, color='gray', linestyle=':', alpha=0.4)
#             plt.text(np.min(imbalance_scores), rank, f" Hits {rank}",
#                      fontsize=9, verticalalignment='center')
#
#     # 5. 添加颜色条
#     cbar = plt.colorbar()
#     cbar.set_label('Log10(Rank)')
#
#     # 6. 添加标题和标签
#     plt.xlabel('f_uni - f_mul', fontsize=14)
#     plt.ylabel('Hit', fontsize=14)
#
#     # 8. 添加网格
#     plt.grid(True, linestyle='--', alpha=0.3)
#
#     # 9. 添加统计信息
#     median_rank = np.median(ranks)
#     hit1_percentage = np.mean(np.array(ranks) <= 1) * 100
#     hit3_percentage = np.mean(np.array(ranks) <= 3) * 100
#     hit10_percentage = np.mean(np.array(ranks) <= 10) * 100
#     mean_imbalance_score = np.mean(imbalance_scores)
#
#     stats_text = (
#                   f"HITS@1: {hit1_percentage:.1f}%\n"
#                   f"HITS@3: {hit3_percentage:.1f}%\n"
#                   f"HITS@10: {hit10_percentage:.1f}%\n"
#                   )
#
#     plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes,
#              fontsize=10, verticalalignment='top', horizontalalignment='right',
#              bbox=dict(boxstyle='round', alpha=0.2))
#
#     # 10. 保存图像
#     plt.tight_layout()
#     plt.savefig(f"rank{name}.png")
#     plt.close()
#     logging.info(f"Saved Rank vs Imbalance plot to rank_imbalance_{name}.png")

def main(args):
    torch.set_num_threads(args.cpu_num)
    device = 'cuda:'+args.gpu if torch.cuda.is_available() else 'cpu'

    if args.perf_file == None:
        args.perf_file = 'results/'+ args.data_path.split('/')[1]  +'-' + args.model + '-' + args.sample_method  + '.txt'
    plot_config(args)
    from KGC_model import KGEModel

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    if args.data_path == 'data/Kuai16K':
        train_triples,  nentity, nrelation = read_triple2id(os.path.join(args.data_path, 'train.txt'))
        logging.info('#train: %d' % len(train_triples))
        valid_triples, _, _ = read_triple2id(os.path.join(args.data_path, 'valid.txt'))
        logging.info('#valid: %d' % len(valid_triples))
        test_triples, _, _ = read_triple2id(os.path.join(args.data_path, 'test.txt'))
        logging.info('#test: %d' % len(test_triples))
        args.nentity = nentity = 16015
        args.nrelation = nrelation = 4
    else:
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('{}/{}.txt'.format(args.data_path, split), encoding='utf-8'):
                parts = line.strip().split()
                sub, rel, obj = parts[:3]

                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        # 读取 entity2id 文件
        entity2id = {}
        with open(os.path.join(args.data_path, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            nentity = int(f.readline().strip())  # 读取第一行的数量
            for line in f:
                ent, idx = line.strip().split()
                entity2id[ent] = int(idx)

        # 读取 relation2id 文件
        relation2id = {}
        with open(os.path.join(args.data_path, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            nrelation = int(f.readline().strip())  # 读取第一行的数量
            for line in f:
                rel, idx = line.strip().split()
                relation2id[rel] = int(idx)

        args.nentity = nentity
        args.nrelation = nrelation


        logging.info('Model: %s' % args.model)
        logging.info('Data Path: %s' % args.data_path)
        logging.info('#entity: %d' % nentity)
        logging.info('#relation: %d' % nrelation)

        train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
        logging.info('#train: %d' % len(train_triples))
        valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
        logging.info('#valid: %d' % len(valid_triples))
        test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
        logging.info('#test: %d' % len(test_triples))

        train_triples_tsr = torch.LongTensor(train_triples).transpose(0,1) #idx X batch
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples





    if args.data_path == 'data/MMKB-DB15K':
        ent_text_emb, ent_img_emb = Emb_MMKB_DB15K(args, entity2id, device)
    elif args.data_path == 'data/Kuai16K':

        ent_text_emb = torch.load("./data/Kuai16K-textual.pth")
        ent_img_emb = torch.load("./data/Kuai16K-video.pth")

    else:
        ent_text_emb, ent_img_emb = Emb_MKG_WY(args, entity2id, device)

    kge_model = KGEModel(
        sample_method = args.sample_method,
        device=device,
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        args=args,
        ent_img_emb = ent_img_emb,
        ent_text_emb = ent_text_emb

    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = kge_model.to(device)


    if args.do_train:
        # Set training dataloader iterator


        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch',args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            drop_last=True
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch',args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            drop_last=True
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        if args.sample_method=='uni':
            optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, kge_model.parameters()), lr=current_learning_rate )
        elif args.sample_method=='gumbel':
            SNS = SNS_model.SNS(args,None, args.nentity,ent_text_emb, ent_img_emb)

            SNS = SNS.to(device)
            optimizer = torch.optim.Adam([{'params':kge_model.parameters(),'lr':current_learning_rate},
                                            {'params':SNS.parameters(),'lr':args.sns_learning_rate},
                                            ])


        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)


    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)



        training_logs = []


        positive_sample_loss,negative_sample_loss,loss = [],[],[]
        # pre_sample_pro = torch.ones([args.batch_size,args.nentity],device=device)
        with tqdm(
                initial=init_step,
                total=args.max_steps,
                desc="Training",
                dynamic_ncols=True,
                postfix={'loss': 0, 'pos_loss': 0, 'neg_loss': 0, 'lr': args.learning_rate}
        ) as pbar:

            for step in range(init_step, args.max_steps):

                positive_sample, negative_uniform_sample, subsampling_weight,mask, mode, idxs = next(train_iterator)

                if args.sample_method=='uni':
                    negative_sample = negative_uniform_sample
                    log = train_step(kge_model, optimizer, positive_sample, negative_sample, subsampling_weight, mode, args,device,0,
                                     0,0,0,0,0,0,0)

                elif args.sample_method=='gumbel':
                    temperature = args.exploration_temp / (1+ torch.log( torch.tensor([step +1],device=device)))

                    # pre_sample = torch.multinomial(pre_sample_pro, args.pre_sample_num, replacement=False,device=device)  # B x pre_sample
                    pre_sample = torch.randperm(args.nentity)[:args.pre_sample_num].to(device) # pre_sample_num
                    pos_neg_mask = torch.le(mask,0.5) # B x num_entity

                    neg_distribution,simil_img, simil_text,simil_t,simil_t2 = SNS(kge_model,positive_sample,mode,temperature,pre_sample,pos_neg_mask) # batch * num_entity
                    ## mask for filter
                    if args.pre_sample_num:
                        neg_distribution = torch.log(neg_distribution) + torch.log(mask.to(device)[:,pre_sample])
                    else:
                        neg_distribution = torch.log(neg_distribution) + torch.log(mask.to(device))

                    neg_all = []
                    neg_for_adv = []
                    for it in range(args.negative_sample_size):
                        neg_onehot = F.gumbel_softmax(neg_distribution,tau=1,hard=True,dim=1)  # batch * presample_entity（one hot）
                        neg_all.append(neg_onehot)
                        neg_for_adv.append(torch.argmax(neg_onehot, dim=1,keepdim=True))
                        # sampling without replacement
                        neg_distribution[neg_onehot.bool()] += torch.log(torch.tensor([1e-38],device=device))

                    neg_all = torch.stack(neg_all,dim=1) # batch * neg_num * presample_entity (one-hot)
                    neg_for_adv = torch.cat(neg_for_adv,dim=1) # batch * neg_num
                    if args.pre_sample_num:
                        neg_emb = torch.matmul(neg_all,kge_model.entity_embedding[pre_sample]) # batch * neg_num * ent_dim
                        neg_img_emb = kge_model.img_proj(torch.matmul(neg_all,kge_model.img_emb(pre_sample)))
                        neg_text_emb = kge_model.text_proj(torch.matmul(neg_all,kge_model.text_emb(pre_sample)))
                        neg_emb2 = torch.matmul(neg_all,kge_model.entity_embedding2[pre_sample]) # batch * neg_num * ent_dim

                    else:
                        neg_emb = torch.matmul(neg_all,kge_model.entity_embedding) # batch * neg_num * ent_dim
                    log = train_step(kge_model, optimizer, positive_sample, neg_emb, subsampling_weight, mode,
                                     args,device,simil_img, simil_text,simil_t,pre_sample,pos_neg_mask,neg_for_adv, neg_img_emb, neg_text_emb, neg_emb2,simil_t2
                                     )


                training_logs.append(log)
                pbar.set_postfix(
                    loss=log.get('loss', 0),
                    pos_loss=log.get('positive_sample_loss', 0),
                    neg_loss=log.get('negative_sample_loss', 0),
                    lr=current_learning_rate,
                    refresh=False  # 避免频繁刷新
                )
                pbar.update(1)
                if step!= 0 and step % 10000 == 0:
                    pbar.refresh()
                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    # optimizer = torch.optim.Adam(
                    #     #filter(lambda p: p.requires_grad, kge_model.parameters()),
                    #     lr=current_learning_rate
                    # )
                    warm_up_steps = warm_up_steps * 3
                    # kge_model.neg_mode = "normal"
                if step!= 0 and step % args.save_checkpoint_steps == 0:
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    # save_model(kge_model, optimizer, save_variable_list, args, step, SNS)


                if step!= 0 and step % 1000 == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                    # log_metrics('Training average', step, metrics)
                    for metric in metrics:
                        logging.info('Training average %s at step %d: %f' % (metric, step, metrics[metric]))
                    training_logs = []
                    positive_sample_loss.append(metrics['positive_sample_loss'])
                    negative_sample_loss.append(metrics['negative_sample_loss'])
                    loss.append(metrics['loss'])

                if step!= 0 and args.do_valid and step % args.valid_steps == 0:
                    logging.info('Evaluating on Valid Dataset...')
                    metrics = test_step(kge_model, valid_triples, all_true_triples, args,device)
                    a = log_metrics('Valid', step, metrics)
                    with open(args.perf_file, 'a') as f:
                        f.write(' Valid at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))

                    logging.info('Evaluating on Test Dataset...')
                    metrics = test_step(kge_model, test_triples, all_true_triples, args,device)
                    a = log_metrics('Test', step, metrics)
                    with open(args.perf_file, 'a') as f:
                        f.write('Test at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))



        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = test_step(kge_model, valid_triples, all_true_triples, args,device)
        a = log_metrics('Valid', step, metrics)
        with open(args.perf_file, 'a') as f:
            f.write(' Valid at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = test_step(kge_model, test_triples, all_true_triples, args,device)
        a = log_metrics('Test', step, metrics)
        with open(args.perf_file, 'a') as f:
            f.write('Test at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = test_step(kge_model, train_triples, all_true_triples, args,device)
        log_metrics('Train', step, metrics)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(42)
    main(parse_args())

