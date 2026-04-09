import json
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join('case_model', 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args, step, kca_model=None):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    checkpoint = {
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    # 如果传入了 KCA 模型，保存其状态字典
    if kca_model is not None:
        checkpoint['kca_model_state_dict'] = kca_model.state_dict()
    torch.save(checkpoint, os.path.join(args.save_path, f'checkpoint_{step}'))
    # entity_embedding = model.entity_embedding.detach().cpu().numpy()
    # np.save(
    #     os.path.join(args.save_path, f'entity_embedding_{step}.npy'),
    #     entity_embedding
    # )

    # relation_embedding = model.relation_embedding.detach().cpu().numpy()
    # np.save(
    #     os.path.join(args.save_path, f'relation_embedding_{step}.npy'),
    #     relation_embedding
    # )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            parts = line.strip().split()
            h, r, t = parts[:3]

            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_triple2id(file_path):
    '''
    Read triples where the first line is the number of triples,
    and subsequent lines are direct h, r, t (already in id format).
    Return the list of triples, entity set, relation set,
    and their counts.
    '''
    triples = []
    entity_set = set()
    relation_set = set()

    with open(file_path, 'r', encoding='utf-8') as fin:
        # 跳过第一行（三元组数目）
        lines = fin.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                h, t, r = parts[:3]
                h = int(h)
                r = int(r)
                t = int(t)
                triples.append((h, r, t))
                # 将实体和关系添加到对应的集合中
                entity_set.add(h)
                entity_set.add(t)
                relation_set.add(r)

    # 计算实体和关系的数量
    entity_count = len(entity_set)
    relation_count = len(relation_set)

    return triples, entity_count, relation_count



def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    a = []
    for metric in metrics:
        a.append(metrics[metric])
    logging.info(
        '%s at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|' % (mode, step, a[0] * 100, a[1], a[2] * 100, a[3] * 100, a[4] * 100))
    return a


def plot_config(args):
    out_str = "\n data_path:{} model:{} n:{} d:{} g:{} a:{} r:{} lr:{} policy_lr:{} sample:{} pre_sample_num:{} loss_rate:{} exploration_temp:{} batch:{}\n".format(
        args.data_path, args.model, args.negative_sample_size, args.hidden_dim, args.gamma,
        args.adversarial_temperature,
        args.regularization, args.learning_rate, args.kca_learning_rate, args.sample_method, args.pre_sample_num,
        args.loss_rate, args.exploration_temp, args.batch_size)
    with open(args.perf_file, 'a') as f:
        f.write(out_str)
