



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_



#
class SNS(nn.Module):
    def __init__(self, args, input_dim, output_dim, ent_text_emb, ent_img_emb):
        super(KCA, self).__init__()
        h_dim = 100
        self.args = args

        attention_dim = 50

        if args.double_entity_embedding:
            entity_dim = args.hidden_dim * 2
        else:
            entity_dim = args.hidden_dim
        if args.double_relation_embedding:
            relation_dim = args.hidden_dim * 2
        else:
            relation_dim = args.hidden_dim


        self.linear1 = nn.Linear(in_features=entity_dim + relation_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=entity_dim, bias=True)
        self.relu = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(in_features=entity_dim + relation_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=entity_dim, bias=True)

        self.linear5 = nn.Linear(in_features=entity_dim + relation_dim, out_features=h_dim, bias=True)
        self.linear6 = nn.Linear(in_features=h_dim, out_features=entity_dim, bias=True)

        self.linear7 = nn.Linear(in_features=entity_dim + relation_dim, out_features=h_dim, bias=True)
        self.linear8 = nn.Linear(in_features=h_dim, out_features=entity_dim, bias=True)

        # self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        # self.ent_attn.requires_grad_(True)


        self.init_network()

    def init_network(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
            elif isinstance(m, nn.Embedding):
                xavier_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                xavier_normal_(m.weight)

    def forward(self, kge_model, positive_sample, mode, temperature, pre_sample, pos_neg_mask):
        if mode == 'head-batch':
            h_or_t = 0
        elif mode == 'tail-batch':
            h_or_t = 2

        relation_emb = kge_model.relation_embedding[positive_sample[:, 1]]  # (B, dim)
        batchsize = relation_emb.size(0)

        # 1) flatten & 投影
        text_vec = kge_model.text_emb.weight.view(-1, 384 * 4)  # (E, 1536)
        img_vec = kge_model.img_emb.weight.view(-1, 383 * 24)  # (E, 9192)

        # text_vec = kge_model.text_emb.weight
        # img_vec = kge_model.img_emb.weight

        text_emb = kge_model.text_proj(text_vec)  # (E, entity_dim)
        img_emb = kge_model.img_proj(img_vec)  # (E, entity_dim)


        text_emb = text_emb[pre_sample]
        img_emb = img_emb[pre_sample]
        t = kge_model.entity_embedding[pre_sample]
        t2 = kge_model.entity_embedding2[pre_sample]


        num_entity = text_emb.size(0)
        relation_emb = relation_emb.unsqueeze(1).expand(-1, num_entity, -1)  # (B, n, dim)

        # 2) 统一门控函数
        def gate(embs, lin1, lin2):
            g = torch.cat([embs.unsqueeze(0).expand(batchsize, -1, -1),
                           relation_emb], dim=2)  # (B, n, dim+dim)
            g = self.relu(lin1(g))
            g = torch.sigmoid(lin2(g))  # (B, n, entity_dim)
            return embs.unsqueeze(0).expand(batchsize, -1, -1) * g

        img_att = gate(img_emb, self.linear5, self.linear6)
        text_att = gate(text_emb, self.linear7, self.linear8)
        t_att = gate(t, self.linear1, self.linear3)
        t2_att = gate(t2, self.linear2, self.linear4)


        pos_img_emb, pos_text_emb, pos_tail_emb, pos_tail_emb2 = \
            self.positive_KCA(kge_model, positive_sample, h_or_t)


        # 4) 计算余弦相似度
        def cosine_sim(mat, vec):
            return F.cosine_similarity(mat, vec.unsqueeze(1), dim=2)

        simil_img = cosine_sim(img_att, pos_img_emb)
        simil_text = cosine_sim(text_att, pos_text_emb)
        simil_t = cosine_sim(t_att, pos_tail_emb)
        simil_t2 = cosine_sim(t2_att, pos_tail_emb2)

        # 5) 温度 softmax
        def soft(x):
            return F.softmax(x / temperature, dim=1)
        return (soft(simil_t) + soft(simil_img) + soft(simil_text) + soft(simil_t2),
                simil_img, simil_text, simil_t, simil_t2)

    def positive_KCA(self, kge_model, positive_sample, h_or_t):
        pos_idx = positive_sample[:, h_or_t]  # (B,)
        rel_emb = kge_model.relation_embedding[positive_sample[:, 1]]  # (B, dim)

        # 1) flatten & 投影
        text_vec = kge_model.text_emb.weight[pos_idx].view(-1, 384 * 4)
        img_vec = kge_model.img_emb.weight[pos_idx].view(-1, 383 * 24)

        # text_vec = kge_model.text_emb.weight[pos_idx]
        # img_vec = kge_model.img_emb.weight[pos_idx]

        text_vec = kge_model.text_proj(text_vec)  # (B, entity_dim)
        img_vec = kge_model.img_proj(img_vec)  # (B, entity_dim)

        ent_vec = kge_model.entity_embedding[pos_idx]
        ent2_vec = kge_model.entity_embedding2[pos_idx]

        # 2) 统一门控
        def gate_vec(vec, lin1, lin2):
            g = torch.cat([vec, rel_emb], dim=1)
            g = self.relu(lin1(g))
            g = torch.sigmoid(lin2(g))
            return vec * g

        img_att = gate_vec(img_vec, self.linear5, self.linear6)
        text_att = gate_vec(text_vec, self.linear7, self.linear8)
        ent_att = gate_vec(ent_vec, self.linear1, self.linear3)
        ent2_att = gate_vec(ent2_vec, self.linear2, self.linear4)

        return img_att, text_att, ent_att, ent2_att


