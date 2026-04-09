
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray



#
class KGEModel(nn.Module):
    def __init__(self,sample_method,device,model_name, nentity, nrelation, hidden_dim, gamma,
                 args=None,  # 需添加 args 参数
                 double_entity_embedding=False, double_relation_embedding=False,
                 ent_text_emb=None, ent_img_emb=None):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.device = device
        self.sample_method = sample_method
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )



        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),requires_grad=False)

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity , self.entity_dim))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item() )

        self.entity_embedding2 = nn.Parameter(torch.zeros(nentity , self.entity_dim))
        nn.init.uniform_(tensor=self.entity_embedding2, a=-self.embedding_range.item(), b=self.embedding_range.item() )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation , self.relation_dim))
        nn.init.uniform_( tensor=self.relation_embedding,  a=-self.embedding_range.item(), b=self.embedding_range.item() )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'PairRE', 'TuckER']:
            raise ValueError('model %s not supported' % model_name)

        # 新增参数
        self.args = args  # 需要从外部传入包含 ote_size 等参数的 args

        self.text_emb = ent_text_emb
        self.text_dim = 384
        self.img_emb = ent_img_emb
        self.img_dim = 383
        print("self.ent_text_emb.shape:", self.img_dim)
        print("self.ent_img_emb.shape:", self.text_dim)
        print("self.ent_dim:", self.entity_dim)
        self.img_proj = nn.Linear(self.img_dim, self.entity_dim)
        self.text_proj = nn.Linear(self.text_dim, self.entity_dim)


        self.ent_token = nn.Parameter(torch.Tensor(1, 1, self.entity_dim))
        self.embdr = nn.Dropout(p=0.9)
        self.embdr2 = nn.Dropout(p=0.9)
        self.visdr = nn.Dropout(p=0.4)
        self.txtdr = nn.Dropout(p=0.1)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, self.entity_dim))
        self.pos_str_ent2 = nn.Parameter(torch.Tensor(1, 1, self.entity_dim))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, self.entity_dim))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, self.entity_dim))

        self.str_ent_ln = nn.LayerNorm(self.entity_dim)
        self.str_ent_ln2 = nn.LayerNorm(self.entity_dim)
        self.vis_ln = nn.LayerNorm(self.entity_dim)
        self.txt_ln = nn.LayerNorm(self.entity_dim)

        ent_encoder_layer = nn.TransformerEncoderLayer(self.entity_dim, 2, 1024, 0.1, batch_first=True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, 4)

    def get_embedding(self):
        return self.relation_embedding, self.entity_embedding


    def agg(self):
        ent_tkn = self.ent_token.tile(self.nentity, 1, 1)

        rep_ent_str = self.embdr(self.str_ent_ln(self.entity_embedding.unsqueeze(1))) + self.pos_str_ent
        rep_ent_str2 = self.embdr2(self.str_ent_ln2(self.entity_embedding2.unsqueeze(1))) + self.pos_str_ent2

        rep_ent_vis = self.visdr(self.vis_ln(self.img_proj(self.img_emb))) + self.pos_vis_ent
        rep_ent_txt = self.txtdr(self.txt_ln(self.text_proj(self.text_emb))) + self.pos_txt_ent


        ent_seq1 = torch.cat([ent_tkn, rep_ent_str, ], dim=1)
        ent_seq2 = torch.cat([ent_tkn, rep_ent_vis, ], dim=1)
        ent_seq3 = torch.cat([ent_tkn, rep_ent_txt], dim=1)
        ent_seq4 = torch.cat([ent_tkn,  rep_ent_str2], dim=1)

        str_embdding = self.ent_encoder(ent_seq1)[:, 0]
        vis_embdding = self.ent_encoder(ent_seq2)[:, 0]
        txt_embdding = self.ent_encoder(ent_seq3)[:, 0]
        str_embdding2 = self.ent_encoder(ent_seq4)[:, 0]

        return str_embdding, str_embdding2, vis_embdding, txt_embdding


    def forward(self, sample, mode, train_or_test):
        entity_embedding, entity_embedding2, img_emb, text_emb = self.agg()
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            relation_index=sample[:, 1]


            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            head2 = torch.index_select(
                self.entity_embedding2,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)
            tail2 = torch.index_select(
                entity_embedding2,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            # 添加 head_img
            head_img = torch.index_select(
                img_emb.weight,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)
            # 添加 tail_img
            tail_img = torch.index_select(
                img_emb.weight,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            #text
            head_text = torch.index_select(
                text_emb.weight,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)
            # 添加 tail_img
            tail_text = torch.index_select(
                text_emb.weight,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            # gumbel test has same code with uni.
            if self.sample_method == 'gumbel' and train_or_test == 'train':
                tail_part, head, head_img, head_text, head2 = sample
            else:
                tail_part, head_part = sample
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
                head = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
                head2 = torch.index_select(
                    entity_embedding2,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

                head_img = torch.index_select(
                    img_emb.weight,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

                head_text = torch.index_select(
                    text_emb.weight,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            relation_index = tail_part[:, 1]


            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            tail2 = torch.index_select(
                entity_embedding2,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            tail_img = torch.index_select(
                img_emb.weight,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            tail_text = torch.index_select(
                text_emb.weight,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            if self.sample_method == 'gumbel' and train_or_test == 'train':
                head_part, tail, tail_img, tail_text, tail2 = sample

            else:
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                tail = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
                tail2 = torch.index_select(
                    entity_embedding2,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
                # 添加 tail_img
                tail_img = torch.index_select(
                    img_emb.weight,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
                # 添加 tail_text
                tail_text = torch.index_select(
                    text_emb.weight,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)
            head2 = torch.index_select(
                entity_embedding2,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            relation_index = head_part[:, 1]


            # 添加 head_img
            head_img = torch.index_select(
                img_emb.weight,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            # 添加 head_text
            head_text = torch.index_select(
                text_emb.weight,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)


        else:
            raise ValueError('mode %s not supported' % mode)
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'PairRE': self.PairRE,
            'TuckER': self.TuckER
        }

        score_s_to_s = model_func[self.model_name](head, relation, tail, mode)
        score_i_to_s = model_func[self.model_name](head_img, relation, tail, mode)
        score_s_to_i = model_func[self.model_name](head, relation, tail_img, mode)
        score_i_to_i = model_func[self.model_name](head_img, relation, tail_img, mode)


        score_s2_to_s2 = model_func[self.model_name](head2, relation, tail2, mode)
        score_t_to_s2 = model_func[self.model_name](head_text, relation, tail2, mode)
        score_s2_to_t = model_func[self.model_name](head2, relation, tail_text, mode)
        score_t_to_t = model_func[self.model_name](head_text, relation, tail_text, mode)



        score_i = (score_s_to_s + score_i_to_s + score_s_to_i + score_i_to_i) / 4
        score_t = (score_s2_to_s2 + score_t_to_s2 + score_s2_to_t + score_t_to_t) / 4
        if train_or_test == "train" :
            # Text分支计算
            score1_text = score_s2_to_s2 + score_t_to_t
            score2_text = score_t_to_s2 + score_s2_to_t
            comparison_text = (score1_text < score2_text)
            mask = comparison_text
            score_t[mask] = score_s2_to_s2[mask]

            # # img分支计算
            score1_img = score_s_to_s + score_i_to_i
            score2_img = score_i_to_s + score_s_to_i
            comparison_img = (score1_img < score2_img)
            mask = comparison_img
            score_i[mask] = score_s_to_s[mask]
            return score_i


        score = (score_i + score_t) / 2

        return score

    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


