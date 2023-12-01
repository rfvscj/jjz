import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class JJZModel(nn.Module):
    def __init__(self, model_path, charge_num, class_num) -> None:
        super(JJZModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.dim = self.encoder.config.hidden_size
        # 追加嵌入
        self.age2vec = nn.Embedding(100, self.dim)
        self.gender2vec = nn.Embedding(2, self.dim)
        self.health2vec = nn.Embedding(2, self.dim)
        self.nation2vec = nn.Embedding(5, self.dim)
        self.praise2vec = nn.Embedding(20, self.dim)
        self.charge2vec = nn.Embedding(charge_num + 1, self.dim)
        # 融合各个嵌入
        self.fuse_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, self.dim)
        )
        # 分类
        self.predict_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim // 4),
            nn.ReLU(),
            nn.Linear(self.dim // 4, class_num)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()
        
        
    def forward(self, inputs, mode="train"):
        age, gender, health, nation, praise, charge = inputs['age'], inputs['gender'], inputs['health'], inputs['nation'], inputs['praise'], inputs['charge']
        
        age_vec = self.age2vec(age).squeeze(dim=1)
        gender_vec = self.gender2vec(gender).squeeze(dim=1)
        health_vec = self.health2vec(health).squeeze(dim=1)
        nation_vec = self.nation2vec(nation).squeeze(dim=1)
        praise_vec = self.praise2vec(praise).squeeze(dim=1)
        charge_vec = self.charge2vec(charge)
        charge_vec = torch.sum(charge_vec, dim=1)
        
        assert age_vec.shape == gender_vec.shape
        assert age_vec.shape == health_vec.shape
        assert age_vec.shape == nation_vec.shape
        assert age_vec.shape == praise_vec.shape
        assert age_vec.shape == charge_vec.shape, charge_vec.shape
    
        
        appendix_vec = age_vec + gender_vec + health_vec + nation_vec + praise_vec + charge_vec
        # print(appendix_vec.shape)
        appendix_vec = self.fuse_layer(appendix_vec)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # 就不用input_embeds了
        pooler_output = self.encoder(input_ids, attention_mask=attention_mask)[1]
        fused_vec = self.relu(pooler_output + appendix_vec)
        output_logits = self.predict_layer(fused_vec)
        
        if mode == "train":
            reduction = inputs['reduction'].squeeze(dim=1)
            loss = self.criterion(output_logits, reduction)
            return loss
        
        return {
            'logits': output_logits
        }
        
        
        