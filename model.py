import pickle
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils import measure_distance, num_clamp, get_summary

class JJZModel(nn.Module):
    def __init__(self, model_path, charge_num, class_num, init=False) -> None:
        super(JJZModel, self).__init__()
        if init:
            model_config = AutoConfig.from_pretrained(model_path)
            self.encoder = AutoModel.from_config(model_config)
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
        self.dim = self.encoder.config.hidden_size
        self.lamb = 1
        # 追加嵌入
        self.age2vec = nn.Embedding(100, self.dim)
        self.gender2vec = nn.Embedding(2, self.dim)
        self.health2vec = nn.Embedding(2, self.dim)
        self.nation2vec = nn.Embedding(5, self.dim)
        self.praise2vec = nn.Embedding(20, self.dim)
        self.charge2vec = nn.Embedding(charge_num + 1, self.dim)
        # 假设
        self.hypo2vec = nn.Embedding(39, self.dim)
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
        # 回归
        self.disc_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim // 4),
            nn.ReLU(),
            nn.Linear(self.dim // 4, 1)
        )
        self.transform_layer = nn.Linear(self.dim, self.dim)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
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
        
        # 生成39个hypos，此处也未必要39个全整一遍，下采样一下更好，否则更容易判出更大的偏差
        # TODO 替换策略，给正确的标签分配更多的权重
        hypos = []
        for b in range(pooler_output.shape[0]):
            _hypo = torch.tensor([h for h in range(39)], dtype=torch.long, device=self.encoder.device)
            hypos.append(_hypo.unsqueeze(dim=0))
        hypos = torch.cat(hypos, dim=0)
        
        hypos_vec = self.hypo2vec(hypos)  # batch, 39, dim
        trans_vec = self.transform_layer(fused_vec)
        # 利用广播
        hypos_vec = self.relu(hypos_vec + trans_vec.unsqueeze(dim=1))
        # 预测的就是到每个标签的距离，可能加个激活函数更合适，可能也不需要
        hypos_logits = self.disc_layer(hypos_vec).squeeze(dim=-1)
        hypos_logits = 4.0 * torch.sigmoid(hypos_logits)
        
        if mode == "train":
            reduction = inputs['reduction'].squeeze(dim=1)
            # batch, 39
            dists = self.get_dists(reduction)
            loss_disc = self.mse(hypos_logits, dists)
            
            loss = self.ce(output_logits, reduction)
            return {
                'loss': loss + self.lamb * loss_disc,
                'loss_disc': loss_disc
                }
        
        return {
            'logits': output_logits,
            'hypos_logits': hypos_logits  # 这个变量可以画个柱状图，以更直观地显示一致性
        }
    
    def get_dists(self, reduction):
        # 给定监督标签，按照度量函数，计算所有标签的得分
        # input: batch, 1
        # output: batch, 39
        dists = []
        for redu in reduction:
            # 39
            distance = torch.tensor([measure_distance(h, redu.item()) for h in range(39)], dtype=torch.float, device=self.encoder.device)
            dists.append(distance.unsqueeze(0))
        dists = torch.cat(dists, dim=0)
        return dists

      
# 作为model的一个壳，负责将模型无关的输入输出和模型相关的输入输出间转化
# 后续可考虑扩展为批量的
class ModelShell:
    def __init__(self, args) -> None:
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.model = JJZModel(model_path=args.model_path, charge_num=args.charge_num, class_num=args.class_num, init=True)
        self.load_ckpt()
        self.model.to(args.device)
        
    def load_ckpt(self, ckpt_path=None):
        print("loading model from ckpt...")
        if ckpt_path is None:
            ckpt = torch.load(self.args.ckpt_path, map_location='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt)
        
    def get_output(self, inputs):
        fact = inputs["fact_jdg"]
        summary = get_summary(fact)
        tokens = self.tokenizer.encode_plus(summary, add_special_tokens=True, max_length=self.args.max_length, padding="max_length")
        input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long, device=self.args.device)
        attetion_mask = torch.tensor(tokens['attention_mask'], dtype=torch.long, device=self.args.device)
        
        age = num_clamp(inputs["age"], 0, 99)
        gender = num_clamp(inputs["gender"], 0, 1)
        health = inputs["health"]
        nation = inputs["nation"]
        praise = num_clamp(inputs["praise"], 0, 19)
        charge = inputs["charge"]
        redu = inputs["redu"] if "redu" in inputs else None
        
        with open("assets/charge2id.pkl", 'rb') as c2id:
            charge2id = pickle.load(c2id, encoding="utf8")
        charge_id = []
        for it in charge:
            if it.strip("罪") in charge2id:
                charge_id.append(charge2id[it.strip("罪")])
        # charge_id = [charge2id[it.strip("罪")] for it in charge]
        
        model_inputs = {
            "input_ids": input_ids.unsqueeze(dim=0),
            "attention_mask": attetion_mask.unsqueeze(dim=0),
            "age": torch.tensor([[age]], dtype=torch.long, device=self.args.device),
            "gender": torch.tensor([[gender]], dtype=torch.long, device=self.args.device),
            "health": torch.tensor([[health]], dtype=torch.long, device=self.args.device),
            "nation": torch.tensor([[nation]], dtype=torch.long, device=self.args.device),
            "praise": torch.tensor([[praise]], dtype=torch.long, device=self.args.device),
            "charge": torch.tensor([charge_id], dtype=torch.long, device=self.args.device),
            "redu": torch.tensor([[redu]], dtype=torch.long, device=self.args.device) if redu is not None else None
        }
        
        
        output_dict = self.model(model_inputs, mode="test")
        
        
        
        
        
        # 至此，模型预测出一个适应的减刑时长，以及判决中减刑是否合适
        # 然后，我们需要计算一致性，即减刑时长和模型的评估是否一致
        
        # 这是模型只看文书的事实描述部分，预测的减刑时长。（文书中的结果不可见）
        
        
        suggestion = torch.argmax(output_dict['logits'], dim=1).item()
        # 这是模型看到文书的裁判结果后，对该结果是否合适的评估
        estimation = output_dict['hypos_logits'][0, redu].item() if redu is not None else -1
        # 这是模型预测结果和文书中裁判结果的距离
        distance = measure_distance(suggestion, redu)
        # TODO 这是模型的预测和评估的不一致性（这里忽略了方向，其实并不严谨，可以重新搞一搞
        # TODO 这里还可以把所有可能的都算一遍然后取平均，理论上对方向问题可以平滑
        # TODO 把前边的绝对值去掉，就可以解决方向问题，其实也都可以不弄
        inconsistency = abs(estimation - distance)
        
        
        output_template = {
            "suggestion": suggestion,        # 模型预测的减刑时长（月）
            "estimation": estimation,        # 模型对判决的评分
            "distance": distance,
            "inconsistency": inconsistency,  # 模型的自我不一致性，越低越好
            "summary": summary,           # 案件摘要
            "age": age,               
            "gender": gender,
            "health": health,
            "nation": nation,
            "praise": praise,
            "charge": charge
        }
        
        return output_template