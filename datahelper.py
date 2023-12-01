import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import pickle
from utils import get_summary, num_clamp
    
    

class JJZDataset(Dataset):
    def __init__(self, file_path, tokenizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), max_length=512, fc2i="assets/charge2id.pkl", fi2c="assets/id2charge.pkl") -> None:
        super().__init__()
        with open(file_path, 'r', encoding='utf8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(fc2i, 'rb') as fc:
            self.charge2id = pickle.load(fc, encoding='utf8')
        with open(fi2c, 'rb') as fi:
            self.id2charge = pickle.load(fi, encoding='utf8')
        self.charge_num = len(self.charge2id)
        self.device = device
        
    def __getitem__(self, index):
        item = self.data[index]
        age = num_clamp(item['age'], 0, 99)
        gender = item['gender']
        nation = item['nation']
        health = item['health']
        praise = num_clamp(item['praise'], 0, 19)
        case_type = item['type']
        # charge
        charge = [0] * 5  # 最多五项罪
        for i, c in enumerate(item['charge']):
            if c not in self.charge2id:
                continue
            charge[i] = self.charge2id[c] + 1
            if i == 4:
                break
        # 以判决书为训练数据
        input_text = get_summary(item['fact2'])
        # 作为label
        reduction = item['reduction']
        hypo = item['hypo']
        score = item['score']
        
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors='pt'
        )
        
        return {
            'age': torch.tensor([age], dtype=torch.long, device=self.device),
            'gender': torch.tensor([gender], dtype=torch.long, device=self.device),
            'nation': torch.tensor([nation], dtype=torch.long, device=self.device),
            'health': torch.tensor([health], dtype=torch.long, device=self.device),
            'charge': torch.tensor(charge, dtype=torch.long, device=self.device),
            'praise': torch.tensor([praise], dtype=torch.long, device=self.device),
            'input_ids': inputs['input_ids'].to(self.device).squeeze(0),
            'attention_mask': inputs['attention_mask'].to(self.device).squeeze(0),
            'type': torch.tensor([case_type], dtype=torch.long, device=self.device),
            'reduction': torch.tensor([reduction], dtype=torch.long, device=self.device),
            'hypo': torch.tensor([hypo], dtype=torch.long, device=self.device),
            'score': torch.tensor([score], dtype=torch.float, device=self.device)
        }
        
    
    def __len__(self):
        return len(self.data)
        
    
# 自定义数据加载策略    
def collate_fn(batch):
    # 懒得搞了
    pass

        
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/public_storage/plms/macbert")
    jjzdata = JJZDataset("data/train_2.json", tokenizer=tokenizer)
    
    for item in jjzdata:
        print(item)
        break
        