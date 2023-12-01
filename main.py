import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--model_path", default="macbert", type=str)
parser.add_argument("--train_file", default="data/train_2.json", type=str)
parser.add_argument("--valid_file", default="data/valid_2.json", type=str)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--epochs", default=5, type=int)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datahelper import JJZDataset
from model import JJZModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import batch_acc


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def eval(model, valid_loader):
    model.eval()
    tq = tqdm(valid_loader)
    acc_sum = 0
    step = 0
    for batch in tq:
        step += 1
        reduction = batch['reduction'].squeeze(1)
        logits = model(batch, mode="eval")['logits']
        exact_acc = batch_acc(logits, reduction)
        acc_sum += exact_acc
        acc_avg = acc_sum / step
        tq.set_postfix_str(f"acc: {round(acc_avg, 4)}")
    return acc_avg
        
        
        


def train(args, model, train_loader, valid_loader=None):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if valid_loader:
        best_acc = eval(model, valid_loader)
    else:
        best_acc = 0
    total_steps = len(train_loader) * args.epochs
    warmup_rate = 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_rate * total_steps, num_training_steps=total_steps)
    for epoch in range(args.epochs):
        total_loss = 0
        step = 0
        tq = tqdm(train_loader)
        model.train()
        for batch in tq:
            step += 1
            loss = model(batch, mode="train")
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            tq.set_postfix_str(f"epoch: {epoch + 1} loss: {str(round(total_loss / step, 4))}")
        
        if valid_loader:
            acc = eval(model, valid_loader)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "ckpt.bin")
        else:
            torch.save(model.state_dict(), "ckpt.bin")


if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    train_dataset = JJZDataset(args.train_file, tokenizer, device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = JJZDataset(args.valid_file, tokenizer, device=DEVICE)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    
    model = JJZModel(model_path=args.model_path, charge_num=train_dataset.charge_num, class_num=39)
    
    model.to(DEVICE)
    
    train(args, model, train_loader, valid_loader)
    
