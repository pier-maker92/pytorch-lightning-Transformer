import sys
sys.path.append('..')

import torch
import argparse
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from transformers import AutoTokenizer
from transformer_pl import TransformerPL
from translate.storage.tmx import tmxfile
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="train_loss",
    mode="min",
    dirpath="lightning_logs/",
    filename="transformer-{epoch:02d}-{train_loss:.5f}",
    save_on_train_epoch_end = True
)

def EnIt_collate(batch): 
    src = []; tgt = []
    for item in batch:
        src.append(item[0])
        tgt.append(item[1])
    return torch.cat(src,dim=0), torch.cat(tgt,dim=0)

class EnIt(Dataset):
    def __init__(self, corpus_dir, split, split_val=.1, reduction=0):
        with open(corpus_dir, 'rb') as fin:
            tmx_file = tmxfile(fin, 'en', 'ar')

        corpus = list(tmx_file.unit_iter())
        if reduction:
            corpus = corpus[:reduction]
        
        # split train/val     
        if split == "train":
            corpus = corpus[:int(len(corpus)*(1-split_val))]
        else: 
            corpus = corpus[int(len(corpus)*(1-split_val)):]

        self.tokenizer_en = AutoTokenizer.from_pretrained("bert-base-cased")
        self.tokenizer_it = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
        
        print('\ncompute tokens ids for src')
        src_tokens = [self.tokenizer_en(doc.source)['input_ids'] for doc in tqdm(corpus)]
        print('compute tokens ids for tgt\n')
        tgt_tokens = [self.tokenizer_it(doc.target)['input_ids'] for doc in tqdm(corpus)]
        # filter for samples with length <= 512 tokens
        src_mask = np.where(np.array([512 <= len(tokens) for tokens in src_tokens]).astype(int)*-1+1)[0]
        tgt_mask = np.where(np.array([512 <= len(tokens) for tokens in tgt_tokens]).astype(int)*-1+1)[0]
        mask = np.array(list(set(src_mask) & set(tgt_mask)))
        src_tokens = list(np.array(src_tokens)[mask])
        tgt_tokens = list(np.array(tgt_tokens)[mask])
        assert len(src_tokens) == len(tgt_tokens)
        print(f"{len(src_tokens)} samples in {split} set")

        self.corpus = {
            'src': src_tokens,
            'tgt': tgt_tokens
        }
        self.max_tgt_len = self.max_src_len = 512
        self.pad_tgt_value = self.tokenizer_it.pad_token_id
        self.pad_src_value = self.tokenizer_en.pad_token_id
    
    def __len__(self):
        return len(self.corpus['src'])

    def __getitem__(self, idx):
        src, tgt = torch.tensor(self.corpus['src'][idx]), torch.tensor(self.corpus['tgt'][idx])
        # pad src
        if src.size(-1) < self.max_src_len:
            src = torch.nn.ConstantPad1d((0, self.max_src_len - src.size(-1)), self.pad_src_value)(src)
        # eos tgt
        if tgt.size(-1) < self.max_tgt_len:
            tgt = torch.nn.ConstantPad1d((0, self.max_tgt_len - tgt.size(-1)), self.pad_tgt_value)(tgt)
        
        return src.unsqueeze(0), tgt.unsqueeze(0)
    
class Trainer():
    def __init__(self, model, training_data, hparams):
        self.model = model

        self.hparams = hparams
        self.train_dataloader = DataLoader(training_data, 
                                           batch_size = self.hparams["batch_size"], 
                                           drop_last = True,
                                           shuffle=True,
                                           collate_fn=EnIt_collate)
        
    def train(self,devices=-1):
        trainer = pl.Trainer(max_epochs=self.hparams["epochs"], 
                             accelerator='gpu', 
                             devices=devices, 
                             #strategy='ddp', not working with mps, use if use cuda and have mutiple gpus
                             callbacks=[checkpoint_callback])
        
        # Train the model ⚡🚅⚡
        trainer.fit(self.model, self.train_dataloader)
        
        
# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', default=100, type=int,
                    help="Numbers of epochs. Default = 100")
parser.add_argument('-b','--batch_size', default=64, type=int,
                    help="Batch size. Default = 64")
parser.add_argument('--lr', default=1e-4, type=float,
                    help="Learning rate. Default = 2e-4")
parser.add_argument('--n_gpus', default=1, type=int,
                    help="Number of gpus to use. -1 means all of them")
parser.add_argument('-r','--reduction', default=0, type=int,
                    help="Number of samples to use. This is for testing purpose. 0 means all data")

if __name__=="__main__":
    # get args
    args = parser.parse_args()
    
    # get training data
    training_data = EnIt(corpus_dir="../data/en-it.tmx", split="train", reduction=args.reduction)

    # define model
    hparams = {
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    src_vocab_size = training_data.tokenizer_en.vocab_size
    tgt_vocab_size = training_data.tokenizer_it.vocab_size

    src_seq_len = training_data.max_src_len
    tgt_seq_len = training_data.max_tgt_len

    src_pad_token = training_data.pad_src_value
    tgt_pad_token = training_data.pad_tgt_value

    # model
    model = TransformerPL(src_vocab_size, 
                          tgt_vocab_size, 
                          src_seq_len, 
                          tgt_seq_len, 
                          src_pad_token, 
                          tgt_pad_token, 
                          n_layers=3, 
                          n_heads=4, 
                          d_ff=1024, 
                          d_model=256, 
                          dropout=.1, 
                          lr=hparams['lr'])

    # train ⚡🚅⚡
    trainer = Trainer(model, training_data, hparams)
    trainer.train(devices=args.n_gpus)