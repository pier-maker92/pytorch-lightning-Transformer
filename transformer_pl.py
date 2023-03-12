import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

from transformer.transformer import Transformer

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def save_history_metrics(**kwargs):
    assert "mode" in kwargs, "please explicit if train or val. mode=train|val"
    mode = kwargs["mode"]
    for name,value in kwargs.items():
        if name!="mode":
            f = open(f"training_results/{mode}-{name}.txt", "a")
            f.write(f"{value:.5f},")
            f.close()

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.2):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        #self.criterion = nn.KLDivLoss(reduction='sum')

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.2):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))

class TransformerPL(pl.LightningModule):
    """
    Pytorch lightning wrapper for the transformer
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, src_pad_token, tgt_pad_token, n_layers=6, n_heads=8, d_ff=2048, d_model=512, dropout=.1, lr=2e-4):
        super(TransformerPL, self).__init__()
        self.transformer = Transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, src_pad_token, tgt_pad_token, n_layers, n_heads, d_ff, d_model, dropout)
        self.criterion = SmoothCrossEntropyLoss() #nn.CrossEntropyLoss(ignore_index=tgt_pad_token)
        self.transformer.apply(initialize_weights)
        self.metric_dir = 'training_results'
        self.logs_setup()
        self.lr = lr

    def logs_setup(self):
        if not os.path.exists(self.metric_dir):
            os.mkdir(self.metric_dir)
        if os.path.exists(os.path.join(self.metric_dir,'train-loss.txt')):
            os.remove(os.path.join(self.metric_dir,'train-loss.txt'))

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        # get transformer output
        output = self.transformer(src, tgt)
        # compute loss
        output = output.contiguous().view(-1,output.size(-1))
        ground_truth = tgt.contiguous().view(-1)
        loss = self.criterion(output,ground_truth)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        save_history_metrics(loss=loss.item(),mode="train")

        return loss



