import torch
import torch.nn as nn


class Accuracy():
    def __init__(self):
        self.total_corrects = 0
        self.total_labels = 0

    def update(self, pred, target):
        correct = pred.eq(target).sum().item()
        self.total_corrects += correct
        self.total_labels += target.numel()
    
    def compute(self):
        return self.total_corrects / self.total_labels
    
    def reset(self):
        self.total_corrects = 0
        self.total_labels = 0