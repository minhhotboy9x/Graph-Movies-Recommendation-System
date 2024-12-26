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

class F1Score():
    def __init__(self, num_classes=2, average=None):
        """
        Class tính F1-score cho bài toán classification.
        
        Args:
            num_classes (int): Số lớp trong bài toán.
            average (str): Kiểu tính trung bình: 
                           - 'macro' cho trung bình không trọng số giữa các lớp.
                           - 'micro' cho trung bình có trọng số dựa trên tổng số mẫu.
                           - None trả về từng f1-score cho từng lớp.
        """
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def update(self, pred, target):
        """
        Cập nhật số liệu dự đoán và nhãn thực tế.

        Args:
            pred (torch.Tensor): Dự đoán (kích thước N, với giá trị thuộc [0, num_classes-1]).
            target (torch.Tensor): Nhãn thực tế (kích thước N, với giá trị thuộc [0, num_classes-1]).
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        for cls in range(self.num_classes):
            # Tính TP, FP, FN cho từng lớp
            tp = ((pred == cls) & (target == cls)).sum().item()
            fp = ((pred == cls) & (target != cls)).sum().item()
            fn = ((pred != cls) & (target == cls)).sum().item()
            
            self.true_positives[cls] += tp
            self.false_positives[cls] += fp
            self.false_negatives[cls] += fn

    def compute(self):
        """
        Tính F1-score dựa trên số liệu đã được cập nhật.
        """
        f1_scores = []
        for cls in range(self.num_classes):
            tp = self.true_positives[cls]
            fp = self.false_positives[cls]
            fn = self.false_negatives[cls]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        if self.average == 'macro':
            return sum(f1_scores) / len(f1_scores)
        elif self.average == 'micro':
            total_tp = sum(self.true_positives)
            total_fp = sum(self.false_positives)
            total_fn = sum(self.false_negatives)
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        elif self.average is None:
            return {f"f1_{cls}": f1 for cls, f1 in enumerate(f1_scores)}
        else:
            raise ValueError("Unsupported average type. Choose 'macro' or 'micro'.")

    def reset(self):
        """
        Reset lại các thông số để tính lại từ đầu.
        """
        self.true_positives = [0] * self.num_classes
        self.false_positives = [0] * self.num_classes
        self.false_negatives = [0] * self.num_classes