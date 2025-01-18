import torch
import torch.nn as nn


class Accuracy:
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


class F1Score:
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
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores.append(f1)

        if self.average == "macro":
            return sum(f1_scores) / len(f1_scores)
        elif self.average == "micro":
            total_tp = sum(self.true_positives)
            total_fp = sum(self.false_positives)
            total_fn = sum(self.false_negatives)
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            return (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
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


class F1_K:
    def __init__(self):
        self.user_item_ratings = {}

    def add_batch(self, user_label_index, labels, preds):

        for idx, user in enumerate(user_label_index):
            user = user.item()
            if user not in self.user_item_ratings:
                self.user_item_ratings[user] = {"true": [], "pred": []}
            self.user_item_ratings[user]["true"].append(labels[idx].item())
            self.user_item_ratings[user]["pred"].append(preds[idx].item())

    def f1_at_k_for_user(self, label, pred, k, threshold=4.0):
        if k <= 0 or len(label) == 0 or len(pred) == 0:
            return -1.0, -1.0, -1.0

        # Xác định các mục ground truth (mục liên quan)
        ground_truth = (label >= threshold).nonzero(as_tuple=True)[0]

        # Nếu không có mục nào trong ground truth
        if len(ground_truth) < k:
            return -1.0, -1.0, -1.0

        # Lấy top-k chỉ số từ dự đoán
        top_k_indices = torch.topk(pred, k).indices

        # Tính True Positives (Giao giữa ground_truth và top_k_predictions)
        true_positives = torch.isin(top_k_indices, ground_truth).sum().item()

        # Tính Precision@K
        precision_at_k = true_positives / k

        # Tính Recall@K
        recall_at_k = true_positives / len(ground_truth)
        # Tính F1@K
        if precision_at_k + recall_at_k == 0:
            f1_at_k = 0.0
        else:
            f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

        return f1_at_k, precision_at_k, recall_at_k

    def compute_f1_at_k(self, k=5, threshold=4.0):
        f1_scores = []
        precision_scores = []
        recall_scores = []
        for user, ratings in self.user_item_ratings.items():
            true_ratings = torch.tensor(ratings["true"])
            pred_ratings = torch.tensor(ratings["pred"])

            f1_k, precision_k, recall_k = self.f1_at_k_for_user(
                true_ratings, pred_ratings, k, threshold
            )

            if f1_k < 0.0:  # khoong co positive ground truth
                continue

            f1_scores.append(f1_k)
            precision_scores.append(precision_k)
            recall_scores.append(recall_k)

            # print(f"User {user}: F1@{k} = {f1_k:.4f}, Precision@{k} = {precision_k:.4f}, Recall@{k} = {recall_k:.4f}, {len(true_ratings)} ratings")
        return (
            sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            sum(precision_scores) / len(precision_scores) if precision_scores else 0,
            sum(recall_scores) / len(recall_scores) if recall_scores else 0,
        )


class NDCG_K:
    def __init__(self):
        self.user_item_ratings = {}

    def add_batch(self, user_label_index, labels, preds):

        for idx, user in enumerate(user_label_index):
            user = user.item()
            if user not in self.user_item_ratings:
                self.user_item_ratings[user] = {"true": [], "pred": []}
            self.user_item_ratings[user]["true"].append(labels[idx].item())
            self.user_item_ratings[user]["pred"].append(preds[idx].item())

    def dcg_at_k(self, label, pred, k):
        top_k_indices = torch.topk(pred, k).indices
        top_k_labels = label[top_k_indices]
        gains = 2**top_k_labels - 1
        discounts = torch.log2(torch.arange(len(gains), device=pred.device) + 2)
        return torch.sum(gains / discounts)

    def ndcg_at_k(self, label, pred, k):
        true_dcg = self.dcg_at_k(label, label, k)
        pred_dcg = self.dcg_at_k(label, pred, k)
        return pred_dcg / true_dcg

    def compute_ndcg_at_k(self, k=5):
        ndcg_scores = []
        for user, ratings in self.user_item_ratings.items():
            true_ratings = torch.tensor(ratings["true"])
            pred_ratings = torch.tensor(ratings["pred"])

            if len(true_ratings) < k or k < 2:
                continue
            ndcg_k = self.ndcg_at_k(true_ratings, pred_ratings, k)
            ndcg_scores.append(ndcg_k)
            # print(f"User {user}: NDCG@{k} = {ndcg_k:.4f}, {len(true_ratings)} ratings")
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0


if __name__ == "__main__":
    f1_k = F1_K()
    nDCG_K = NDCG_K()
    label = torch.tensor([5, 4, 4, 5, 1, 3, 4])
    pred = torch.tensor([4, 5, 3, 5, 1, 1, 1])
    res = nDCG_K.ndcg_at_k(label, pred, 3)
