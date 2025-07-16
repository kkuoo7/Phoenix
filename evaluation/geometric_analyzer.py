import torch
import logging

logger = logging.getLogger(__name__)

def to_tensor_safe(f):
    if isinstance(f, torch.Tensor):
        return f.detach().clone()
    elif isinstance(f, list):
        return torch.stack([to_tensor_safe(x) for x in f])
    else:
        return torch.tensor(f, dtype=torch.float32)

class GeometricAnalyzer:
    """
    GNC2/UNC3 등 기하학적 representation collapse 메트릭 계산 전용 클래스
    """
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    def group_features_by_token(self, features, token_ids):
        token_features = {}
        special_token_ids = set()
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                special_token_ids.add(self.tokenizer.eos_token_id)
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                special_token_ids.add(self.tokenizer.pad_token_id)
            if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                special_token_ids.add(self.tokenizer.bos_token_id)
            if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
                special_token_ids.add(self.tokenizer.unk_token_id)
        for i, token_id in enumerate(token_ids):
            token_id = token_id.item() if hasattr(token_id, 'item') else int(token_id)
            if token_id in special_token_ids:
                continue
            if token_id not in token_features:
                token_features[token_id] = []
            token_features[token_id].append(features[i])
        for token_id in list(token_features.keys()):
            if len(token_features[token_id]) >= self.config.min_samples_per_token:
                feature_list = token_features[token_id]
                try:
                    tensor_features = [to_tensor_safe(f) for f in feature_list]
                    token_features[token_id] = torch.stack(tensor_features)
                except Exception as e:
                    logger.error(f"Failed to stack tensors for token {token_id}: {e}")
                    del token_features[token_id]
            else:
                del token_features[token_id]
        return token_features

    def compute_gnc2(self, token_features):
        try:
            class_means = []
            for token_id, features in token_features.items():
                if isinstance(features, list):
                    if len(features) == 0:
                        continue
                    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features])
                if features.shape[0] > 1:
                    class_mean = features.mean(dim=0)
                    class_means.append(class_mean)
            if len(class_means) < 2:
                return 0.0
            class_means = torch.stack(class_means)
            global_mean = class_means.mean(dim=0, keepdim=True)
            centered_means = class_means - global_mean
            normalized_means = torch.nn.functional.normalize(centered_means, p=2, dim=1)
            pdist = torch.pdist(normalized_means, p=2)
            log_inv_distances = torch.log(1.0 / pdist)
            mean_dist = log_inv_distances.mean()
            std_dist = log_inv_distances.std()
            return (std_dist / mean_dist).item()
        except Exception as e:
            logger.error(f"Failed to compute GNC2: {e}")
            return 0.0

    def compute_unc3(self, token_features, classifier_weights):
        try:
            class_means_map = {}
            for token_id, features in token_features.items():
                if isinstance(features, list):
                    if len(features) == 0:
                        continue
                    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features])
                if features.shape[0] > 1:
                    class_means_map[token_id] = features.mean(dim=0)
            if len(class_means_map) < 2:
                return 0.0
            common_ids = sorted(list(class_means_map.keys()))
            class_means = torch.stack([class_means_map[cid] for cid in common_ids])
            weights = classifier_weights[common_ids, :]
            global_mean = class_means.mean(dim=0, keepdim=True)
            normalized_means = torch.nn.functional.normalize(class_means - global_mean, p=2, dim=1)
            normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
            similarities = (normalized_weights * normalized_means).sum(dim=1)
            mean_sim = similarities.mean()
            std_sim = similarities.std()
            return (std_sim / mean_sim).item()
        except Exception as e:
            logger.error(f"Failed to compute UNC3: {e}")
            return 0.0

    def get_gnc2_unc3_metrics(self, features, token_ids, classifier_weights=None):
        token_features = self.group_features_by_token(features, token_ids)
        gnc2 = self.compute_gnc2(token_features)
        unc3 = None
        if classifier_weights is not None:
            unc3 = self.compute_unc3(token_features, classifier_weights)
        return {'gnc2': gnc2, 'unc3': unc3} 