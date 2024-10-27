from typing import List
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from .metrics import Metric


@METRICS.register_module()
class ISTDMetrics(BaseMetric):
    default_prefix = "ISTD"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = []
        self.bins = 10
        for _ in range(self.bins):
            self.metrics.append(Metric())

    def process(self, data_batch, data_samples):
        for i, result in enumerate(data_samples):
            y = result.unsqueeze(0)
            gt = data_batch["gt"][i : i + 1, :].to(result.device)
            for i in range(self.bins):
                self.metrics[i].update(y > ((i + 1) / self.bins), gt > 0)

    def compute_metrics(self, results: List):
        ret = {}
        for i in range(self.bins):
            metric_result = self.metrics[i].get()
            ret[f"Thres{((i + 1) / self.bins):1f}/iou%"] = metric_result[1] * 100
            ret[f"Thres{((i + 1) / self.bins):1f}/niou%"] = metric_result[2] * 100
            ret[f"Thres{((i + 1) / self.bins):1f}/Fa1e-6"] = metric_result[3] * 1e6
            ret[f"Thres{((i + 1) / self.bins):1f}/Pd%"] = metric_result[4] * 100
        return ret
