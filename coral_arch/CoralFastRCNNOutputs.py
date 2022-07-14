from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from torch.nn import functional as F

class CoralFastRCNNOutputs(FastRCNNOutputs):
    ""
    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            background_class = self.pred_class_logits.shape[1] - 1
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean", ignore_index=background_class)