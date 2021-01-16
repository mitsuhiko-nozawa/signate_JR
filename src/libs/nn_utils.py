from pytorch_tabnet.metrics import Metric
from torch import nn
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, n_cls=2):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing + smoothing / n_cls
        self.smoothing = smoothing / n_cls

    def forward(self, x, target):
        probs = torch.nn.functional.sigmoid(x,)
        # ylogy + (1-y)log(1-y)
        #with torch.no_grad():
        target1 = self.confidence * target + (1-target) * self.smoothing
        #print(target1.cpu())
        loss = -(torch.log(probs+1e-15) * target1 + (1-target1) * torch.log(1-probs+1e-15))
        #print(loss.cpu())
        #nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        #nll_loss = nll_loss.squeeze(1)
        #smooth_loss = -logprobs.mean(dim=-1)
        #loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()