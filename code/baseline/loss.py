import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import focal_loss

from config import config

# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

def dice_loss(input: torch.FloatTensor, target: torch.LongTensor, use_weights: bool = False, k: int = 0, eps: float = 0.0001):
    """
    Returns the Generalized Dice Loss Coefficient of a batch associated to the input and target tensors. In case `use_weights` \
        is specified and is `True`, then the computation of the loss takes the class weights into account.
    Args:
        input (torch.FloatTensor): NCHW tensor containing the probabilities predicted for each class.
        target (torch.LongTensor): NCHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        use_weights (bool): specifies whether to use class weights in the computation or not.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    """
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # Multiple class case
    n_classes = input.size()[1]
    if n_classes != 1:
        # Convert target to one hot encoding
        target = F.one_hot(target, n_classes).squeeze()
        if target.ndim == 3:
            target = target.unsqueeze(0)
        target = torch.transpose(torch.transpose(target, 2, 3), 1, 2).type(torch.FloatTensor).cuda().contiguous()
        input = torch.softmax(input, dim=1)
    else:
        input = torch.sigmoid(input)   

    class_weights = None
    for i, c in enumerate(zip(input, target)):
        if use_weights:
            class_weights = torch.pow(torch.sum(c[1], (1,2)) + eps, -2)
        s = s + __dice_loss(c[0], c[1], class_weights, k=k)

    return s / (i + 1)

def __dice_loss(input: torch.FloatTensor, target: torch.LongTensor, weights: torch.FloatTensor = None, k: int = 0, eps: float = 0.0001):
    """
    Returns the Generalized Dice Loss Coefficient associated to the input and target tensors, as well as to the input weights,\
    in case they are specified.
    Args:
        input (torch.FloatTensor): CHW tensor containing the classes predicted for each pixel.
        target (torch.LongTensor): CHW one-hot encoded tensor, containing the ground truth segmentation mask. 
        weights (torch.FloatTensor): 2D tensor of size C, containing the weight of each class, if specified.
        k (int): weight for pGD function. Default is 0 for ordinary dice loss.
    """  
    n_classes = input.size()[0]

    if weights is not None:
        for c in range(n_classes):
            intersection = (input[c] * target[c] * weights[c]).sum()
            union = (weights[c] * (input[c] + target[c])).sum() + eps
    else:
        intersection = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps    

    gd = (2 * intersection.float() + eps) / union.float()
    return 1 - (gd / (1 + k*(1-gd)))


class DiceAndFocalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = 0
    def forward(self,pred,label):
        self.loss = focal_loss(pred, label.squeeze(1), alpha=0.25, gamma = 2, reduction='mean').unsqueeze(0)
        self.loss += dice_loss(pred, label.squeeze(1), True, k = 0.75)
        return self.loss


class DiceWCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceWCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        num_classes = inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + smooth) / (cardinality + smooth)).mean()
        dice_loss = (1 - dice_loss)

        wce = F.cross_entropy(inputs, targets, reduction='mean', weight=self.weight)
        dice_wce = wce  + dice_loss 
        return dice_wce    

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        print('===================================')
        print(weights)
        print(score.shape)
        print('===================================')
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])

_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'dfloss': DiceAndFocalLoss,
    'dwc':DiceWCELoss,
    'ohem':OhemCrossEntropy,
}



def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
