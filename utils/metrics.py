from common import *


def criterion_AngularError(logits, labels, areas):
    # eps = 10**-6
    # weights = torch.gt(areas, 0).float() / torch.sqrt(areas+eps)
    # weights = weights / (weights.max()+eps)
    weights = areas
    loss = AngularErrorLoss()(logits[:, :2], labels[:, :2], weights) + \
           AngularErrorLoss()(logits[:, 2:], labels[:, 2:], weights)

    return loss


def criterion_BCE_SoftDice(logits, labels, dice_w=None, use_weight=False):
    # compute weights
    batch_size, C, H, W = labels.shape
    if use_weight:
        if H <= 128:
            kernel_size = 11
        elif H <= 256:
            kernel_size = 21
        elif H <= 512:
            kernel_size = 21
        else:
            kernel_size = 41
        a = F.avg_pool2d(labels[:, 0], kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        border = (a.ge(0.01) * a.le(0.99)).float()
        weights = torch.ones(a.shape).cuda(async=True)
        w0 = weights.sum()
        weights = weights + border*2
        w1 = weights.sum()
        weights = weights * (w0 / w1)
        weights = weights.repeat((C, 1, 1)).reshape(labels.shape)
    else:
        weights = torch.ones(labels.shape).cuda(async=True)

    loss = WeightedBCELoss2d()(logits, labels, weights)
    for d in range(C):
        w = 1/C if dice_w is None else dice_w[d]
        loss = loss + w * WeightedSoftDiceLoss()(logits[:, d], labels[:, d], weights[:, d])

    return loss


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num   = labels.shape[0]
        w     = weights.view(num,-1)
        w2    = w*w
        m1    = probs.view(num,-1)
        m2    = labels.view(num,-1)
        intersection = (m1 * m2)
        smooth = 1
        score = (2. * (w2*intersection).sum(1) + smooth) / ((w2*m1).sum(1) + (w2*m2).sum(1) + smooth)
                # + (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss = 1 - score.sum()/num
        return loss


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view(-1)
        t = labels.view(-1)
        # Pytorch implementation https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py (binary_cross_entropy_with_logits)
        max_val = (-z).clamp(min=0)
        loss = z - z*t + max_val + torch.log(torch.exp(-max_val) + torch.exp(-z-max_val))
        loss = loss * w
        loss = loss.mean()
        # Heng implementation
        #loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        #loss = loss.sum()/w.sum()
        return loss


class AngularErrorLoss(nn.Module):
    def __init__(self):
        super(AngularErrorLoss, self).__init__()

    def forward(self, logits, labels, weights):
        # probs = F.tanh(logits)
        probs = F.sigmoid(logits)
        probs = F.normalize(probs, p=2, dim=1) * 0.999999  # multiplying by 0.999999 prevents 'nan'!
        dot_prods = torch.sum(probs * labels, 1)
        dot_prods = dot_prods.clamp(-1, 1)
        error_angles = torch.acos(dot_prods)
        loss = torch.sum(error_angles * error_angles * weights)
        return loss


def dice_value(logits, labels, dice_w=None):
    C = labels.shape[1]
    dice = 0
    for d in range(C):
        w = 1/C if dice_w is None else dice_w[d]
        dice = dice + w * dice_value_1d(logits[:, d], labels[:, d])
    return dice


def dice_value_1d(logits, labels):
    probs = F.sigmoid(logits)
    num   = labels.shape[0]
    m1    = probs.view(num,-1)
    m2    = labels.view(num,-1)
    intersection = (m1 * m2)
    smooth = 1
    score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    return score.sum()/num


class MetricMonitor:
    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.metrics = collections.defaultdict(lambda: {'sum': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, value, n=None):
        if n is None:
            n = self.batch_size
        metric = self.metrics[metric_name]
        metric['sum'] += value * n
        metric['count'] += n
        metric['avg'] = metric['sum'] / metric['count']

    def get_avg(self, metric_name):
        return self.metrics[metric_name]['avg']

    def get_metric_values(self):
        return [(metric, values['avg']) for metric, values in self.metrics.items()]

    def __str__(self):
        return ' | '.join(
            f'{metric_name} {metric["avg"]:.6f}' for metric_name, metric in
            self.metrics.items()
        )


#------------------------------------------------------------------------------------------


def dice_index(pred_mask, gt_mask):
    m1    = pred_mask.reshape(-1)
    m2    = gt_mask.reshape(-1)
    intersection = (m1 * m2)
    smooth = 1
    return (2. * intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)


def aggregated_jaccard(pred_labels, gt_labels):
    C = 0
    U = 0
    pred_ls = list(range(1, pred_labels.max()+1))
    for gt_l in range(1, gt_labels.max()+1):
        gt_label = (gt_labels == gt_l).astype(np.int)
        max_iou = 0
        meeting_pred_ls = [p for p in np.unique(pred_labels * gt_label) if p != 0]
        if len(meeting_pred_ls) == 0:
            U += gt_label.sum()
        for pred_l in meeting_pred_ls:
            if pred_l not in pred_ls:
                continue
            pred_label = (pred_labels == pred_l).astype(np.int)
            intersection = (pred_label * gt_label).sum()
            union = pred_label.sum() + gt_label.sum() - intersection
            eps = 10**-6
            iou = intersection / (union+eps)
            if (iou > max_iou):
                max_iou = iou
                m_l = pred_l
                m_intersection = intersection
                m_union = union
        if max_iou != 0:
            C += m_intersection
            U += m_union
            pred_ls.remove(m_l)
    for pred_l in pred_ls:
        pred_label = (pred_labels == pred_l).astype(np.int)
        U += pred_label.sum()
    return C / U if U != 0 else 0
