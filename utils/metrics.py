from common import *


def criterion_fn(logits, labels, use_weight=False):
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
        weights = torch.ones(a.shape, requires_grad=True).cuda(async=True)
        w0 = weights.sum()
        weights = weights + border*2
        w1 = weights.sum()
        weights = weights * (w0 / w1)
        weights = weights.repeat((C, 1, 1)).reshape(labels.shape)
    else:
        weights = torch.ones(labels.shape, requires_grad=True).cuda(async=True)

    l = WeightedBCELoss2d()(logits, labels, weights) + \
        WeightedSoftDiceLoss()(logits[:, 0], labels[:, 0], weights[:, 0]) + \
        WeightedSoftDiceLoss()(logits[:, 1], labels[:, 1], weights[:, 1])

    return l


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


def dice_value(logits, labels):
    probs = F.sigmoid(logits)
    num   = labels.shape[0]
    m1    = probs.view(num,-1)
    m2    = labels.view(num,-1)
    intersection = (m1 * m2)
    smooth = 1
    score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    return score.sum()/num


def aggregated_jaccard(pred_labels, gt_labels):
    C = 0
    U = 0
    pred_ls = range(pred_labels.max())
    used_ls = set()
    for gt_l in range(gt_labels.max()):
        gt_label = (gt_labels == (gt_l+1)).astype(np.int)
        max_iou = 0
        for pred_l in pred_ls:
            pred_label = (pred_labels == (pred_l+1)).astype(np.int)
            intersection = (pred_label * gt_label).sum()
            union = pred_label.sum() + gt_label.sum() - intersection
            eps = 10**-6
            iou = intersection / (union+eps)
            if (iou > max_iou):
                max_iou = iou
                m_l = pred_l
                m_intersection = intersection
                m_union = union
        if (max_iou != 0):
            C += m_intersection
            U += m_union
            used_ls.add(m_l)
    for pred_l in set(pred_ls) - used_ls:
        pred_label = (pred_labels == (pred_l + 1)).astype(np.int)
        U += pred_label.sum()
    return C / U