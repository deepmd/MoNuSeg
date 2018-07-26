from common import *


def criterion_MSELoss(input, target):
    return F.mse_loss(input, target)


def criterion_AngularError(input, target, area, vectors_count=2):
    C = target.shape[1]
    vectors_dims = C // vectors_count
    eps = 10 ** -6
    weight = torch.gt(area, 0).float() / torch.sqrt(area + eps)
    # weight = weight / (weights.max()+eps)
    loss = 0
    for i in range(0, C, vectors_dims):
        loss += angular_error(input[:, i:i+vectors_dims], target[:, i:i+vectors_dims], weight)

    return loss


def criterion_BCE_SoftDice(input, target, dice_w=None, use_weight=False):
    # compute weight
    if use_weight:
        batch_size, C, H, W = target.shape
        if H <= 128:
            kernel_size = 11
        elif H <= 256:
            kernel_size = 21
        elif H <= 512:
            kernel_size = 21
        else:
            kernel_size = 41
        a = F.avg_pool2d(target[:, 0], kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        border = (a.ge(0.01) * a.le(0.99)).float()
        weight = torch.ones(a.shape).cuda(async=True)
        w0 = weight.sum()
        weight = weight + border*2
        w1 = weight.sum()
        weight = weight * (w0 / w1)
        weight = weight.repeat((C, 1, 1)).reshape(target.shape)
    else:
        weight = None

    loss = F.binary_cross_entropy_with_logits(input, target, weight) + \
           dice_loss_with_logits(input, target, weight, dice_w)

    return loss


def dice_loss_with_logits(input, target, weight=None, dice_w=None):
    probs = F.sigmoid(input)
    C = target.shape[1]
    loss = 0
    w = [1/C]*C if dice_w is None else dice_w
    for d in range(C):
        loss += w[d] * (1 - soft_dice_1d(probs[:, d], target[:, d], None if weight is None else weight[:, d]))
    return loss



def angular_error(input, target, weight):
    dot_prods = torch.sum(input * target, 1)
    dot_prods = dot_prods.clamp(-1, 1)
    error_angles = torch.acos(dot_prods)
    # loss = torch.sum(error_angles * error_angles * weight)
    loss = torch.mean(error_angles * error_angles * weight)
    return loss


def dice_value(input, target, dice_w=None):
    probs = F.sigmoid(input)
    C = target.shape[1]
    dice = 0
    w = [1 / C] * C if dice_w is None else dice_w
    for d in range(C):
        dice += w[d] * soft_dice_1d(probs[:, d], target[:, d])
    return dice


def soft_dice_1d(input, target, weight=None):
    num = target.shape[0]
    m1 = input.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2)
    smooth = 1
    if weight is None:
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    else:
        w = weight.view(num, -1)
        w2 = w * w
        score = (2. * (w2 * intersection).sum(1) + smooth) / ((w2 * m1).sum(1) + (w2 * m2).sum(1) + smooth)
    return score.sum() / num


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
    for gt_l in [l for l in np.unique(gt_labels) if l != 0]:
        gt_label = (gt_labels == gt_l).astype(np.int)
        meeting_pred_ls = [p for p in np.unique(pred_labels * gt_label) if p != 0]
        if len(meeting_pred_ls) == 0:
            U += gt_label.sum()
        max_iou = 0
        for pred_l in meeting_pred_ls:
            pred_label = (pred_labels == pred_l).astype(np.int)
            intersection = (pred_label * gt_label).sum()
            union = pred_label.sum() + gt_label.sum() - intersection
            iou = intersection / union
            if iou > max_iou:
                max_iou = iou
                m_label = pred_label
                m_intersection = intersection
                m_union = union
        if max_iou != 0:
            C += m_intersection
            U += m_union
            pred_labels *= 1 - m_label
    U += (pred_labels > 0).sum()
    return C / U


# def aggregated_jaccard(pred_labels, gt_labels):
#     C = 0
#     U = 0
#     for gt_l in [l for l in np.unique(gt_labels) if l != 0]:
#         gt_label = (gt_labels == gt_l).astype(np.int)
#         meeting_pred_ls = [p for p in np.unique(pred_labels * gt_label) if p != 0]
#         max_iou = 0
#         if len(meeting_pred_ls) == 0:
#             m_label = (pred_labels == 1).astype(np.int)
#             m_intersection = 0
#             m_union = m_label.sum() + gt_label.sum()
#         for pred_l in meeting_pred_ls:
#             pred_label = (pred_labels == pred_l).astype(np.int)
#             intersection = (pred_label * gt_label).sum()
#             union = pred_label.sum() + gt_label.sum() - intersection
#             iou = intersection / union
#             if iou > max_iou:
#                 max_iou = iou
#                 m_label = pred_label
#                 m_intersection = intersection
#                 m_union = union
#         C += m_intersection
#         U += m_union
#         pred_labels *= 1 - m_label
#     U += (pred_labels > 0).sum()
#     return C / U



