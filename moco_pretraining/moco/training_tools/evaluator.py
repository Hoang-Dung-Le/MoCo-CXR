import argparse
import os
import random
import time
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.special import softmax

from .meters import AverageMeter
from .meters import ProgressMeter
from .combiner import detach_tensor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

'''
def pred_accuracy(output, target, k):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    output = detach_tensor(output)
    target = detach_tensor(target)

    batch_size = target.size(0)

    argsorted_out = np.argsort(output)[:,-k:]
    return np.asarray(np.any(argsorted_y.T == target, axis=0).mean(dtype='f')),

    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]           # Seems like we only want the 1st
'''


def decorator_detach_tensor(function):
    def wrapper(*args, **kwargs):
        # TODO Find a simple way to handle this business ...
        # If is eval, or if fast debug, or
        # is train and not heavy, or is train and heavy
        output = detach_tensor(args[0])
        target = detach_tensor(args[1])
        args = args[2:]

        result = function(output, target, *args, **kwargs)
        # result = computeAUROC(output, target)
        return result
    return wrapper

# @decorator_detach_tensor
# def topk_acc(output, target, k):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     argsorted_out = np.argsort(output)[:,-k:]
#     matching = np.asarray(np.any(argsorted_out.T == target, axis=0))
#     return matching.mean(dtype='f')


# @decorator_detach_tensor
# def compute_auc_binary(output, target):
#     #assuming output and target are all vectors for binary case
#     try:
#         o = softmax(output, axis=1)
#         auc = roc_auc_score(target, o[:,1])
#     except:
#         return -1
#     return auc
@decorator_detach_tensor
def computeAUROC(dataPRED, dataGT, classCount=14):

    outAUROC = []
    fprs, tprs, thresholds = [], [], []
    
    for i in range(classCount):
        try:
            # Apply sigmoid to predictions
            pred_probs = torch.sigmoid(torch.tensor(dataPRED[:, i]))
            # pred_probs = dataPRED[:, i]
            # print(pred_probs)
            # print(pred_probs)
            # print(dataGT[:, i].shape)
            # print(dataGT)
            # print("_________________________")
            # print(pred_probs)
            # Calculate ROC curve for each class
            fpr, tpr, threshold = roc_curve(dataGT[:, i], pred_probs)
            roc_auc = roc_auc_score(dataGT[:, i], pred_probs)
            outAUROC.append(roc_auc)

            # Store FPR, TPR, and thresholds for each class
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)
        except:
            outAUROC.append(0.)

    auc_each_class_array = np.array(outAUROC)

    print("each class: ",auc_each_class_array)
    # Average over all classes
    result = np.average(auc_each_class_array[auc_each_class_array != 0])
    # print(result)
    plt.figure(figsize=(10, 8))  # Đặt kích thước hình ảnh chung

    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], label=f'Class {i} (AUC = {outAUROC[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for all Classes')
    plt.legend()

    output_file = f'./roc_auc.png'  # Đường dẫn lưu ảnh

    # Lưu hình xuống file
    plt.savefig(output_file)

    return result

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.reshape(1, -1).expand_as(pred))
#     return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class Evaluator:

    def __init__(self, model, loss_func, metrics, loaders, args):

        self.model = model
        self.loss_func = loss_func
        self.metrics = metrics
        self.loaders = loaders
        self.args = args

        self.metric_best_vals = {metric: 0 for metric in self.metrics}


    def evaluate(self, eval_type, epoch):

        print(f'==> Evaluation for {eval_type}, epoch {epoch}')

        loader = self.loaders[eval_type]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        metric_meters = {metric: AverageMeter(metric, self.metrics[metric]['format']) \
                                                    for metric in self.metrics}
        list_meters = [metric_meters[m] for m in metric_meters]

        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, *list_meters],
            prefix=f'{eval_type}@Epoch {epoch}: ')

        # switch to evaluate mode
        self.model.eval()
        all_output = []
        all_gt = []

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                if self.args.gpu is not None:
                    images = images.cuda(self.args.gpu, non_blocking=True)
                target = target.cuda(self.args.gpu, non_blocking=True)
                all_gt.append(target.cpu())        
                # print("len cua tar get", len(target))
                # compute output
                output = self.model(images)
                all_output.append(output.cpu())
                
                loss = self.loss_func(output, target)
                
                # JBY: For simplicity do losses first
                losses.update(loss.item(), images.size(0))
                # print(output, "+++++++++++++++++++++++++++++++++++++++",target)
            
                for metric in self.metrics:
                    # args = [target, output, *self.metrics[metric]['args']]    
                    args = [target, output]  
                    metric_func = globals()[self.metrics[metric]['func']]
                    # result = metric_func(*args)

                    auc_each_class = metric_func(*args)
                    
                    auc_each_class_array = np.array(auc_each_class)
                    
                    # result = np.average(auc_each_class_array[auc_each_class_array != 0])
                    if np.any(auc_each_class_array != 0):
                        result = np.average(auc_each_class_array[auc_each_class_array != 0])
                    else:
                        result = 0.0
                    
                    
                    metric_meters[metric].update(result, images.size(0))
            

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #    .format(top1=top1, top5=top5))
            progress.display(i + 1)

        all_output = np.concatenate(all_output)
        # _, preds = torch.max(all_output, 1)
        # for i in all_output:
        # y = [torch.max(all_output).item()]
        # import tensorflow as tf
        # for tensor in all_output:
        #     print(torch.Tensor(tensor))

        y = [torch.argmax(torch.Tensor(tensor)).item() for tensor in all_output]
        
        all_gt = np.concatenate(all_gt)
        # print(y)
        # print("==============================")
        
        # print(all_gt)
        # 1/0
        # from torchmetrics.functional.classification import BinaryConfusionMatrix
        # matrix = multiclass_confusion_matrix(all_output, all_gt, num_classes=2)
        # cf_matrix = confusion_matrix(all_gt, y)
        # print(cf_matrix)
        # print(classification_report(all_gt, y))
        # print(accuracy(all_gt, y))
        # from sklearn.metrics import ConfusionMatrixDisplay
        # disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)

        # disp.plot(cmap=plt.cm.Blues)
        # plt.show()
        for metric in self.metrics:
            args = [all_output, all_gt, *self.metrics[metric]['args']]    
            metric_func = globals()[self.metrics[metric]['func']]
            result = metric_func(*args)
            
            metric_meters[metric].update(result, images.size(0))
            # print("---")
            # print(metric_meters[metric].avg)
            print(self.metric_best_vals[metric])
            self.metric_best_vals[metric] = max(metric_meters[metric].avg[0],
                                                self.metric_best_vals[metric])
        print("====================================", self.metrics)
        # progress.display(i + 1, summary=True)