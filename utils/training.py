import os
import sys
import math
import string
import random
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils
from . import tta as tta_util

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

THRESHOLD = 0.5


def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def threshold(preds):
    assert THRESHOLD == 0.5
    if preds.dtype == torch.float32:
        return preds.round().int()
    else:
        return preds

def get_predictions(output_batch):
    return threshold(output_batch.data.cpu()[0])
#     print(output_batch.shape)
#     print(output_batch)
#     bs,c,h,w = output_batch.size()
#     tensor = output_batch.data
#     values, indices = tensor.cpu().max(1)
#     indices = indices.view(bs,h,w)
#     return indices



def error(preds, targets):
    return 1 - jaccard(preds, targets)
#     assert preds.size() == targets.size()
#     bs,h,w = preds.size()
# #     print(bs,h,w)
#     n_pixels = bs*h*w
#     incorrect = preds.ne(targets).cpu().sum()
# #     print("incorrect",incorrect.item())
#     err = incorrect.item()/n_pixels
# #     print("err",err)
#     return round(err,5)

# Matches standard definition of jaccard (and definition on codelab)  
def jaccard(preds, targets):
#     preds = preds.round().int()
#     targets = targets.round().int()
    preds = threshold(preds)
    
    intersection = preds & targets
    union = preds | targets
    
#     print("p", preds.sum())
#     print("t", targets.sum())
#     print("i", intersection.sum())
#     print("u", union.sum())

    # If the union of preds and targets are both empty, we define jaccard to be 1
    if (union.sum() == 0):
        return 1


    return float(intersection.sum())/float(union.sum())


def train(model, trn_loader, optimizer, criterion, epoch, MAX_BATCH_PER_CARD=1, batch_size=16):
    model.train()
    trn_loss = 0
    trn_error = 0
    
    multiplier = batch_size / (MAX_BATCH_PER_CARD * torch.cuda.device_count())
    print("torch.cuda.device_count()",torch.cuda.device_count())
    print("multiplier",multiplier)
    
    accumulator = multiplier
    batchloss = 0
    batchcount = 0
    
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(trn_loader):         
        if (len(trn_loader) - i + 1) < batch_size:
            # out of room for another batch
            break
        
        
        
#         print("idx",idx)
#         print("data",data[0].shape,data[1].shape)
#         img_utils.view_image(data[0][0])
#         img_utils.view_image(data[1])

        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        
        targets = targets.unsqueeze(0)

        output = model(inputs)
#         print("targets.shape",targets.shape)
#         print("outputs.shape",output.shape)

#         print("output.shape",output.shape)
#         print("target.shape",targets.shape)

#         print("output.max()",output.max())
#         print("target.max()",targets.max())
        loss = criterion(output, targets)
        loss /= multiplier
        loss.backward()
        
#         print(loss, batchloss)

        if not torch.isnan(loss):
            batchloss += loss.item()
        else:
            print("WARNING: Loss is nan")
        accumulator -= 1
        
        if accumulator == 0:
            # after accumulation of batch, we can step into direction of gradient
            optimizer.step()
            optimizer.zero_grad()
            accumulator = multiplier
            batchcount += 1
            trn_loss += batchloss
            batchloss = 0            


#         print(loss.item())
#         trn_loss += loss.item()
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())

#     trn_loss /= len(trn_loader)
    trn_loss /= batchcount
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1, use_tta=False):
    model.eval()
    test_loss = 0
    test_error = 0
    jacc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data.cuda())
            target = Variable(target.cuda())
            target = target.unsqueeze(0)

            if use_tta:
                output = tta_util.get_tta_output_rot(data, model)
            else:
                output = model(data)
    #         test_loss += criterion(output, target).data[0]

#             print("targets.shape",target.shape)
#             print("outputs.shape",output.shape)

            test_loss += criterion(output, target).item()
            pred = get_predictions(output)
            test_error += error(pred, target.data.cpu())
            jacc += jaccard(pred, target.data.cpu())
    #         print(test_error)
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    jacc /= len(test_loader)
    return test_loss, test_error, jacc

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
#     input_loader.batch_size = 1
    predictions = []
    model.eval()
    with torch.no_grad():
        for input, target in input_loader:
            data = Variable(input.cuda())
            label = Variable(target.cuda())
            output = model(data)
            pred = get_predictions(output)
            predictions.append([input,target,pred])
        return predictions
    
def predict_validation(model, input_loader, n_batches=1):
#     input_loader.batch_size = 1
    predictions = []
    model.eval()
    with torch.no_grad():
        for input,path in input_loader:
            data = Variable(input.cuda())
            output = model(data)
            pred = get_predictions(output)
            predictions.append([input,pred,path])
        return predictions

def view_sample_predictions(model, loader, n, criterion=None):
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        data = Variable(inputs.cuda())
        label = Variable(targets.cuda())
        output = model(data)
    #     output = (output+8)*(255/8)
        pred = get_predictions(output)
    #     print("target",label,label.shape)
    #     print("output",output[0],output[0].shape)
    #     print("max(output)",output[0].max())
    #     print("min(output)",output[0].min())
    #     print("predic",pred,pred.shape)
        batch_size = inputs.size(0)
        for i in range(min(n, batch_size)):
            img_utils.view_image(inputs[i])
            img_utils.view_annotated(targets[i])
            img_utils.view_annotated(pred[i])

        if criterion:
            return criterion(output, label).item(), error(pred, label.data.cpu()), jaccard(pred, label.data.cpu())


def get_sample_predictions(model, loader, n, criterion=None, idx=None):
    with torch.no_grad():
        if idx:
            inputs, targets = loader.dataset[idx]
        else:
            inputs, targets = next(iter(loader))
        data = Variable(inputs.cuda())
        label = Variable(targets.cuda()).unsqueeze(0)
        output = model(data)
        # TODO: I don't think I need to do get_predictions. Or modify it to just do a threshold. That's all. (Same for view_sample_predictions)
        pred = get_predictions(output)
        batch_size = inputs.size(0)
    #     for i in range(min(n, batch_size)):
    #         img_utils.view_image(inputs[i])
    #         img_utils.view_annotated(targets[i])
    #         img_utils.view_annotated(pred[i])

        if criterion:
            return inputs, targets, pred, criterion(output, label).item(), error(pred, label.data.cpu()), jaccard(pred, label.data.cpu())

