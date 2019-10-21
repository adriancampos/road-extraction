import torch
import torch.nn as nn

epsilon = 1e-5

# from https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge/blob/master/loss.py
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
#         self.bce_loss = nn.BCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    # NOTE: pred and truth are swapped from original implementation (everywhere in this class)
    def soft_dice_coeff(self, y_pred, y_true):
        y_true = y_true.float()
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + epsilon) / (i + j + epsilon)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss
        
    def __call__(self, y_pred, y_true):
        # normalize. Shouldn't do here.
#         y_pred = y_pred/torch.max(y_pred)
#         y_true = y_true/torch.max(y_true)
#         y_pred = y_pred.float()
        y_true = y_true.float()
    
#         print("minimum", y_pred.min())
        # shift min from some value to 0
        y_pred = y_pred - y_pred.min()
#         print("sc_minimum", y_pred.min())

                
        # scale pred from [0,255] to [0,1]
        # todo: chekc if y.max() is zero. If so, no need to scale (though is 0/0 okay?)
        y_pred = y_pred/y_pred.max()
        
        # scale true from [0,255] to [0,1]
        y_true = y_true/y_true.max()
        
#        # print approx 1% of logs        
#         print("y_pred",y_pred.shape,y_pred)
#         print("y_true",y_true.shape,y_true)
    
#         print("pred min", y_pred.min(), "pred max", y_pred.max())
#         print("true min", y_true.min(), "true max", y_true.max())

        a =  self.bce_loss(y_pred, y_true)
#         print("bce_loss", a)
        b =  self.soft_dice_loss(y_pred, y_true)
#         print("dice_loss",b)
        return a + b