import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

DEBUG_PLOT = False

def get_tta_output_rot(inputs, model):
    '''
    Makes predictions on 4 rotations of the inputs and averages them
    '''

    outputs = model(inputs)

    rots = 4

    # Predict non-rotated image
    outputs = model(inputs)

    # Predict the rotated images
    for rot in range(1, rots):
        # Incredibly slow way to do it, defeats the purpose of using gpu
        inputs_rot = torch.stack([
            transforms.ToTensor()(
            transforms.functional.rotate(
                transforms.ToPILImage()(inpt.cpu()),
                90*rot)
            ).to(torch.cuda.current_device())

            for inpt in inputs
        ])

        outputs_rot = model(inputs_rot)

        # Unrotate and add to the running average
        outputs += torch.stack([
            transforms.ToTensor()(
            transforms.functional.rotate(
                transforms.ToPILImage()(otpt.cpu()),
                -90*rot)
            ).to(torch.cuda.current_device())

            for otpt in outputs_rot
        ])


    # Taking an average, so divide by the number of rotations
    outputs /= rots

    return outputs

def get_tta_output_flip(inputs, model):
    '''
    Makes predictions on 8 flips (horizontal, vertical, diagonals, and combinations) of the inputs and averages them
    '''
    y_axis = 2
    x_axis = 3

    t1 = lambda t : t  # no transform
    t1i = t1

    t2 = lambda t : t.flip(x_axis)  # ⬌
    t2i = t2

    t3 = lambda t : t.flip(y_axis)  # ⬍
    t3i = t3

    t4 = lambda t : t2(t).flip(y_axis)  # ⬌⬍
    t4i = t4

    t5 = lambda t : t.transpose(y_axis,x_axis)  # ↗
    t5i = t5

    t6 = lambda t : t2(t).rot90(-1, (y_axis, x_axis))  # ↘
    t6i = t6

    t7 = lambda t : t5(t3(t))  # ⬍,↗
    t7i = lambda t : t3i(t5i(t))

    t8 = lambda t : t6(t3(t)) # ⬍,↘
    t8i = lambda t : t3i(t6i(t))

    transforms = [t1, t2, t3, t4, t5, t6, t7, t8]
    transforms_inverse = [t1i, t2i, t3i, t4i, t5i, t6i, t7i, t8i]




    # Transform inputs
    transformed_inputs = []
    for i,t in enumerate(transforms):
        transformed_inputs.append(t(inputs))





    # plot transformed images
    if DEBUG_PLOT:
        fig=plt.figure(figsize=(32, 32))
        columns = len(transforms)
        rows = len(inputs)

        for i, t_images in enumerate(transformed_inputs):  # for all transformed batches
            for j, t_image in enumerate(t_images):  # for all images in batch
                fig.add_subplot(rows, columns, (i+1) + j*len(transformed_inputs) )
                plt.imshow(t_image.cpu()[0]) # plot only single color of image    
        plt.show()



    # Make predictions
    transformed_predictions = []
    for i,inpt in enumerate(transformed_inputs):
        transformed_predictions.append(model(inpt))


    # plot (transformed) predictions
    if DEBUG_PLOT:
        fig=plt.figure(figsize=(32, 32))
        columns = len(transforms)
        rows = len(inputs)

        for i, t_images in enumerate(transformed_predictions):  # for all transformed batches
            for j, t_image in enumerate(t_images):  # for all images in batch
                fig.add_subplot(rows, columns, (i+1) + j*len(transformed_inputs) )
                plt.imshow(t_image.cpu()[0]) # plot only single color of image    
        plt.show()


    # Inverse transform the predictions
    inverted_predictions = []
    for i,(t, pred) in enumerate(zip(transforms_inverse, transformed_predictions)):
        inverted_predictions.append(t(pred))



    # plot (inverted) predictions
    if DEBUG_PLOT:
        fig=plt.figure(figsize=(32, 32))
        columns = len(transforms)
        rows = len(inputs)

        for i, t_images in enumerate(inverted_predictions):  # for all transformed batches
            for j, t_image in enumerate(t_images):  # for all images in batch
                fig.add_subplot(rows, columns, (i+1) + j*len(transformed_inputs) )
                plt.imshow(t_image.cpu()[0]) # plot only single color of image    
        plt.show()


    # Taking an average, so sum and divide by the number of transforms
    out = torch.stack(inverted_predictions, dim=0).sum(dim=0)
    out /= len(inverted_predictions)

    return out