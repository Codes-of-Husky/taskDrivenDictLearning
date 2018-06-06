import numpy as np
import matplotlib.pyplot as plt
import pdb

#Order defines the order in weights_matrix for num_weights, y, x, f
def plot_weights(weights_matrix, outPrefix, order=[0, 1, 2, 3]):
    assert(weights_matrix.ndim == 4)
    num_weights = weights_matrix.shape[order[0]]
    patch_y = weights_matrix.shape[order[1]]
    patch_x = weights_matrix.shape[order[2]]
    patch_f = weights_matrix.shape[order[3]]

    if(order == [0, 1, 2, 3]):
        permute_weights = weights_matrix
    else:
        permute_weights = weights_matrix.copy()
        permute_weights = np.transpose(weights_matrix, order)

    subplot_x = int(np.ceil(np.sqrt(num_weights)))
    subplot_y = int(np.ceil(num_weights/float(subplot_x)))

    outWeightMat = np.zeros((patch_y*subplot_y, patch_x*subplot_x, patch_f)).astype(np.float32)

    #Normalize each patch individually
    for weight in range(num_weights):
        weight_y = int(weight/subplot_x)
        weight_x = weight%subplot_x

        startIdx_x = weight_x*patch_x
        endIdx_x = startIdx_x+patch_x

        startIdx_y = weight_y*patch_y
        endIdx_y = startIdx_y+patch_y

        weight_patch = permute_weights[weight, :, :, :].astype(np.float32)

        #Set mean to 0
        weight_patch = weight_patch - np.mean(weight_patch)
        scaleVal = np.max([np.fabs(weight_patch.max()), np.fabs(weight_patch.min())])
        weight_patch = weight_patch / (scaleVal+1e-6)
        weight_patch = (weight_patch + 1)/2

        outWeightMat[startIdx_y:endIdx_y, startIdx_x:endIdx_x, :] = weight_patch

    if(patch_f == 1):
        outWeightMat = np.tile(outWeightMat, [1, 1, 3])
    if(patch_f == 1 or patch_f == 3):
        fig = plt.figure()
        plt.imshow(outWeightMat)
        plt.savefig(outPrefix+".png")
        plt.close(fig)
    else:
        for f in range(patch_f):
            fig = plt.figure()
            outMat = outWeightMat[:, :, f]
            plt.imshow(outMat, cmap="gray")
            plt.savefig(outPrefix+ "f_" + str(f) + ".png")
            plt.close(fig)

    fig = plt.figure()
    plt.hist(weights_matrix.flatten(), 50)
    plt.savefig(outPrefix+ "_hist.png")
    plt.close(fig)
