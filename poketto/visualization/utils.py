import matplotlib.pyplot as plt
import numpy as np

def matplotlib_imshow(img):
    if img.ndim < 3 or img.shape[0] == 1:
        one_channel = True
        img = img.mean(dim=0)
    else:
        one_channel = False
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))