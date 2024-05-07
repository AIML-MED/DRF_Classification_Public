import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import socket


figure_no = 0

if socket.gethostname() == 'zliao-AIML':
    matplotlib.use('TkAgg')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def plot(img, img_no=0, channel_no=0, normalize=False, title=None):
    global figure_no
    if str(type(img)) == "<class 'torch.Tensor'>":
        img = img.detach().cpu().numpy()

    if len(img.shape) == 4:
        print('plot using img no: {}'.format(img_no))
        img = img[img_no]

    if img.shape[0] == 1 and len(img.shape) == 3:
        img = np.squeeze(img)
    elif img.shape[0] == 3 and len(img.shape) == 3:
        img = np.moveaxis(img, (0, 1, 2), (2, 0, 1))
    elif img.shape[-1] == 3 and len(img.shape) == 3:
        img = img
    elif len(img.shape) == 3:
        img = img[channel_no]

    # if len(np.unique(img)) < 256:
    #     if np.max(img) < 1:
    #         img = img*255
    #
    #     img = img.astype(np.uint8)

    if len(img.shape) == 2:
        fig=plt.figure(num=figure_no)
        plt.imshow(img, cmap='gray')
        fig.suptitle(title)
    else:
        plt.figure(num=figure_no)
        plt.imshow(img)
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, num=figure_no)
        # fig.suptitle(title)
        # if normalize:
        #     img = img - np.min(img, axis=(1, 2), keepdims=True)
        #     img = img / np.max(img, axis=(1, 2), keepdims=True)
        #
        # ax[0, 0].imshow(img)
        # ax[0, 0].set_title("All channels")
        #
        # img_r = img.copy() * 0 + img.min()
        # img_r[:, :, 0] = img[:, :, 0]
        # ax[0, 1].imshow(img_r)
        # ax[0, 1].set_title('R')
        #
        # img_g = img.copy() * 0 + img.min()
        # img_g[:, :, 1] = img[:, :, 1]
        # ax[1, 0].imshow(img_g)
        # ax[1, 0].set_title('G')
        #
        # img_b = img.copy() * 0 + img.min()
        # img_b[:, :, 2] = img[:, :, 2]
        # ax[1, 1].imshow(img_b)
        # ax[1, 1].set_title('B')

    # plt.pause(0.5)
    plt.show()
    figure_no += 1
