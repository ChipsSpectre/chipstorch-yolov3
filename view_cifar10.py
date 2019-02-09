import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

"""
    Utility script to view the results of cifar10 net.
"""
if __name__ == "__main__":
    n = 9
    if len(sys.argv) != 2:
        print("Usage: view_cifar10.py <path_to_cifar10_output>")
    path = sys.argv[1]

    label_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    for i in range(n):
        print(i)
        module = torch.jit.load(os.path.join(path, "image{}.pt".format(i)))
        images = list(module.parameters())[0]

        image = images.detach().cpu().numpy()
        image = np.rollaxis(image, 2)
        image = np.rollaxis(image, 2)

        axis = plt.subplot(3, 3, 1 + i)
        plt.imshow(image, cmap="gray")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

        module = torch.jit.load(os.path.join(path, "target{}.pt".format(i)))
        target = list(module.parameters())[0].detach().cpu().numpy()
        module = torch.jit.load(os.path.join(path, "prediction{}.pt".format(i)))
        prediction = list(module.parameters())[0].detach().cpu().numpy()

        prediction = np.argmax(prediction)

        own = int(prediction)

        plt.title("{} vs. {}".format(label_map[int(prediction)], label_map[int(target)]))
    plt.show()
