import os

import hydra

from neural_bezier.cnn_model_train import train


@hydra.main(config_path='configs/config.yaml', strict=True)
def main(config):
    print("Working directory : {}".format(os.getcwd()))
    print(config.pretty())
    train(config)


if __name__ == '__main__':
    main()
