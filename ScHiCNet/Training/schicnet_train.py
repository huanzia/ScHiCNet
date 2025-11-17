import sys

sys.path.append('.')
sys.path.append('../')

from experi.train_schicnet import schicnet_trainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train schicnet for Hi-C super-resolution')
    parser.add_argument('-e', '--epoch', type=int, default=60)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-n', '--celln', type=int, default=1)
    parser.add_argument('-l', '--celline', type=str, default='Mouse')
    parser.add_argument('-p', '--percent', type=float, default=0.75)
    args = parser.parse_args()

    model = schicnet_trainer(epoch=args.epoch, batch_s=args.batch_size,
                         cellN=args.celln, celline=args.celline, percentage=args.percent)
    model.fit_model()
