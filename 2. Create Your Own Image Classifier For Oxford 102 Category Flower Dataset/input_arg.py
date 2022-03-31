import argparse


def train_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir',
                        type=str,
                        help='path to the directory of the data')
    parser.add_argument('--save_dir',
                        type=str,
                        default='checkpoint.pth',
                        help='path to save checkpoints')
    parser.add_argument('--arch',
                        type=str,
                        default='mobilenet_v2',
                        help='CNN Model Architecture. Options are mobilenet_v2 and densenet')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Set learning rate')
    parser.add_argument('--hidden_units',
                        type=int,
                        default=0,
                        help='Set hidden units')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Set epochs')
    parser.add_argument('--gpu',
                        action='store_true',
                        dest='use_gpu',
                        help='Use GPU for training')

    return parser.parse_args()


def test_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('image',
                        type=str,
                        help='path to an image')
    parser.add_argument('checkpoint',
                        type=str,
                        help='path to checkpoint')
    parser.add_argument('--top_k',
                        type=int,
                        default=1,
                        help='wanted top-k likely classes')
    parser.add_argument('--category_names',
                        type=str,
                        default=None,
                        help='path to mapping file')
    parser.add_argument('--gpu',
                        action='store_true',
                        dest='use_gpu',
                        help='Use GPU for training')

    return parser.parse_args()
