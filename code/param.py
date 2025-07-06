import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=21, help='Random seed.') #初始为0
    parser.add_argument('--k_folds', type=int, default=5, help='k_folds.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')  # weight_decay=5e-4# 权重衰减
    parser.add_argument('--hidden_channels', type=int, default=128,  help='Number of hidden units.')
    parser.add_argument("--out_channels", type=int, default=32, help="out-channels. Default is 8.")
    parser.add_argument("--num_intervals", type=int, default=5, help=".")
    parser.add_argument("--k", type=int, default=3, help="k")
    parser.add_argument("--num_gcn_layers", type=int, default=1, help="gcn_layers.")
    parser.add_argument("--gat_heads", type=int, default=1, help="gat_heads.")
    parser.add_argument("--num_mamba_layers", type=int, default=1, help="num_mamba_layers.")
    parser.add_argument("--d_state", type=int, default=16, help="mamba")
    parser.add_argument("--expand", type=int, default=2, help="mamba")
    parser.add_argument("--num_kan_layers", type=int, default=1, help="layers2.")
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--nheads', type=float, default=1, help='Number of head attentions.')
    parser.add_argument('--Lambda', type=float, default=0.05, help='')

    return parser.parse_args()