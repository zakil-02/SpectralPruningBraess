import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
    parser.add_argument('--method', type=str, help='Max/Min/Add/Delete/FoSR/SDRF')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset to download')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
    parser.add_argument('--model', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN'], help='Model to use')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in GATv2')
    parser.add_argument('--existing_graph', type=str,default=None, help='.pt file')
    parser.add_argument('--out', type=str, default='log.csv', help='name of log file')
    parser.add_argument('--max_iters', type=int, default=10, help='maximum number of edge change iterations')
    parser.add_argument('--comm_delete', type=float, default=0.0, help='fraction of inter-community edges to delete')
    parser.add_argument('--comm_add', type=float, default=0.0, help='fraction of edges to add relative to current edge count')
    #parser.add_argument('--removal_bound', type=float, default=0.95, help='removal bound for SDRF')
    #parser.add_argument('--tau', type=int, default=163, help='Temperature for SDRF')
    parser.add_argument('--update_period', type=int, default=1, help='Times to recalculate criterion')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
    parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
    parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
    parser.add_argument('--device',type=str,default='cuda',help='Device to use')
    parser.add_argument('--seed',type=int,default=3164711608,help='Seed to use')
    parser.add_argument('--splits', type=int, default=50, help='Number of splits for training and testing')
    parser.add_argument('--num_train', type=int, default=20, help='Number of training nodes per class')
    parser.add_argument('--num_val', type=int, default=500, help='Number of validation nodes per class')
    return parser.parse_args()

#args, _ = parser.parse_known_args()
# if args.method == 'community_rewiring':
#     parser.add_argument('--comm_delete', type=float, default=0.1, help='fraction of inter-community edges to delete')
#     parser.add_argument('--comm_add', type=float, default=0.1, help='fraction of edges to add relative to current edge count')
