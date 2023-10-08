import torch
from src.data.get_data import DataLoader
from src.models.ShallowNetworks.Shallow import ShallowEmbeddings
from src.models.GNN.GNN import GNN
import argparse
from src.models.utils import setup_argparser


def argparser():
    parser = argparse.ArgumentParser(
        description="This script is used for running all the experiments regarding this project."
    )

    # Task and dataset
    parser.add_argument(
        "--task",
        default="NodeClassification",
        type=str,
        required=False,
        help="Choose which task to perform either NodeClassification or LinkPrediction",
    )
    parser.add_argument(
        "--dataset",
        default="ogbn-arxiv",
        type=str,
        required=False,
        help="Choose which dataset to use",
    )

    # Model
    # parser.add_argument('--model_architecture',default='Shallow',required=False)
    parser.add_argument(
        "--model",
        default="Node2Vec",
        type=str,
        required=False,
        help="Choose which model to use",
    )
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)

    # Training arguments
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        required=False,
        help="Choose which device to use",
    )
    parser.add_argument(
        "--batch_size",
        default="256",
        type=str,
        required=False,
        help="Choose which batch size to use",
    )
    parser.add_argument(
        "--lr", default="0.001", type=str, required=False, help="Choose which lr to use"
    )
    parser.add_argument(
        "--num_workers",
        default="0",
        type=int,
        required=False,
        help="Choose which num_workers to use",
    )

    args = parser.parse_args()
    setup_argparser(parser, args)
    return args


def main():
    args = argparser()
    device = args.device
    # Get Data
    data = DataLoader(task_type=args.task, dataset=args.dataset).get_data()

    model = GNN(
        GNN_type=args.model,
        task=args.task,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        dropout=args.dropout,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
