import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
import sys
import os
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils import clean_state_dict, create_dummy_prediction, setup_logging
from datasets.test_dataset import TestDataset
from models.network import GeoLocalizationNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(args, eval_ds, model):
    """Compute features of the given dataset and compute the recalls."""
    all_features = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            all_features[indices.numpy(), :] = features

        logging.debug("Extracting queries features for evaluation/testing")
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            all_features[indices.numpy(), :] = features

    database_features = all_features[:eval_ds.database_num]
    queries_features = all_features[eval_ds.database_num:]
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    logging.debug("Calculating recalls")
    _, prediction = faiss_index.search(queries_features, max(args.recall_values))
    np.save(f'{args.output_folder}/prediction.npy', prediction)
    if args.dataset_name == 'msls' and args.split == 'test':
        create_dummy_prediction(args, eval_ds, prediction)
        return
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(prediction):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    logging.info(f"Global retrieval recalls: {recalls_str}")
    return recalls, recalls_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14",
                        choices=["ResNet50", "dinov2_vitb14"])
    parser.add_argument("--test_method", type=str, default="DSFormer")
    parser.add_argument("--fc_output_dim", type=int, default=512)
    parser.add_argument("--dataset_name", type=str, default='msls',
                        choices=['msls', 'pitts30k', 'pitts250k', 'tokyo247', 'nordland'],
                        help='Name of the tested dataset.')
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test', None])
    parser.add_argument("--dataset_folder", type=str, default='/mnt/data', help="path/to/datasets")
    parser.add_argument("--resize", type=int, default=[320, 320], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument("--soft_positives_dist_threshold", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 15, 20, 25, 100],
                        help="values for recall (e.g. recall@1, recall@5)")

    args = parser.parse_args()

    if args.backbone == 'ResNet50':
        num_patches = [400, 100]
        args.resize = [320, 320]
        num_layers = 3
    elif args.backbone == 'dinov2_vitb14':
        num_patches = [529, 529]
        args.resize = [322, 322]
        num_layers = 1
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    args.output_folder = f"logs/test/{args.backbone}_{args.test_method}/{args.dataset_name}"
    args.output_folder = os.path.join(args.output_folder, args.split) if args.split else args.output_folder
    setup_logging(args.output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(
        f"Testing with {args.test_method} with a {args.backbone} backbone and descriptors dimension {args.fc_output_dim}")
    logging.info(f"The outputs are being saved in {args.output_folder}")
    model = GeoLocalizationNet(backbone=args.backbone,
                               num_patches = num_patches,
                               num_layers = num_layers,
                               fc_output_dim=args.fc_output_dim)
    model_state_dict = torch.load(f'./checkpoints/{args.backbone}_{args.test_method}.pth', map_location=torch.device('cpu'))
    model_state_dict = clean_state_dict(model_state_dict)
    model.load_state_dict(model_state_dict)
    logging.info('model load state_dict completed!')
    model = model.eval().to(args.device)
    start_time = datetime.now()
    test_ds = TestDataset(args, args.dataset_name, args.split)
    logging.info(f"Test set: {test_ds}")
    test(args, test_ds, model)
    logging.info(f"The retrieval time is {str((datetime.now() - start_time).total_seconds()/test_ds.queries_num)[:7]}s "
                 f"for per query")



