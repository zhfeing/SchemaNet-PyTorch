"""
cluster_features.py

Given a pre-trained model, and a layer to extract intermediate features, this script will
run the k-means clustering on flattened features (N * dim). The k cluster centers are save
as initial feature ingredients.
"""

import argparse
import os
from typing import Dict, Any, List

import tqdm
import h5py
import numpy as np

import torch
import torch.utils.data as data

import cv_lib.utils as cv_utils

from schema_inference.data import build_train_dataset
from schema_inference.utils import load_pretrain_model
from models import get_model
from discretization import Adapter


class KMeansClustering:
    def __init__(self, num_clusters: int, method: str):
        self.num_clusters = num_clusters
        self.method = method

    def scipy_kmeans(self, x: np.ndarray) -> np.ndarray:
        from scipy.cluster.vq import kmeans
        centers, _ = kmeans(x, self.num_clusters)
        return centers

    def minibatch_kmeans(self, x: np.ndarray) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans
        k_means = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=1024,
            verbose=True,
            compute_labels=False,
            n_init="auto"
        )
        k_means.fit(x)
        centers = k_means.cluster_centers_
        return centers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        methods = {
            "cpu_kmeans": self.scipy_kmeans,
            "minibatch": self.minibatch_kmeans
        }
        return methods[self.method](x)


def collect_features(args):
    global_cfg = cv_utils.get_cfg(args.cfg_fp)
    # split configs
    data_cfg: Dict[str, Any] = cv_utils.get_cfg(global_cfg["dataset"])
    model_cfg: Dict[str, Any] = cv_utils.get_cfg(global_cfg["model"])
    discretization_cfg: Dict[str, Any] = global_cfg["discretization"]
    # make deterministic
    generator = torch.Generator()
    if args.seed is not None:
        cv_utils.make_deterministic(args.seed)
        generator.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloader
    print("Building dataset...")
    train_set, _, n_classes, _ = build_train_dataset(data_cfg)
    sampler = data.RandomSampler(train_set, generator=generator)
    data_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers
    )
    # create model
    print("Building model...")
    model = get_model(model_cfg["model"], n_classes)
    print("Loading pre-train parameters...")
    load_pretrain_model(model_cfg["resume"][data_cfg["name"]], model)
    extract_name = discretization_cfg["encoder_layer"]
    extractor = cv_utils.MidExtractor(model, extract_names=[extract_name])
    model.eval().to(device)
    adaptor = Adapter()

    # extracting mid features
    print("Extracting mid features")
    features: List[torch.Tensor] = list()
    with torch.no_grad():
        for x, _ in tqdm.tqdm(data_loader):
            x = x.to(device)
            model(x)
            feat: torch.Tensor = extractor.features[extract_name]
            # adapt feature: [bs, dim, h, w] -> [h * w, bs, dim]
            feat = adaptor.adapt(feat)
            # [h * w, bs, dim] -> [h * w * bs, dim]
            feat = feat.flatten(0, 1).cpu()
            # old version # [bs, dim, h, w] -> [bs, dim, h * w] -> [bs, h * w, dim] -> [bs * h * w, dim]
            # feat = feat.cpu().flatten(2).permute(0, 2, 1).flatten(0, 1)
            features += feat.unbind(0)
            if len(features) > args.max_features:
                print(f"Collected more than {args.max_features} features.")
                break
    features = features[:args.max_features]
    features = torch.stack(features).numpy()
    with h5py.File(os.path.join(args.save_path, "saved_features.h5"), "w") as file:
        file["features"] = features
    return features


def clustering(args, features: np.ndarray):
    num_features = features.shape[0]
    clustering = KMeansClustering(args.num_clusters, args.kmeans_method)
    cluster_centers = clustering(features)
    save_fp = os.path.join(args.save_path, f"cluster_{args.num_clusters}_from_{num_features}.pth")
    cluster_centers = torch.from_numpy(cluster_centers).to(torch.float32)
    torch.save(cluster_centers, save_fp)
    print("Done")


def main(args):
    args.num_clusters = cv_utils.get_cfg(args.cfg_fp)["discretization"]["vocabulary"]["size"]

    features: np.ndarray
    if args.saved_features_fp is not None:
        with h5py.File(args.saved_features_fp) as f:
            features = f["saved_features"][:]
        print("Loaded saved features from file")
    else:
        print("Generating new features")
        features = collect_features(args)
    clustering(args, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_fp", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--saved_features_fp", type=str, default=None)
    parser.add_argument("--kmeans_method", type=str, default="cpu_kmeans")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_features", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    main(args)

