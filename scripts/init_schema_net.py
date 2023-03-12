import argparse
import os
from typing import Dict, Any

import tqdm

import torch
import torch.cuda
from torch.utils.data import DataLoader
import torch.backends.cudnn

import cv_lib.utils as cv_utils

import schema_inference.graph as graph
from schema_inference.data import build_train_dataset
import schema_inference.utils as utils


@torch.no_grad()
def init_graph(
    dataloader: DataLoader,
    wrapper: utils.IngredientModelWrapper,
    graph: graph.SchemaNet,
    device: torch.device
):
    n_tracked = torch.zeros(graph.num_classes)
    for x, gt in tqdm.tqdm(dataloader, total=len(dataloader)):
        x, gt = utils.move_data_to_device(x, gt, device)
        label = gt["label"]
        output: Dict[str, torch.Tensor] = wrapper(x)
        edges = graph.feat_to_limited_edges(output["ingredients"], output["attn"], label).to(device)
        with torch.no_grad():
            for cls_id, instance_e in zip(label, edges):
                graph.edge_weights.tensor[cls_id] += instance_e
                n_tracked[cls_id] += 1

    with torch.no_grad():
        graph.edge_weights.tensor /= n_tracked[:, None, None].to(device)

    graph.normalize()


@torch.no_grad()
def init_class_vertices(
    dataloader: DataLoader,
    wrapper: utils.IngredientModelWrapper,
    graph: graph.SchemaNet,
    device: torch.device
):
    class_vertices = torch.zeros(graph.num_classes, graph.num_vertices, device=device)
    n_tracked = torch.zeros(graph.num_classes)
    for x, gt in tqdm.tqdm(dataloader, total=len(dataloader)):
        x, gt = utils.move_data_to_device(x, gt, device)
        output: Dict[str, torch.Tensor] = wrapper(x)
        vertices = graph.feat_to_full_vertices(
            output["ingredients"],
            output["attn_cls"]
        )
        for cls_id, instance_v in zip(gt["label"], vertices):
            class_vertices[cls_id] += instance_v
            n_tracked[cls_id] += 1

    class_vertices /= n_tracked[:, None].to(device)
    class_vertices /= class_vertices.sum(dim=-1, keepdim=True)
    return class_vertices


def main(args):
    # split configs
    cfg: Dict[str, Any] = cv_utils.get_cfg(args.schema_net_cfg)
    data_cfg: Dict[str, Any] = cv_utils.get_cfg(cfg["dataset"])
    schema_net_cfg = cfg["schema_net"]

    # set cuda
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make deterministic
    if args.seed is not None:
        cv_utils.make_deterministic(args.seed)

    # get dataloader
    print("Building dataset...")
    data_cfg["make_partial"] = args.make_partial
    train_dataset, _, n_classes, _ = build_train_dataset(data_cfg)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    # load jit models
    print("Loading jit models...")
    backbone: torch.jit.ScriptModule = torch.jit.load(schema_net_cfg["backbone_jit"], map_location=device)
    discretization: torch.jit.ScriptModule = torch.jit.load(schema_net_cfg["discretization_jit"], map_location=device)
    print("Loaded jit models.")
    wrapper = utils.IngredientModelWrapper(backbone, discretization)
    # create schema net
    print("Creating graphs...")
    schema_net = graph.SchemaNet(
        num_vertices=wrapper.num_ingredients,
        num_classes=n_classes,
        **schema_net_cfg["ir_atlas"]
    ).to(device)

    class_max_vertices = schema_net.class_max_vertices

    wrapper.eval().to(device)
    print("Running vertex initialization...")
    init_weights = init_class_vertices(
        dataloader,
        wrapper,
        graph=schema_net,
        device=device
    )
    init_weights, valid_vertices = init_weights.topk(class_max_vertices, dim=1)
    print("Running graph initialization...")
    schema_net.register_class_vertices(valid_vertices)
    schema_net.vertex_weights.copy_(init_weights)
    init_graph(
        dataloader,
        wrapper,
        graph=schema_net,
        device=device
    )
    state_dict = schema_net.state_dict()
    torch.save(state_dict, args.save_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_net_cfg", type=str)
    parser.add_argument("--save_fp", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--make_partial", type=float, default=None)
    args = parser.parse_args()
    save_path = os.path.dirname(args.save_fp)
    os.makedirs(save_path, exist_ok=True)
    main(args)
