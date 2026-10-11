import json
import re
import os
from collections import Counter
from typing import List, Tuple, Dict, Optional

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

import pandas as pd
import math

from transformers import AutoTokenizer, AutoModel

def apply_thresholds(probs: torch.Tensor, threshold) -> torch.Tensor:
    if isinstance(threshold, (list, tuple)):
        thr = torch.tensor(threshold, dtype=probs.dtype, device=probs.device)
    elif isinstance(threshold, torch.Tensor):
        thr = threshold.to(device=probs.device, dtype=probs.dtype)
    else:
        return (probs >= float(threshold)).float()

    if thr.dim() == 1:
        thr = thr.view(1, -1).expand_as(probs)
    elif thr.shape != probs.shape:
        thr = thr.view(1, -1).expand_as(probs)

    return (probs >= thr).float()

def split_into_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def find_context_sentences_for_nodes(
    doc_text: str,
    node_names: List[str],
) -> List[str]:
    sentences = split_into_sentences(doc_text)
    if not sentences:
        sentences = [doc_text[:512]]

    contexts = []
    lower_sents = [s.lower() for s in sentences]

    for name in node_names:
        ctx = None
        if isinstance(name, str) and name.strip():
            name_lower = name.lower().strip()
            for sent, sent_lower in zip(sentences, lower_sents):
                if name_lower in sent_lower:
                    ctx = sent
                    break

        if ctx is None:
            ctx = doc_text[:256] if isinstance(doc_text, str) else ""
        contexts.append(ctx)

    return contexts

def build_node_type_vocab(json_paths: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            graphs = json.load(f)
        for g in graphs:
            for t in g.get("node_types", []):
                if t not in vocab:
                    vocab[t] = len(vocab)
    print("\n[Node type vocab]")
    print(vocab)
    return vocab

def build_edge_type_vocab(json_paths: List[str], max_types: int = 200) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            graphs = json.load(f)
        for g in graphs:
            for et in g.get("edge_type", []):
                counter[str(et)] += 1

    vocab: Dict[str, int] = {
        et: idx for idx, (et, _) in enumerate(counter.most_common(max_types))
    }
    vocab["__OTHER__"] = len(vocab)

    print(
        f"\n[Edge type vocab] num_relations = {len(vocab)} "
        f"(top {max_types} of {len(counter)} raw relation labels + __OTHER__)"
    )
    sample_items = list(vocab.items())[:20]
    print("  sample:", sample_items)
    return vocab

def build_doc_text_lookup(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    for col in ["webTitle", "trailText", "bodyText"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path} 缺少列 {col}")
    df["doc_text"] = (
        df["webTitle"].fillna("") + ". "
        + df["trailText"].fillna("") + ". "
        + df["bodyText"].fillna("")
    )
    lookup = dict(zip(df["id"].astype(str), df["doc_text"].astype(str)))
    print(f"[build_doc_text_lookup] {csv_path} -> {len(lookup)} 條 doc_text")
    return lookup

class NewsGraphDataset(InMemoryDataset):
    def __init__(
        self,
        json_path: str,
        doc_text_lookup: Dict[str, str],
        tokenizer: AutoTokenizer,
        bert_model: AutoModel,
        node_type_vocab: Dict[str, int],
        relation_type_vocab: Dict[str, int],
        bert_max_seq_len: int = 128,
        transform=None,
        pre_transform=None,
    ):
        self.json_path = json_path
        self.doc_text_lookup = doc_text_lookup or {}
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.node_type_vocab = node_type_vocab or {}
        self.relation_type_vocab = relation_type_vocab or {}
        self.bert_max_seq_len = bert_max_seq_len

        if self.tokenizer is None or self.bert_model is None:
            raise ValueError("NewsGraphDataset 需要 tokenizer 和 bert_model 來構建 BERT 節點特徵")

        super().__init__(root=".", transform=transform, pre_transform=pre_transform)

        data_list = self._load_graphs_from_json(self.json_path)
        self.data, self.slices = self.collate(data_list)

    def _one_hot_node_types(self, node_types: List[str]) -> torch.Tensor:
        dim = len(self.node_type_vocab)
        if dim == 0:
            return torch.zeros((len(node_types), 0), dtype=torch.float)

        onehots = torch.zeros((len(node_types), dim), dtype=torch.float)
        for i, t in enumerate(node_types):
            idx = self.node_type_vocab.get(t)
            if idx is not None:
                onehots[i, idx] = 1.0
        return onehots

    def _build_bert_entity_features(self, g: Dict) -> torch.Tensor:
        doc_id = g.get("doc_id")
        node_names = g.get("node_names", [])
        node_types = g.get("node_types", [])

        doc_text = self.doc_text_lookup.get(str(doc_id), "")
        if not doc_text:
            doc_text = g.get("title", "") or ""

        contexts = find_context_sentences_for_nodes(doc_text, node_names)
        enc = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.bert_max_seq_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                bert_emb = outputs.pooler_output
            else:
                bert_emb = outputs.last_hidden_state[:, 0, :]

        bert_emb = bert_emb.to(torch.float32)
        type_onehot = self._one_hot_node_types(node_types)
        if type_onehot.numel() > 0:
            x = torch.cat([bert_emb, type_onehot], dim=1)
        else:
            x = bert_emb
        return x

    def _load_graphs_from_json(self, path: str) -> List[Data]:
        with open(path, "r", encoding="utf-8") as f:
            graphs = json.load(f)

        data_list: List[Data] = []
        for g in graphs:
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
            if edge_index.numel() == 0:
                continue

            num_edges = edge_index.size(1)

            y = torch.tensor(g["y"], dtype=torch.float)

            x = self._build_bert_entity_features(g)

            raw_edge_types = g.get("edge_type", [])
            if len(raw_edge_types) != num_edges:
                raise ValueError(
                    f"edge_index edges={num_edges}, but len(edge_type)={len(raw_edge_types)}"
                )

            if self.relation_type_vocab:
                other_idx = self.relation_type_vocab.get("__OTHER__", 0)
                edge_type_ids = [
                    self.relation_type_vocab.get(str(et), other_idx)
                    for et in raw_edge_types
                ]
            else:
                edge_type_ids = [0] * num_edges
            edge_type_ids = torch.tensor(edge_type_ids, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_type=edge_type_ids, y=y)

            data.doc_id = g.get("doc_id")
            data.title = g.get("title")
            data.url = g.get("url")
            data.published_utc = g.get("published_utc")
            data.risk_types = g.get("risk_types", [])
            data.node_names = g.get("node_names", [])
            data.node_types = g.get("node_types", [])
            data.edge_type_str = raw_edge_types

            data_list.append(data)

        return data_list


def build_dataloaders(
    train_json_path: str,
    val_json_path: str,
    test_json_path: str,
    batch_size: int,
    doc_text_lookup_train: Dict[str, str],
    doc_text_lookup_val: Dict[str, str],
    doc_text_lookup_test: Dict[str, str],
    tokenizer: AutoTokenizer,
    bert_model: AutoModel,
    node_type_vocab: Dict[str, int],
    relation_type_vocab: Dict[str, int],
    bert_max_seq_len: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = NewsGraphDataset(
        json_path=train_json_path,
        doc_text_lookup=doc_text_lookup_train,
        tokenizer=tokenizer,
        bert_model=bert_model,
        node_type_vocab=node_type_vocab,
        relation_type_vocab=relation_type_vocab,
        bert_max_seq_len=bert_max_seq_len,
    )
    val_dataset = NewsGraphDataset(
        json_path=val_json_path,
        doc_text_lookup=doc_text_lookup_val,
        tokenizer=tokenizer,
        bert_model=bert_model,
        node_type_vocab=node_type_vocab,
        relation_type_vocab=relation_type_vocab,
        bert_max_seq_len=bert_max_seq_len,
    )
    test_dataset = NewsGraphDataset(
        json_path=test_json_path,
        doc_text_lookup=doc_text_lookup_test,
        tokenizer=tokenizer,
        bert_model=bert_model,
        node_type_vocab=node_type_vocab,
        relation_type_vocab=relation_type_vocab,
        bert_max_seq_len=bert_max_seq_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class RiskRGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 4,
        num_relations: int = 1,
        num_bases: Optional[int] = None,
        num_gnn_layers: int = 3,
        mlp_hidden_dims: Optional[list[int]] = None,
    ):
        super().__init__()
        if num_relations <= 0:
            raise ValueError("num_relations 必須 > 0")
        if num_bases is None:
            num_bases = min(30, num_relations)
        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers 必須 >= 1")

        self.convs = nn.ModuleList()
        self.convs.append(
            RGCNConv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                num_relations=num_relations,
                num_bases=num_bases,
            )
        )
        for _ in range(num_gnn_layers - 1):
            self.convs.append(
                RGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases,
                )
            )

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64]

        self.mlp_layers = nn.ModuleList()
        in_fc_dim = hidden_dim
        for dim in mlp_hidden_dims:
            self.mlp_layers.append(nn.Linear(in_fc_dim, dim))
            in_fc_dim = dim

        self.out = nn.Linear(in_fc_dim, num_classes)

    def forward(self, x, edge_index, edge_type, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_type)
            h = F.relu(h)

        h_graph = global_mean_pool(h, batch)

        h_fc = h_graph
        for fc in self.mlp_layers:
            h_fc = F.relu(fc(h_fc))
        logits = self.out(h_fc)
        return logits

def build_model(
    in_dim: int,
    num_relations: int,
    device: torch.device,
    hidden_dim: int = 64,
    num_classes: int = 4,
    num_gnn_layers: int = 2,
    mlp_hidden_dims: Optional[list[int]] = None,
) -> RiskRGCN:
    if mlp_hidden_dims is None:
        mlp_hidden_dims = [64]

    model = RiskRGCN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_relations=num_relations,
        num_bases=None,
        num_gnn_layers=num_gnn_layers,
        mlp_hidden_dims=mlp_hidden_dims,
    ).to(device)

    return model

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch) 

        y = batch.y.view(batch.num_graphs, -1) 

        if logits.shape != y.shape:
            raise RuntimeError(f"Shape mismatch in train: logits={logits.shape}, y={y.shape}")

        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    threshold = 0.5,
) -> Dict[str, float]:
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

        y = batch.y.view(batch.num_graphs, -1)
        if logits.shape != y.shape:
            raise RuntimeError(f"Shape mismatch in eval: logits={logits.shape}, y={y.shape}")

        loss = loss_fn(logits, y)

        total_loss += loss.item()
        total_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    if total_batches == 0:
        return {
            "loss": float("nan"),
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "f1_label_0": float("nan"),
            "f1_label_1": float("nan"),
            "f1_label_2": float("nan"),
            "f1_label_3": float("nan"),
        }

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(logits)
    preds = apply_thresholds(probs, threshold)

    y_true = labels.view(-1)
    y_pred = preds.view(-1)

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    micro_f1  = 2 * precision * recall / (precision + recall + 1e-8)

    per_label_f1 = []
    num_labels = labels.size(1)
    for k in range(num_labels):
        yk_true = labels[:, k]
        yk_pred = preds[:, k]
        tp_k = ((yk_true == 1) & (yk_pred == 1)).sum().item()
        fp_k = ((yk_true == 0) & (yk_pred == 1)).sum().item()
        fn_k = ((yk_true == 1) & (yk_pred == 0)).sum().item()

        if tp_k + fp_k + fn_k == 0:
            f1_k = float("nan")
        else:
            prec_k = tp_k / (tp_k + fp_k + 1e-8)
            rec_k  = tp_k / (tp_k + fn_k + 1e-8)
            f1_k   = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-8)
        per_label_f1.append(f1_k)

    avg_loss = total_loss / total_batches

    valid_f1 = [f for f in per_label_f1 if not math.isnan(f)]
    macro_f1 = sum(valid_f1) / len(valid_f1) if valid_f1 else 0.0

    metrics = {
        "loss": avg_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }
    for i, f1_k in enumerate(per_label_f1):
        metrics[f"f1_label_{i}"] = f1_k

    return metrics

@torch.no_grad()
def compute_pos_weight(
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    pos_counts = None
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, -1)

        if pos_counts is None:
            pos_counts = torch.zeros(y.size(1), dtype=torch.float64)

        pos_counts += y.sum(dim=0).cpu().double()
        total_samples += y.size(0)

    if pos_counts is None:
        raise ValueError("compute_pos_weight: train_loader is empty.")

    neg_counts = total_samples - pos_counts
    pos_counts_f = pos_counts.to(torch.float32)
    neg_counts_f = neg_counts.to(torch.float32)

    pos_weight = torch.where(
        pos_counts_f > 0,
        neg_counts_f / pos_counts_f,
        torch.ones_like(pos_counts_f),
    )

    print("\n[Label stats from train_loader]")
    print("  total_samples:", total_samples)
    print("  pos_counts:", pos_counts_f.tolist())
    print("  neg_counts:", neg_counts_f.tolist())
    print("  pos_weight:", pos_weight.tolist())

    return pos_weight.to(device=device)

def train_and_eval_once(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    num_epochs: int,
    base_threshold: float,
    best_model_path: str,
) -> Tuple[float, int]:
    best_val_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics = evaluate(
            model, val_loader, device, loss_fn,
            threshold=base_threshold
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_micro_f1={val_metrics['micro_f1']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(
                f"  -> New best model saved "
                f"(epoch {epoch}, val_macro_f1={best_val_macro_f1:.4f})"
            )

    return best_val_macro_f1, best_epoch

@torch.no_grad()
def calibrate_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    candidate_thresholds = None,
) -> List[float]:
    model.eval()

    if candidate_thresholds is None:
        candidate_thresholds = [0.1 * i for i in range(1, 10)]

    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        y = batch.y.view(batch.num_graphs, -1)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    if not all_logits:
        raise ValueError("calibrate_thresholds: loader is empty.")

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits)

    thresholds = []
    for k in range(num_classes):
        y_true = labels[:, k]
        best_t = 0.5
        best_f1 = 0.0

        for t in candidate_thresholds:
            y_pred = (probs[:, k] >= t).float()

            tp = ((y_true == 1) & (y_pred == 1)).sum().item()
            fp = ((y_true == 0) & (y_pred == 1)).sum().item()
            fn = ((y_true == 1) & (y_pred == 0)).sum().item()

            if tp + fp + fn == 0:
                f1 = 0.0
            else:
                prec = tp / (tp + fp + 1e-8)
                rec  = tp / (tp + fn + 1e-8)
                f1   = 2 * prec * rec / (prec + rec + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds.append(best_t)

    return thresholds

@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold = 0.5,
) -> List[Dict]:
    model.eval()
    results = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        probs = torch.sigmoid(logits).cpu()
        preds = apply_thresholds(probs, threshold)

        doc_ids = batch.doc_id
        if not isinstance(doc_ids, (list, tuple)):
            doc_ids = [doc_ids] * probs.size(0)

        for i in range(probs.size(0)):
            results.append({
                "doc_id": doc_ids[i],
                "probs": probs[i].tolist(),
                "preds": preds[i].tolist(),
            })

    return results

def main():
    train_json = "graph_dataset_train.json"
    val_json   = "graph_dataset_val.json"
    test_json  = "graph_dataset_test.json"

    guardian_train_csv = "guardian_news_train.csv"
    guardian_val_csv   = "guardian_news_val.csv"
    guardian_test_csv  = "guardian_news_test.csv"

    bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    bert_max_seq_len = 128

    batch_size = 32
    num_epochs = 30
    hidden_dim = 64
    fc_dim     = 64
    lr         = 1e-3
    best_model_path = "risk_rgcn_best.pt"
    test_csv_path   = "test_predictions_rgcn.csv"

    risk_label_names = ["credit", "market", "operational", "fraud"]
    base_threshold_for_training = 0.5

    if os.getenv("RGCN_USE_MPS", "").lower() in {"1", "true", "yes"} and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    print("\n[Init] 使用 BERT entity-in-context 節點特徵")

    doc_text_lookup_train = build_doc_text_lookup(guardian_train_csv)
    doc_text_lookup_val   = build_doc_text_lookup(guardian_val_csv)
    doc_text_lookup_test  = build_doc_text_lookup(guardian_test_csv)

    node_type_vocab = build_node_type_vocab([train_json, val_json, test_json])

    print(f"\n[Init] Loading BERT model: {bert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)
    bert_model.eval()

    relation_type_vocab = build_edge_type_vocab([train_json, val_json, test_json])
    num_relations = len(relation_type_vocab)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_json_path=train_json,
        val_json_path=val_json,
        test_json_path=test_json,
        batch_size=batch_size,
        doc_text_lookup_train=doc_text_lookup_train,
        doc_text_lookup_val=doc_text_lookup_val,
        doc_text_lookup_test=doc_text_lookup_test,
        tokenizer=tokenizer,
        bert_model=bert_model,
        node_type_vocab=node_type_vocab,
        relation_type_vocab=relation_type_vocab,
        bert_max_seq_len=bert_max_seq_len,
    )

    first_batch = next(iter(train_loader))
    in_dim = first_batch.x.size(1)
    print("Input feature dim =", in_dim)
    print("First batch y shape =", first_batch.y.shape)
    print("First batch num_graphs =", first_batch.num_graphs)

    model = build_model(
        in_dim=in_dim,
        num_relations=num_relations,
        device=device,
        hidden_dim=hidden_dim,
        num_classes=4,
        num_gnn_layers=3,
        mlp_hidden_dims=[64],
    )

    pos_weight = compute_pos_weight(train_loader, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_macro_f1, best_epoch = train_and_eval_once(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        base_threshold=base_threshold_for_training,
        best_model_path=best_model_path,
    )

    print(
        f"\nTraining finished. "
        f"Best epoch = {best_epoch}, best val_macro_f1 = {best_val_macro_f1:.4f}"
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    calibrated_thresholds = calibrate_thresholds(
        model, val_loader, device, num_classes=4
    )
    print("\nCalibrated per-label thresholds (credit, market, operational, fraud):")
    print(calibrated_thresholds)

    test_metrics = evaluate(
        model, test_loader, device, loss_fn,
        threshold=calibrated_thresholds
    )
    print("\n===== Test metrics (using best val model & calibrated thresholds, RGCN) =====")
    print(f"test_loss      = {test_metrics['loss']:.4f}")
    print(f"test_micro_f1  = {test_metrics['micro_f1']:.4f}")
    print(f"test_macro_f1  = {test_metrics['macro_f1']:.4f}")
    print(f"test_f1_label0 = {test_metrics['f1_label_0']:.4f}  # credit")
    print(f"test_f1_label1 = {test_metrics['f1_label_1']:.4f}  # market")
    print(f"test_f1_label2 = {test_metrics['f1_label_2']:.4f}  # operational")
    print(f"test_f1_label3 = {test_metrics['f1_label_3']:.4f}  # fraud")

    model.eval()
    rows = []

    def to_list_field(field, n, default=None):
        if isinstance(field, (list, tuple)):
            return list(field)
        if field is None:
            return [default] * n
        return [field] * n

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

            probs_sigmoid = torch.sigmoid(logits).cpu()

            preds = apply_thresholds(probs_sigmoid, calibrated_thresholds)

            probs_softmax = torch.softmax(logits, dim=1).cpu()

            y_true_batch = batch.y.view(batch.num_graphs, -1).cpu()

            B = probs_sigmoid.size(0)

            doc_ids = to_list_field(batch.doc_id, B, default=None)
            titles = to_list_field(getattr(batch, "title", None), B, default=None)
            urls = to_list_field(getattr(batch, "url", None), B, default=None)
            published_utcs = to_list_field(getattr(batch, "published_utc", None), B, default=None)

            for i in range(B):
                pred_vec = preds[i].tolist()
                y_vec    = y_true_batch[i].tolist()

                sig_vec  = probs_sigmoid[i].tolist()
                smax_vec = probs_softmax[i].tolist()

                pred_names = [risk_label_names[j] for j, v in enumerate(pred_vec) if v >= 0.5]

                rows.append({
                    "doc_id":         doc_ids[i],
                    "title":          titles[i],
                    "url":            urls[i],
                    "published_utc":  published_utcs[i],

                    "has_credit_risk":       int(y_vec[0]),
                    "has_market_risk":       int(y_vec[1]),
                    "has_operational_risk":  int(y_vec[2]),
                    "has_fraud_risk":        int(y_vec[3]),

                    "p_credit_sigmoid":      sig_vec[0],
                    "p_market_sigmoid":      sig_vec[1],
                    "p_operational_sigmoid": sig_vec[2],
                    "p_fraud_sigmoid":       sig_vec[3],

                    "p_credit_softmax":      smax_vec[0],
                    "p_market_softmax":      smax_vec[1],
                    "p_operational_softmax": smax_vec[2],
                    "p_fraud_softmax":       smax_vec[3],

                    "pred_risk_labels": ";".join(pred_names) if pred_names else "",
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(test_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nTest predictions CSV saved to: {test_csv_path}")

    sample_results = predict_loader(
        model, test_loader, device,
        threshold=calibrated_thresholds
    )
    print("\nSample predictions from test set (RGCN):")
    for r in sample_results[:5]:
        print(f"doc_id={r['doc_id']}, probs={r['probs']}, preds={r['preds']}")

if __name__ == "__main__":
    main()
