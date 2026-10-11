import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from RGCN_4 import (
    build_model,
    build_node_type_vocab,
    build_edge_type_vocab,
    build_doc_text_lookup,
    build_dataloaders,
    compute_pos_weight,
    train_one_epoch,
    evaluate,
)

def main():
    train_json = "graph_dataset_train.json"
    val_json   = "graph_dataset_val.json"
    test_json  = "graph_dataset_test.json"

    guardian_train_csv = "guardian_news_train.csv"
    guardian_val_csv   = "guardian_news_val.csv"
    guardian_test_csv  = "guardian_news_test.csv"

    bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    bert_max_seq_len = 64

    batch_size = 64
    lr = 1e-3
    num_epochs_ablation = 3
    hidden_dim = 64
    num_classes = 4
    base_threshold = 0.5

    if os.getenv("RGCN_USE_MPS", "").lower() in {"1", "true", "yes"} and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    doc_text_lookup_train = build_doc_text_lookup(guardian_train_csv)
    doc_text_lookup_val   = build_doc_text_lookup(guardian_val_csv)
    doc_text_lookup_test  = build_doc_text_lookup(guardian_test_csv)

    node_type_vocab = build_node_type_vocab([train_json, val_json, test_json])
    relation_type_vocab = build_edge_type_vocab([train_json, val_json, test_json])
    num_relations = len(relation_type_vocab)

    print(f"[Ablation] num_relations = {num_relations}")

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)
    bert_model.eval()

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
    print("[Ablation] in_dim =", in_dim)

    pos_weight = compute_pos_weight(train_loader, device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gnn_depth_list = [1, 2, 3]
    mlp_configs    = [[], [64], [64,64],[64,64,64]]

    results = []

    for gnn_depth in gnn_depth_list:
        for mlp_hidden in mlp_configs:
            print(f"\n=== GNN layers = {gnn_depth}, MLP hidden = {mlp_hidden} ===")

            model = build_model(
                in_dim=in_dim,
                num_relations=num_relations,
                device=device,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_gnn_layers=gnn_depth,
                mlp_hidden_dims=mlp_hidden,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_macro_f1 = -1.0
            best_epoch = -1

            for epoch in range(1, num_epochs_ablation + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
                with torch.no_grad():
                    val_metrics = evaluate(
                        model, val_loader, device, loss_fn,
                        threshold=base_threshold,
                    )
                val_macro_f1 = val_metrics["macro_f1"]

                print(
                    f"  [Epoch {epoch:02d}] "
                    f"train_loss={train_loss:.4f}, "
                    f"val_macro_f1={val_macro_f1:.4f}"
                )

                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    best_epoch = epoch

            results.append({
                "gnn_layers": gnn_depth,
                "mlp_hidden": str(mlp_hidden),
                "best_val_macro_f1": best_val_macro_f1,
                "best_epoch": best_epoch,
            })

    df = pd.DataFrame(results)
    print("\n[Ablation results]")
    print(df)

    out_path = "ablation_rgcn_depth_mlp_fast.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nAblation results saved to: {out_path}")

if __name__ == "__main__":
    main()
