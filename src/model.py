import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, SEED,
    CONGRESSES, TRAIN_CONGRESSES, VAL_CONGRESS, TEST_CONGRESSES,
    HIDDEN_DIM, N_HEADS, N_TEMPORAL_HEADS, DROPOUT,
    LR, WEIGHT_DECAY, EPOCHS, SCHEDULER_STEP, SCHEDULER_GAMMA,
    GRAD_CLIP, COALITION_PAIRS,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjacency_to_edge_index(adjacency):
    rows, cols = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return edge_index


def load_congress_data(congress_num):
    path = PROCESSED_DIR / f"congress_{congress_num}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    features = torch.tensor(data["features"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    adjacency = data["adjacency"]
    edge_index = adjacency_to_edge_index(adjacency)
    party_codes = data["party_codes"]
    defection_rates = data["defection_rates"]
    return Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        party_codes=torch.tensor(party_codes, dtype=torch.long),
        defection_rates=torch.tensor(defection_rates, dtype=torch.float32),
        num_nodes=features.shape[0],
    )


class CongressGAT(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS,
                 temporal_heads=N_TEMPORAL_HEADS, dropout=DROPOUT):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=n_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=temporal_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.defection_head = nn.Sequential(
            nn.Linear(hidden_dim + in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.coalition_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.polarization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.elu(self.gat1(x, edge_index))
        h = self.dropout(h)
        h = F.elu(self.gat2(h, edge_index))
        return h

    def predict_defection(self, h, x):
        combined = torch.cat([h, x], dim=-1)
        return torch.sigmoid(self.defection_head(combined)).squeeze(-1)

    def predict_coalition(self, h_i, h_j):
        combined = torch.cat([h_i, h_j], dim=-1)
        return torch.sigmoid(self.coalition_head(combined)).squeeze(-1)

    def predict_polarization(self, graph_embedding):
        return self.polarization_head(graph_embedding).squeeze(-1)

    def temporal_forward(self, embeddings_seq):
        seq = torch.stack(embeddings_seq, dim=0).unsqueeze(0)
        attn_out, _ = self.temporal_attn(seq, seq, seq)
        out = self.temporal_norm(seq + attn_out)
        return out.squeeze(0)

    def get_attention_weights(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.elu(self.gat1(x, edge_index))
        h = self.dropout(h)
        _, attn2 = self.gat2(h, edge_index, return_attention_weights=True)
        return attn2


class CongressGCN(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=HIDDEN_DIM,
                 temporal_heads=N_TEMPORAL_HEADS, dropout=DROPOUT):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=temporal_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.defection_head = nn.Sequential(
            nn.Linear(hidden_dim + in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.coalition_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.polarization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.elu(self.gcn1(x, edge_index))
        h = self.dropout(h)
        h = F.elu(self.gcn2(h, edge_index))
        return h

    def predict_defection(self, h, x):
        combined = torch.cat([h, x], dim=-1)
        return torch.sigmoid(self.defection_head(combined)).squeeze(-1)

    def predict_coalition(self, h_i, h_j):
        combined = torch.cat([h_i, h_j], dim=-1)
        return torch.sigmoid(self.coalition_head(combined)).squeeze(-1)

    def predict_polarization(self, graph_embedding):
        return self.polarization_head(graph_embedding).squeeze(-1)

    def temporal_forward(self, embeddings_seq):
        seq = torch.stack(embeddings_seq, dim=0).unsqueeze(0)
        attn_out, _ = self.temporal_attn(seq, seq, seq)
        out = self.temporal_norm(seq + attn_out)
        return out.squeeze(0)


def compute_defection_loss(pred, labels, pos_weight=None):
    if pos_weight is None:
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    return F.binary_cross_entropy(pred, labels.float(), reduction="mean")


def sample_coalition_pairs(data, n_pairs=COALITION_PAIRS, epoch=0):
    party = data.party_codes
    adjacency_flat = data.edge_index
    n = data.num_nodes
    rng = np.random.RandomState(SEED + epoch)

    pairs_i = []
    pairs_j = []
    targets = []

    edge_set = set()
    for k in range(adjacency_flat.shape[1]):
        i, j = adjacency_flat[0, k].item(), adjacency_flat[1, k].item()
        edge_set.add((i, j))

    cross_edges = [(i, j) for (i, j) in edge_set if party[i] != party[j]]
    same_edges = [(i, j) for (i, j) in edge_set if party[i] == party[j]]

    n_cross = min(n_pairs // 2, len(cross_edges))
    n_same = min(n_pairs - n_cross, len(same_edges))

    if n_cross > 0:
        idx = rng.choice(len(cross_edges), size=n_cross, replace=False)
        for k in idx:
            i, j = cross_edges[k]
            pairs_i.append(i)
            pairs_j.append(j)
            targets.append(1.0)

    if n_same > 0:
        idx = rng.choice(len(same_edges), size=n_same, replace=False)
        for k in idx:
            i, j = same_edges[k]
            pairs_i.append(i)
            pairs_j.append(j)
            targets.append(0.0)

    if len(pairs_i) == 0:
        return None, None, None

    return (
        torch.tensor(pairs_i, dtype=torch.long),
        torch.tensor(pairs_j, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32),
    )


def load_fiedler_targets():
    import json
    path = RESULTS_DIR / "spectral_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        spectral = json.load(f)
    targets = {}
    for c_str, data in spectral.items():
        if c_str.isdigit() and isinstance(data, dict) and "fiedler" in data:
            targets[int(c_str)] = data["fiedler"]
    return targets


def train_model(model, model_name="GAT"):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_data = {}
    for c in TRAIN_CONGRESSES:
        d = load_congress_data(c)
        if d is not None:
            train_data[c] = d.to(device)

    val_data = load_congress_data(VAL_CONGRESS)
    if val_data is not None:
        val_data = val_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

    fiedler_targets = load_fiedler_targets()

    congress_order = sorted(train_data.keys())
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        graph_embeddings = []

        for c in congress_order:
            data = train_data[c]
            h = model.encode(data)
            graph_embeddings.append(h.mean(dim=0))

            defection_pred = model.predict_defection(h, data.x)
            defection_loss = F.binary_cross_entropy(defection_pred, data.y.float())
            total_loss = total_loss + defection_loss

            pairs_i, pairs_j, coalition_targets = sample_coalition_pairs(data, epoch=epoch)
            if pairs_i is not None:
                pairs_i = pairs_i.to(device)
                pairs_j = pairs_j.to(device)
                coalition_targets = coalition_targets.to(device)
                coalition_pred = model.predict_coalition(h[pairs_i], h[pairs_j])
                coalition_loss = F.binary_cross_entropy(coalition_pred, coalition_targets)
                total_loss = total_loss + coalition_loss * 0.5

        if len(graph_embeddings) > 1:
            temporal_out = model.temporal_forward(graph_embeddings)
            for t_idx, c in enumerate(congress_order):
                if c not in fiedler_targets:
                    continue
                pol_target = torch.tensor([fiedler_targets[c]], dtype=torch.float32, device=device)
                pol_pred = model.predict_polarization(temporal_out[t_idx])
                pol_loss = F.mse_loss(pol_pred, pol_target)
                total_loss = total_loss + pol_loss * 0.1

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if val_data is not None and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                h_val = model.encode(val_data)
                val_pred = model.predict_defection(h_val, val_data.x)
                val_loss = F.binary_cross_entropy(val_pred, val_data.y.float()).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 50 == 0:
                print(f"  [{model_name}] Epoch {epoch+1}/{EPOCHS}, val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    print("Training CongressGAT...")
    gat = CongressGAT()
    gat_params = sum(p.numel() for p in gat.parameters())
    print(f"  GAT parameters: {gat_params}")
    gat = train_model(gat, model_name="GAT")
    torch.save(gat.state_dict(), RESULTS_DIR / "gat_model.pt")

    print("Training CongressGCN...")
    gcn = CongressGCN()
    gcn_params = sum(p.numel() for p in gcn.parameters())
    print(f"  GCN parameters: {gcn_params}")
    assert gat_params != gcn_params, "GAT and GCN should have different param counts"
    gcn = train_model(gcn, model_name="GCN")
    torch.save(gcn.state_dict(), RESULTS_DIR / "gcn_model.pt")

    print(f"GAT params: {gat_params}, GCN params: {gcn_params}")
    print("Model training complete.")


if __name__ == "__main__":
    main()
