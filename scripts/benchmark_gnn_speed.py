"""
Benchmark GNN task head latency vs linear head.
Measures end-to-end inference time per image at batch sizes 1, 8, 32.
Reports mean ± std over 200 warmup + 500 timed iterations.
"""
import time
import torch
import numpy as np
from src.models.csr_baseline import CSRModel
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
IMAGE_SIZE   = 224
WARMUP       = 200
REPEATS      = 500

print(f"Device: {DEVICE}")
print(f"Concepts: {NUM_CONCEPTS}, Classes: {NUM_CLASSES}")

# Build a small dense graph (all edges present — worst case for GNN)
labels = torch.zeros(100, NUM_CONCEPTS)
for i in range(100):
    labels[i, i % NUM_CONCEPTS] = 1.0
labels = labels.to(DEVICE)
edge_index, edge_weight = build_cooccurrence_graph(labels, threshold=0.05)
edge_weight = normalize_edge_weights(edge_index, edge_weight, NUM_CONCEPTS)

def make_model(use_gnn: bool) -> CSRModel:
    m = CSRModel(
        num_concepts=NUM_CONCEPTS,
        num_prototypes=100,
        num_classes=NUM_CLASSES,
        proto_dim=256,
        use_gnn=use_gnn,
        use_uncertainty=False,
        use_vlm=False,
        pretrained_backbone=False,
    ).to(DEVICE)
    m.eval()
    if use_gnn:
        m.set_concept_graph(edge_index, edge_weight)
    return m

def benchmark(model, batch_size: int) -> dict:
    imgs = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(imgs)
    torch.cuda.synchronize()

    # Timed
    latencies = []
    with torch.no_grad():
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(imgs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    arr = np.array(latencies)
    return {
        "mean_ms":   arr.mean(),
        "std_ms":    arr.std(),
        "p50_ms":    np.percentile(arr, 50),
        "p95_ms":    np.percentile(arr, 95),
        "per_img_us": arr.mean() / batch_size * 1000,  # µs per image
    }

print("\n" + "=" * 72)
print(f"{'INFERENCE SPEED BENCHMARK':^72}")
print("=" * 72)
print(f"{'Model':<20} {'BS':>4} {'Mean (ms)':>12} {'Std':>8} {'P95 (ms)':>10} {'µs/img':>10}")
print("-" * 72)

for bs in [1, 8, 32]:
    for name, use_gnn in [("Linear head", False), ("GNN head (GAT)", True)]:
        model = make_model(use_gnn)
        r = benchmark(model, bs)
        print(f"{name:<20} {bs:>4} {r['mean_ms']:>12.2f} {r['std_ms']:>8.2f} {r['p95_ms']:>10.2f} {r['per_img_us']:>10.2f}")
    print()

# Also compute overhead: GNN - Linear
print("GNN overhead (absolute):")
for bs in [1, 8, 32]:
    lin = make_model(False)
    gnn = make_model(True)
    r_lin = benchmark(lin, bs)
    r_gnn = benchmark(gnn, bs)
    overhead_ms = r_gnn["mean_ms"] - r_lin["mean_ms"]
    overhead_pct = overhead_ms / r_lin["mean_ms"] * 100
    print(f"  BS={bs:>2}: +{overhead_ms:.2f} ms  ({overhead_pct:+.1f}%)")

print("\nDone.")
