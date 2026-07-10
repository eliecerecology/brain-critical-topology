import numpy as np

files = [
    "linespace-original_lumi/results_final_linspace200_0_to_1lumi.npz",
    "results_final_linspace200_1_to_4.npz",
    "results_final_linspace200_4_to_7.npz",
    "results_final_linspace200_7_to_11.npz",
]

parts = [np.load(f, allow_pickle=True) for f in files]

# Verify k_values are consistent across all files
for p in parts[1:]:
    assert np.allclose(p["k_values"], parts[0]["k_values"]), "k_values mismatch!"

# Concatenate along axis 0 (network dimension)
matrix_keys = ["order_matrix", "variability_matrix", "plv_matrix", "dfa_matrix", "fa_matrix"]

combined = {
    key: np.concatenate([p[key] for p in parts], axis=0)
    for key in matrix_keys
}
combined["k_values"]      = parts[0]["k_values"]
combined["network_names"] = np.concatenate([p["network_names"] for p in parts])
combined["i_start"]       = 0
combined["i_end"]         = 11

np.savez("results_final_linspace200_all.npz", **combined)

# Sanity check
out = np.load("results_final_linspace200_all.npz", allow_pickle=True)
print("Combined shape:", out["order_matrix"].shape)   # expect (11, n_replicas, n_k)
print("Network names:", out["network_names"])