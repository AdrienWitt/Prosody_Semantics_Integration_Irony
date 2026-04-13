import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image
from nilearn.image import resample_to_img, load_img, coord_transform
from scipy.ndimage import label, center_of_mass
from nibabel.affines import apply_affine
from statsmodels.stats.multitest import fdrcorrection

# ================================
# SETTINGS — only change this block
# ================================
RESULTS_DIR      = "results_nopca"
MIN_CLUSTER_SIZE = 5
FDR_ALPHA        = 0.05
TOP_PERCENTILE   = 99.5   # None = cluster all FDR-significant voxels; float = top X% spatial filter
N_SPLITS         = 5

EMB_PRIMARY  = "cross_attention"
EMB_COMPARE  = "statement_only"

# Choose analysis mode:
#   "text"             → r_text (cross_attention model)
#   "audio"            → r_audio
#   "delta_text_audio" → r_text_audio - max(r_text, r_audio)   [multimodal integration]
#   "delta_emb"        → r_cross_attention - r_statement_only  [contextual benefit]
#   "all"              → runs all four analyses above
MODE = "all"

# ================================
# Shared resources (loaded once)
# ================================
icbm       = datasets.fetch_icbm152_2009()
example    = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
brain_mask = resample_to_img(image.load_img(icbm['mask']), example, interpolation='nearest').get_fdata() > 0
affine     = example.affine

i_coords, j_coords, k_coords = np.where(brain_mask)
n_voxels    = len(i_coords)
ijk_to_flat = {(i_coords[v], j_coords[v], k_coords[v]): v for v in range(n_voxels)}

aal      = datasets.fetch_atlas_aal('SPM12')
aal_img  = load_img(aal.maps)
aal_inv  = np.linalg.inv(aal_img.affine)
aal_data = aal_img.get_fdata()

# ================================
# Helpers
# ================================
def load_corr(feature_str, emb):
    return np.load(f"{RESULTS_DIR}/correlation_map_flat_{feature_str}_{emb}_{N_SPLITS}.npy")

def load_perm(feature_str, emb):
    return np.load(f"{RESULTS_DIR}/permutation_results/perm_scores_{feature_str}_{emb}.npy")

def get_aal(mni):
    try:
        vox = tuple(int(round(c)) for c in coord_transform(*mni, aal_inv))
        idx = int(aal_data[vox])
        return aal.labels[aal.indices.index(str(idx))] if str(idx) in aal.indices else "Unknown"
    except Exception:
        return "Unknown"

def format_pval(p):
    if p < 0.001:  return "<.001"
    elif p < 0.01: return f"{p:.3f}".lstrip("0")
    else:          return f"{p:.4f}".lstrip("0")

def build_obs_and_null(mode):
    """Return (obs, perm_null, stat_label) for a given mode."""
    if mode == "text":
        obs       = load_corr("text_base", EMB_PRIMARY)
        perm_null = load_perm("text_base", EMB_PRIMARY)
        label_str = f"r_text ({EMB_PRIMARY})"

    elif mode == "audio":
        obs       = load_corr("audio_base", EMB_PRIMARY)
        perm_null = load_perm("audio_base", EMB_PRIMARY)
        label_str = "r_audio"

    elif mode == "delta_text_audio":
        r_comb  = load_corr("text_audio_base", EMB_PRIMARY)
        r_text  = load_corr("text_base",       EMB_PRIMARY)
        r_audio = load_corr("audio_base",      EMB_PRIMARY)
        obs     = r_comb - np.maximum(r_text, r_audio)

        p_comb  = load_perm("text_audio_base", EMB_PRIMARY)
        p_text  = load_perm("text_base",       EMB_PRIMARY)
        p_audio = load_perm("audio_base",      EMB_PRIMARY)
        perm_null = p_comb - np.maximum(p_text, p_audio)
        label_str = f"Δr text_audio - max(text,audio) [{EMB_PRIMARY}]"

    elif mode == "delta_emb":
        obs       = load_corr("text_base", EMB_PRIMARY) - load_corr("text_base", EMB_COMPARE)
        perm_null = load_perm("text_base", EMB_PRIMARY) - load_perm("text_base", EMB_COMPARE)
        label_str = f"Δr {EMB_PRIMARY} - {EMB_COMPARE}"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return obs, perm_null, label_str


def run_analysis(mode):
    print(f"\n{'='*110}")
    print(f"ANALYSIS: {mode.upper()}")
    print(f"{'='*110}")

    obs, perm_null, stat_label = build_obs_and_null(mode)
    print(f"  {stat_label}  |  obs range [{obs.min():.4f}, {obs.max():.4f}]")

    # --- p-values + FDR ---
    pvals                = (perm_null >= obs[np.newaxis, :]).mean(axis=0)
    reject_fdr, pvals_fdr = fdrcorrection(pvals, alpha=FDR_ALPHA)
    print(f"  FDR q<{FDR_ALPHA}: {reject_fdr.sum():,} / {len(pvals):,} significant voxels")

    # --- project to 3D ---
    obs_3d    = np.zeros(brain_mask.shape)
    reject_3d = np.zeros(brain_mask.shape, dtype=bool)
    obs_3d[brain_mask]    = obs
    reject_3d[brain_mask] = reject_fdr

    # --- clustering mask ---
    if TOP_PERCENTILE is not None:
        threshold  = np.percentile(obs, TOP_PERCENTILE)
        to_cluster = obs_3d > threshold
        mode_label = f"top{(100-TOP_PERCENTILE):.1f}pct_FDR{int(FDR_ALPHA*100)}"
        print(f"  Top {100-TOP_PERCENTILE:.1f}% threshold = {threshold:.6f} → {to_cluster.sum():,} voxels")
    else:
        to_cluster = reject_3d
        mode_label = f"FDR{int(FDR_ALPHA*100)}"
        print(f"  No percentile filter → {to_cluster.sum():,} FDR-significant voxels entering clustering")

    output_name = f"{mode}_{mode_label}_k{MIN_CLUSTER_SIZE}_{EMB_PRIMARY}"

    labeled, n_clusters = label(to_cluster, structure=np.ones((3, 3, 3), bool))
    print(f"  Connected clusters before size filter: {n_clusters}")

    # --- filter clusters ---
    valid_clusters = []
    for lbl in range(1, n_clusters + 1):
        mask = (labeled == lbl)
        size = int(mask.sum())
        if size < MIN_CLUSTER_SIZE:
            continue

        vals     = obs_3d[mask]
        peak_val = vals.max()
        peak_ijk = np.array(np.where((obs_3d == peak_val) & mask))[:, 0]
        i, j, k  = peak_ijk

        flat_peak = ijk_to_flat.get((i, j, k))
        if flat_peak is None or not reject_fdr[flat_peak]:
            continue

        valid_clusters.append({
            "label":    lbl,
            "size":     size,
            "mean_r":   float(vals.mean()),
            "peak_r":   float(peak_val),
            "peak_mni": apply_affine(affine, (i, j, k)),
            "com_mni":  apply_affine(affine, center_of_mass(mask)),
            "peak_q":   float(pvals_fdr[flat_peak]),
        })

    # --- table ---
    valid_clusters = sorted(valid_clusters, key=lambda x: x["peak_r"], reverse=True)
    col_stat = "Peak Δr" if mode.startswith("delta") else "Peak r"
    col_mean = "Mean Δr" if mode.startswith("delta") else "Mean r"

    table = []
    for rank, c in enumerate(valid_clusters, 1):
        region = get_aal(c["peak_mni"])
        table.append({
            "#":          rank,
            "Size (vox)": c["size"],
            col_mean:     round(c["mean_r"], 5),
            col_stat:     round(c["peak_r"], 5),
            "Peak FDR q": format_pval(c["peak_q"]),
            "Peak MNI":   tuple(round(x, 1) for x in c["peak_mni"]),
            "CoM MNI":    tuple(round(x, 1) for x in c["com_mni"]),
            "AAL Region": region,
        })

    df = pd.DataFrame(table)
    print(f"\n  {len(valid_clusters)} clusters kept ({mode_label}, k≥{MIN_CLUSTER_SIZE})")
    print(df.to_string(index=False))

    # --- save ---
    df.to_excel(f"{RESULTS_DIR}/{output_name}.xlsx", index=False)

    final_map = np.zeros_like(obs_3d)
    for c in valid_clusters:
        final_map[labeled == c["label"]] = obs_3d[labeled == c["label"]]
    nib.save(nib.Nifti1Image(final_map, affine), f"{RESULTS_DIR}/{output_name}.nii")

    print(f"\n  Saved: {RESULTS_DIR}/{output_name}.{{xlsx,nii}}")
    return df


# ================================
# Run
# ================================
ALL_MODES = ["text", "audio", "delta_text_audio", "delta_emb"]
modes_to_run = ALL_MODES if MODE == "all" else [MODE]

all_results = {}
for m in modes_to_run:
    all_results[m] = run_analysis(m)

print(f"\n{'='*110}")
print(f"Done — {len(modes_to_run)} analyses saved to {RESULTS_DIR}/")
print(f"{'='*110}")
