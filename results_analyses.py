import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image
from nilearn.image import resample_to_img, load_img, coord_transform
from scipy.ndimage import label, center_of_mass
from nibabel.affines import apply_affine
from statsmodels.stats.multitest import fdrcorrection

# ================================
# SETTINGS 
# ================================
TOP_PERCENTILE   = 99.5        # 99.5 = top 0.5%, 99.9 = top 0.1%
MIN_CLUSTER_SIZE = 5           # minimum cluster size
OUTPUT_NAME      = f"top{(100 - TOP_PERCENTILE):.1f}pct_k{MIN_CLUSTER_SIZE}_ALLvox_FDR05"


# ================================
# 1. Load correlation maps
# ================================

results_dir = "results_nopca"
r_comb = np.load(f"{results_dir}/correlation_map_flat_text_audio_base_5.npy")
r_text = np.load(f"{results_dir}/correlation_map_flat_text_base_5.npy")
r_audio = np.load(f"{results_dir}/correlation_map_flat_audio_base_5.npy")
delta_r = r_comb - np.maximum(r_text, r_audio)

# 2. Load permutation results and compute p-values
# ================================
perm_text  = np.load(f"{results_dir}/permutation_results/perm_scores_text_base.npy")
perm_audio = np.load(f"{results_dir}/permutation_results/perm_scores_audio_base.npy")
perm_comb  = np.load(f"{results_dir}/permutation_results/perm_scores_text_audio_base.npy")

# Permutation distribution of delta_r (same formula as observed)
perm_delta_r = perm_comb - np.maximum(perm_text, perm_audio)

# One-tailed p-value: fraction of permutations >= observed delta_r
pvals = np.mean(perm_delta_r >= delta_r[np.newaxis, :], axis=0)

# FDR correction (Benjamini-Hochberg, q < .05)
reject_fdr, pvals_fdr = fdrcorrection(pvals, alpha=0.05)


# ================================
# 3. Brain mask + coordinate mapping
# ================================
icbm = datasets.fetch_icbm152_2009()
example = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
brain_mask = resample_to_img(image.load_img(icbm['mask']), example, interpolation='nearest').get_fdata() > 0
affine = example.affine


i_coords, j_coords, k_coords = np.where(brain_mask)
n_voxels = len(i_coords)


ijk_to_flat = {}
for idx in range(n_voxels):
    ijk_to_flat[(i_coords[idx], j_coords[idx], k_coords[idx])] = idx

print(f"Brain mask has {n_voxels:,} voxels")

# ================================
# 4. Threshold at top percentile
# ================================
delta_r_3d = np.zeros(brain_mask.shape, dtype=float)
delta_r_3d[brain_mask] = delta_r

threshold = np.percentile(delta_r, TOP_PERCENTILE)
suprathresh = delta_r_3d > threshold
n_supra = suprathresh.sum()
print(f"Top {100-TOP_PERCENTILE:.1f}% threshold = {threshold:.6f} → {n_supra:,} voxels")

# ================================
# 5. Clustering
# ================================
labeled, n_clusters = label(suprathresh, structure=np.ones((3,3,3), bool))

# ================================
# 6. Filter: ALL voxels must be FDR-significant
# ================================
valid_clusters = []
print(f"\nFiltering {n_clusters} clusters → keeping only k ≥ {MIN_CLUSTER_SIZE} with ALL voxels q < .05...")

for lbl in range(1, n_clusters + 1):
    mask = (labeled == lbl)
    size = mask.sum()
    if size < MIN_CLUSTER_SIZE:
        continue

    ijk_list = list(zip(*np.where(mask)))
    
    flat_idxs = []
    for ijk in ijk_list:
        if ijk in ijk_to_flat:
            flat_idxs.append(ijk_to_flat[ijk])

    if len(flat_idxs) > 0 and not np.all(reject_fdr[flat_idxs]):
        continue

    vals = delta_r_3d[mask]
    peak_val = vals.max()
    peak_ijk = np.where((delta_r_3d == peak_val) & mask)
    i, j, k = peak_ijk[0][0], peak_ijk[1][0], peak_ijk[2][0]

    peak_mni = apply_affine(affine, (i, j, k))
    com_mni = apply_affine(affine, center_of_mass(mask))
    peak_q = pvals_fdr[ijk_to_flat[(i, j, k)]]

    valid_clusters.append({
        "label": lbl,
        "size": size,
        "mean_Δr": vals.mean(),
        "peak_Δr": peak_val,
        "peak_mni": peak_mni,
        "com_mni": com_mni,
        "peak_q": peak_q
    })
    print(f"  Kept cluster {lbl:3d} | k={size:4d} | peak Δr={peak_val:.5f} | q={peak_q:.3f}")

# ================================
# 7. P-value formatting
# ================================
def format_pval(p):
    if p < 0.001:
        return "<.001"
    elif p < 0.01:
        return f"{p:.3f}".lstrip("0") 
    else:
        return f"{p:.4f}".lstrip("0")  

# ================================
# 8. AAL labeling
# ================================
def get_aal(mni):
    try:
        aal = datasets.fetch_atlas_aal('SPM12')
        img = load_img(aal.maps)
        vox = tuple(int(round(c)) for c in coord_transform(*mni, np.linalg.inv(img.affine)))
        idx = int(img.get_fdata()[vox])
        return aal.labels[aal.indices.index(str(idx))] if str(idx) in aal.indices else "Unknown"
    except:
        return "Unknown"

# ================================
# 9. Final table
# ================================
valid_clusters = sorted(valid_clusters, key=lambda x: x['mean_Δr'], reverse=True)

table = []
for rank, c in enumerate(valid_clusters, 1):
    region = get_aal(c['peak_mni'])
    table.append({
        '#': rank,
        'Size (vox)': c['size'],
        'Mean Δr': round(c['mean_Δr'], 5),
        'Peak Δr': round(c['peak_Δr'], 5),
        'Peak FDR q': format_pval(c['peak_q']),
        'Peak MNI': tuple(round(x, 1) for x in c['peak_mni']),
        'CoM MNI': tuple(round(x, 1) for x in c['com_mni']),
        'AAL Region': region
    })

df = pd.DataFrame(table)
print("\n" + "="*110)
print(f"FINAL RESULT – {len(valid_clusters)} clusters (top {100-TOP_PERCENTILE:.1f}%, all voxels q < .05, k ≥ {MIN_CLUSTER_SIZE})")
print("="*110)
print(df.to_string(index=False))
print("="*110)

# Save
df.to_excel(f"{results_dir}/{OUTPUT_NAME}.xlsx", index=False)

final_map = np.zeros_like(delta_r_3d)
for c in valid_clusters:
    final_map[labeled == c['label']] = delta_r_3d[labeled == c['label']]
nib.save(nib.Nifti1Image(final_map, affine), f"{results_dir}/{OUTPUT_NAME}.nii")

print(f"\nResults saved:")
print(f"   Table → {results_dir}/{OUTPUT_NAME}.xlsx")
print(f"   Map   → {results_dir}/{OUTPUT_NAME}.nii")
print(f"   Clusters: {len(valid_clusters)}")