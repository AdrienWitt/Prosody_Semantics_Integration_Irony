import os
import time
import argparse
import numpy as np
import pandas as pd
import logging
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import resample_to_img
from joblib import Parallel, delayed

import analysis_helpers
from ridge_cv import ridge_cv

# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Permutation test for multimodal fMRI encoding models."
    )
    parser.add_argument("--use_text", action="store_true")
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_base_features", action="store_true")
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--pca_threshold", type=float, default=0.6)
    parser.add_argument("--include_tasks", type=str, nargs="+", default=["irony", "sarcasm"])
    parser.add_argument("--n_splits", type=int, default=None,
                               help="Number of splits for cross-validation (default: number of participants for LOO CV).")
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_jobs", type=int, default=7,
                               help="Outer parallel jobs for permutations.")
    parser.add_argument("--include_mod", type=str, nargs="+", default=["text", "audio", "text_audio"],
                               choices=["text", "audio", "text_audio"],
                               help="Modalities to include: 'text', 'audio', or 'text_audio'. Can specify multiple.")
    parser.add_argument("--corrmin", type=float, default=0.0)
    parser.add_argument("--normalpha", action="store_true", default=True)
    parser.add_argument("--use_corr", action="store_true", default=True)
    parser.add_argument("--normalize_stim", action="store_true")
    parser.add_argument("--normalize_resp", action="store_true", default=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--text_embedding_type", type=str, default="cross_attention",
                               choices=["cross_attention", "joint_encoding", "statement_only"],
                               help="Primary text embedding type.")
    parser.add_argument("--compare_embedding_type", type=str, default=None,
                               choices=["cross_attention", "joint_encoding", "statement_only"],
                               help="Optional second text embedding type to compare against the primary. "
                                    "Both are shuffled with the same permutation index so that "
                                    "delta = r_primary - r_compare has a valid null distribution.")
    return parser.parse_args()


# ----------------------------------------------------------------------
# One permutation: permute modality blocks independently per participant
# ----------------------------------------------------------------------
def run_one_permutation(
    stim_df: pd.DataFrame,
    resp: np.ndarray,
    ids_list: np.ndarray,
    cols_text: list,
    cols_audio: list,
    cols_combined: list,
    valphas_per_mod: dict,
    args: argparse.Namespace,
    seed: int,
    col_groups: dict = None,
    # --- optional comparison embedding ---
    stim_df2: pd.DataFrame = None,
    cols_text2: list = None,
    cols_combined2: list = None,
):
    rng = np.random.RandomState(seed)
    stim_perm = stim_df.copy()
    stim_perm2 = stim_df2.copy() if stim_df2 is not None else None

    # --- Within-participant shuffling ---
    # text and audio use independent shuffles;
    # both embedding types share the SAME text shuffle so delta is valid.
    for pid in np.unique(ids_list):
        idx = np.where(ids_list == pid)[0]

        if "text" in args.include_mod or "text_audio" in args.include_mod or stim_perm2 is not None:
            perm_idx_text = rng.permutation(idx)
            stim_perm.loc[idx, cols_text] = stim_perm.loc[perm_idx_text, cols_text].values
            # apply the SAME shuffle to the comparison embedding
            if stim_perm2 is not None:
                stim_perm2.loc[idx, cols_text2] = stim_perm2.loc[perm_idx_text, cols_text2].values
        else:
            rng.permutation(idx)  # keep rng state aligned

        if "audio" in args.include_mod or "text_audio" in args.include_mod:
            perm_idx_audio = rng.permutation(idx)
            stim_perm.loc[idx, cols_audio] = stim_perm.loc[perm_idx_audio, cols_audio].values

    def _cg(cols):
        """Filter col_groups to only columns present in cols."""
        if col_groups is None:
            return {}
        s = set(cols)
        return {k: [c for c in v if c in s] for k, v in col_groups.items()}

    _ridge_kwargs_base = dict(
        resp=resp,
        alphas=None,
        participant_ids=ids_list,
        n_lopo=0,
        n_splits=args.n_splits,
        corrmin=args.corrmin,
        singcutoff=1e-10,
        normalpha=args.normalpha,
        use_corr=args.use_corr,
        return_wt=False,
        normalize_stim=args.normalize_stim,
        normalize_resp=args.normalize_resp,
        n_jobs=-1,
        with_replacement=False,
        optimize_alpha=False,
        logger=ridge_logger,
    )

    results = {}

    if "text" in args.include_mod:
        _, corr_text, _, _, _ = ridge_cv(stim_df=stim_perm[cols_text],
                                          col_groups=_cg(cols_text),
                                          valphas=valphas_per_mod["text"], **_ridge_kwargs_base)
        results["text"] = corr_text

    if "audio" in args.include_mod:
        _, corr_audio, _, _, _ = ridge_cv(stim_df=stim_perm[cols_audio],
                                           col_groups=_cg(cols_audio),
                                           valphas=valphas_per_mod["audio"], **_ridge_kwargs_base)
        results["audio"] = corr_audio

    if "text_audio" in args.include_mod:
        _, corr_comb, _, _, _ = ridge_cv(stim_df=stim_perm[cols_combined],
                                          col_groups=_cg(cols_combined),
                                          valphas=valphas_per_mod["text_audio"], **_ridge_kwargs_base)
        results["text_audio"] = corr_comb

    # --- Comparison embedding type (same text shuffle, independent ridge fit) ---
    if stim_perm2 is not None:
        if "text" in args.include_mod:
            _, corr_text2, _, _, _ = ridge_cv(stim_df=stim_perm2[cols_text2],
                                               col_groups=_cg(cols_text2),
                                               valphas=valphas_per_mod["text_compare"], **_ridge_kwargs_base)
            results["text_compare"] = corr_text2

        if "text_audio" in args.include_mod:
            _, corr_comb2, _, _, _ = ridge_cv(stim_df=stim_perm2[cols_combined2],
                                               col_groups=_cg(cols_combined2),
                                               valphas=valphas_per_mod["text_audio_compare"], **_ridge_kwargs_base)
            results["text_audio_compare"] = corr_comb2

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    start_time = time.time()

    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("perm_test")
    global ridge_logger
    ridge_logger = logging.getLogger("ridge_corr")
    logger.info("=== Starting permutation test ===")

    # --- Load primary dataset ---
    paths = analysis_helpers.get_paths(text_embedding_type=args.text_embedding_type)
    participant_list = sorted(os.listdir(paths["data_path"]))

    icbm = datasets.fetch_icbm152_2009()
    mask = image.load_img(icbm["mask"])
    example_nii = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
    resampled_mask = resample_to_img(mask, example_nii, interpolation="nearest")

    stim_df, resp, ids_list, col_groups = analysis_helpers.load_dataset(
        args, paths, participant_list, resampled_mask
    )

    cols_text_emb   = [c for c in stim_df.columns if c.startswith(("emb_weighted_", "pc_weighted_"))]
    cols_text_base  = [c for c in stim_df.columns if c.startswith(("context_", "semantic_"))]
    cols_audio_emb  = [c for c in stim_df.columns if c.startswith(("emb_audio_opensmile_", "pc_audio_opensmile_"))]
    cols_audio_base = [c for c in stim_df.columns if c.startswith("prosody_")]

    cols_text     = cols_text_emb + cols_text_base
    cols_audio    = cols_audio_emb + cols_audio_base
    cols_combined = cols_text + cols_audio

    logger.info(
        f"Primary ({args.text_embedding_type}) — "
        f"text: {len(cols_text)} features | audio: {len(cols_audio)} features"
    )
    logger.info(f"Running with modalities: {', '.join(args.include_mod)}")

    base_suffix = "_base" if args.use_base_features else ""
    emb = args.text_embedding_type

    def _load_valphas(feature_str, emb_type):
        path = os.path.join(args.results_dir, f"valphas_{feature_str}{base_suffix}_{emb_type}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Valphas file not found: {path}")
        return np.load(path)

    valphas_per_mod = {}
    if "text" in args.include_mod:
        valphas_per_mod["text"] = _load_valphas("text", emb)
    if "audio" in args.include_mod:
        valphas_per_mod["audio"] = _load_valphas("audio", emb)
    if "text_audio" in args.include_mod:
        valphas_per_mod["text_audio"] = _load_valphas("text_audio", emb)

    # --- Load comparison embedding dataset (if requested) ---
    stim_df2 = cols_text2 = cols_combined2 = None
    emb2 = args.compare_embedding_type

    if emb2 is not None:
        logger.info(f"Loading comparison embedding: {emb2}")
        args2 = argparse.Namespace(**vars(args))  # shallow copy of args
        args2.text_embedding_type = emb2
        paths2 = analysis_helpers.get_paths(text_embedding_type=emb2)
        stim_df2, _, _, _ = analysis_helpers.load_dataset(
            args2, paths2, participant_list, resampled_mask
        )
        # audio/base columns are shared; only text embeddings differ
        cols_text_emb2  = [c for c in stim_df2.columns if c.startswith(("emb_weighted_", "pc_weighted_"))]
        cols_text2      = cols_text_emb2 + cols_text_base   # base cols are the same
        cols_combined2  = cols_text2 + cols_audio

        if "text" in args.include_mod:
            valphas_per_mod["text_compare"] = _load_valphas("text", emb2)
        if "text_audio" in args.include_mod:
            valphas_per_mod["text_audio_compare"] = _load_valphas("text_audio", emb2)

        logger.info(
            f"Comparison ({emb2}) — text: {len(cols_text2)} features"
        )

    # --- Run permutations ---
    rng = np.random.RandomState(args.random_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.n_perms)

    perm_results = Parallel(n_jobs=args.num_jobs)(
        delayed(run_one_permutation)(
            stim_df, resp, ids_list,
            cols_text, cols_audio, cols_combined,
            valphas_per_mod, args, seed,
            col_groups=col_groups,
            stim_df2=stim_df2,
            cols_text2=cols_text2,
            cols_combined2=cols_combined2,
        )
        for seed in seeds
    )

    # Collect all keys that were actually computed
    all_keys = list(perm_results[0].keys())
    perm_scores = {
        key: np.vstack([r[key] for r in perm_results])
        for key in all_keys
    }
    logger.info(f"Completed {args.n_perms} permutations.")

    # --- Save results ---
    results_path = args.results_dir if args.results_dir else paths["results_path"]
    perm_dir = os.path.join(results_path, "permutation_results")
    os.makedirs(perm_dir, exist_ok=True)

    # Primary modalities
    for mod in args.include_mod:
        if mod in perm_scores:
            out_file = os.path.join(perm_dir, f"perm_scores_{mod}{base_suffix}_{emb}.npy")
            np.save(out_file, perm_scores[mod])
            logger.info(f"Saved {out_file} — shape {perm_scores[mod].shape}")

    # Comparison embedding modalities
    if emb2 is not None:
        for mod in args.include_mod:
            key = f"{mod}_compare"
            if key in perm_scores:
                out_file = os.path.join(perm_dir, f"perm_scores_{mod}{base_suffix}_{emb2}.npy")
                np.save(out_file, perm_scores[key])
                logger.info(f"Saved {out_file} — shape {perm_scores[key].shape}")

    logger.info(f"Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
