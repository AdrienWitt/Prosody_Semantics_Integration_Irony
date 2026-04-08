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
                               help="Outer parallel jobs for permutations. "
                                    "Suggested: 7 for model 3 (3 ridge_cv/perm), "
                                    "14-21 for models 1 or 2 (1 ridge_cv/perm).")
    parser.add_argument("--include_mod", type=str, nargs="+", default=["text", "audio", "text_audio"],
                               choices=["text", "audio", "text_audio"],
                               help="Modalities to include: 'text', 'audio', or 'text_audio'. Can specify multiple.")
    parser.add_argument("--corrmin", type=float, default=0.0)
    parser.add_argument("--normalpha", action="store_true", default=True)
    parser.add_argument("--use_corr", action="store_true", default=True)
    parser.add_argument("--normalize_stim", action="store_true")
    parser.add_argument("--normalize_resp", action="store_true", default=True)
    parser.add_argument("--results_dir", type=str, default=None)
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
    valphas: np.ndarray,
    args: argparse.Namespace,
    seed: int,
):
    rng = np.random.RandomState(seed)
    stim_perm = stim_df.copy()

    # --- Within-participant independent shuffling per modality ---
    for pid in np.unique(ids_list):
        idx = np.where(ids_list == pid)[0]

        if "text" in args.include_mod:
            perm_idx_text = rng.permutation(idx)
            stim_perm.loc[idx, cols_text] = stim_perm.loc[perm_idx_text, cols_text].values

        if "audio" in args.include_mod:
            perm_idx_audio = rng.permutation(idx)
            stim_perm.loc[idx, cols_audio] = stim_perm.loc[perm_idx_audio, cols_audio].values

    _ridge_kwargs = dict(
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
        valphas=valphas,
        logger=ridge_logger,
    )

    # --- Ridge CV on permuted data (only the needed modalities) ---
    results = {}

    if "text" in args.include_mod:
        _, corr_text, _, _, _ = ridge_cv(stim_df=stim_perm[cols_text], **_ridge_kwargs)
        results["text"] = corr_text

    if "audio" in args.include_mod:
        _, corr_audio, _, _, _ = ridge_cv(stim_df=stim_perm[cols_audio], **_ridge_kwargs)
        results["audio"] = corr_audio

    if "text_audio" in args.include_mod:
        _, corr_comb, _, _, _ = ridge_cv(stim_df=stim_perm[cols_combined], **_ridge_kwargs)
        results["text_audio"] = corr_comb

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    start_time = time.time()

    # --- Args & logging ---
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("perm_test")
    global ridge_logger
    ridge_logger = logging.getLogger("ridge_corr")
    logger.info("=== Starting permutation test ===")

    # --- Load data ---
    paths = analysis_helpers.get_paths()
    participant_list = sorted(os.listdir(paths["data_path"]))

    icbm = datasets.fetch_icbm152_2009()
    mask = image.load_img(icbm["mask"])
    example_nii = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")

    resampled_mask = resample_to_img(mask, example_nii, interpolation="nearest")

    stim_df, resp, ids_list = analysis_helpers.load_dataset(
        args, paths, participant_list, resampled_mask
    )

    cols_text_emb = [c for c in stim_df.columns if c.startswith(("emb_weighted_", "pc_weighted_"))]
    cols_text_base = [c for c in stim_df.columns if c.startswith(("context_", "semantic_"))]
    cols_audio_emb = [c for c in stim_df.columns if c.startswith(("emb_audio_opensmile_", "pc_audio_opensmile_"))]
    cols_audio_base = [c for c in stim_df.columns if c.startswith("prosody_")]

    cols_text = cols_text_emb + cols_text_base
    cols_audio = cols_audio_emb + cols_audio_base
    cols_combined = cols_text + cols_audio

    logger.info(
        f"Text model: {len(cols_text)} features "
        f"(emb: {len(cols_text_emb)}, base: {len(cols_text_base)}) | "
        f"Audio model: {len(cols_audio)} features "
        f"(emb: {len(cols_audio_emb)}, base: {len(cols_audio_base)})"
    )
    logger.info(f"Running with modalities: {', '.join(args.include_mod)}")
    logger.info(f"Suggested --num_jobs: 7 (3 modalities) or 14-21 (1-2 modalities). Currently: {args.num_jobs}")

    # --- Load valphas ---
    valphas = np.load(os.path.join(args.results_dir, "valphas_text_audio_base.npy"))

    # --- Run permutations ---
    rng = np.random.RandomState(args.random_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.n_perms)

    perm_results = Parallel(n_jobs=args.num_jobs)(
        delayed(run_one_permutation)(
            stim_df, resp, ids_list,
            cols_text, cols_audio, cols_combined,
            valphas, args, seed
        )
        for seed in seeds
    )

    # Restructure: list of dicts -> dict of (n_perms, n_voxels) arrays
    perm_scores = {
        mod: np.vstack([r[mod] for r in perm_results])
        for mod in args.include_mod
    }
    logger.info(f"Completed {args.n_perms} permutations.")

    # --- Save one .npy per modality ---
    results_path = args.results_dir if args.results_dir else paths["results_path"]
    perm_dir = os.path.join(results_path, "permutation_results")
    os.makedirs(perm_dir, exist_ok=True)

    for mod in args.include_mod:
        out_file = os.path.join(perm_dir, f"perm_scores_{mod}_base.npy")
        np.save(out_file, perm_scores[mod])
        logger.info(f"Saved {out_file} — shape {perm_scores[mod].shape}")

    logger.info(f"Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
