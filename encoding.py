import os
import time
import argparse
import numpy as np
import analysis_helpers
import nibabel as nib
from nilearn.image import resample_to_img
import logging
from ridge_cv import ridge_cv
from nilearn import datasets, image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
ridge_logger = logging.getLogger("ridge_corr")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # Dataset-related arguments
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                               help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--use_text", action="store_true", 
                               help="Include text in dataset (default: False).")
    dataset_group.add_argument("--use_audio", action="store_true",
                               help="Include audio in dataset (default: False).")
    dataset_group.add_argument("--text_embedding_type", type=str, default="cross_attention",
                               choices=["cross_attention", "joint_encoding", "statement_only"],
                               help="Text embedding approach to use (default: cross_attention).")
    dataset_group.add_argument("--use_pca", action="store_true", 
                               help="Use PCA for embeddings with a certain amount of explained variance directly in the dataset method (default: False).")
    dataset_group.add_argument("--pca_threshold", type=float, default=0.60,
                               help="Explained variance threshold for PCA (default: 0.60).")
    dataset_group.add_argument("--include_tasks", type=str, nargs='+', default=["sarcasm", "irony", "prosody", "semantic", "tom"],
                               help="List of tasks to include (default: all available tasks).")
    # Analysis-related arguments
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--num_jobs", type=int, default=-1,
                               help="Number of parallel jobs for voxel processing (default: -1 for all cores).")
    analysis_group.add_argument("--optimize_alpha", action="store_true",
                               help="Optimize alpha values using LOO bootstrapping (default: False, use precomputed valphas if provided).")
    analysis_group.add_argument("--alpha_min", type=float, default=-2,
                               help="Minimum exponent for alpha values in logspace (default: -2).")
    analysis_group.add_argument("--alpha_max", type=float, default=3,
                               help="Maximum exponent for alpha values in logspace (default: 3).")
    analysis_group.add_argument("--num_alphas", type=int, default=10,
                               help="Number of alpha values to test in logspace (default: 10).")
    analysis_group.add_argument("--nboots", type=int, default=None,
                               help="Number of bootstrap iterations (default: number of participants).")
    analysis_group.add_argument("--corrmin", type=float, default=0.0,
                               help="Minimum correlation threshold for fMRI correlations (default: 0.0).")
    analysis_group.add_argument("--n_splits", type=int, default=None,
                               help="Number of splits for cross-validation (default: number of participants for LOO CV).")
    analysis_group.add_argument("--normalpha", action="store_true", default = True,
                               help="Normalize alpha values for reuse across models (default: False).")
    analysis_group.add_argument("--use_corr", action="store_true", default = True,
                               help="Use correlation as the evaluation metric (default: False).")
    analysis_group.add_argument("--return_wt", action="store_true",
                               help="Return weight maps from ridge regression (default: False).")
    analysis_group.add_argument("--normalize_resp", action="store_true", default=True,
                               help="Normalize response (fMRI) data before regression (default: True).")
    analysis_group.add_argument("--with_replacement", action="store_true",
                               help="Perform bootstrapping with replacement (default: False).")
    analysis_group.add_argument("--results_dir", type=str, default="results",
                               help="Custom directory to save results (default: uses paths from analysis_helpers).")


    return parser.parse_args()

def main():
    start_time = time.time()  # Start timing

    args = parse_arguments()

    # Print settings, including new arguments
    print(f"Running with settings:\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- Use text: {args.use_text}\n"
          f"- Use audio: {args.use_audio}\n"
          f"- Use PCA: {args.use_pca}\n"
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Number of jobs: {args.num_jobs}\n"
          f"- Optimize alpha: {args.optimize_alpha}\n"
          f"- Alpha range: 10^{args.alpha_min} to 10^{args.alpha_max} with {args.num_alphas} values\n"
          f"- Number of bootstraps: {args.nboots if args.nboots is not None else 'num_participants'}\n"
          f"- Correlation minimum: {args.corrmin}\n"
          f"- Number of CV splits: {args.n_splits if args.n_splits is not None else 'num_participants'}\n"
          f"- Normalize alphas: {args.normalpha}\n"
          f"- Use correlation metric: {args.use_corr}\n"
          f"- Return weights: {args.return_wt}\n"
          f"- Normalize response: {args.normalize_resp}\n"
          f"- Bootstrap with replacement: {args.with_replacement}\n"
          f"- Results directory: {args.results_dir if args.results_dir else 'default'}\n"
          f"- Included tasks: {', '.join(args.include_tasks)}")

    paths = analysis_helpers.get_paths(text_embedding_type=args.text_embedding_type)
    participant_list = os.listdir(paths["data_path"])

    icbm = datasets.fetch_icbm152_2009()
    mask_path = icbm['mask']
    # Load the mask as a Nifti image object
    mask = image.load_img(mask_path)
    
    exemple_data = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
    resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')

    stim_df, resp, ids_list, col_groups = analysis_helpers.load_dataset(args, paths, participant_list, resampled_mask)

    # Set alphas based on arguments
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)

    # Handle precomputed alphas
    if not args.optimize_alpha:
        valphas_path = os.path.join(args.results_dir, f"valphas_text_audio_base.npy")
        if not os.path.exists(valphas_path):
            raise ValueError("Must provide a valid --precomputed_valphas path when --optimize_alpha is False.")
        valphas = np.load(valphas_path)
        ridge_logger.info(f"Using precomputed valphas at {valphas_path}")
    else:
        valphas = None

    # Set nboots and n_splits based on arguments or default to number of participants
    nboots = args.nboots if args.nboots is not None else len(participant_list)
    n_splits = args.n_splits if args.n_splits is not None else len(participant_list)

    # Perform ridge regression with LOO CV
    weights, corrs, valphas, fold_corrs, _ = ridge_cv(
        stim_df=stim_df,
        resp=resp,
        alphas=alphas,
        participant_ids=ids_list,
        col_groups=col_groups,
        use_pca=args.use_pca,
        pca_threshold=args.pca_threshold,
        nboots=nboots,
        corrmin=args.corrmin,
        n_splits=n_splits,
        singcutoff=1e-10,
        normalpha=args.normalpha,
        use_corr=args.use_corr,
        return_wt=args.return_wt,
        normalize_resp=args.normalize_resp,
        n_jobs=args.num_jobs,
        with_replacement=args.with_replacement,
        optimize_alpha=args.optimize_alpha,
        valphas=valphas,
        logger=ridge_logger
    )
    
    features_used = []
    if args.use_text:
        features_used.append("text")
    if args.use_audio:
        features_used.append("audio")
    if args.use_base_features:
        features_used.append("base")
    feature_str = "_".join(features_used) if features_used else "nofeatures"
    
    # Set results directory
    os.makedirs(args.results_dir , exist_ok=True)

    
    # Save corrs (mean correlations across folds) in flattened space
    np.save(os.path.join(args.results_dir, f"correlation_map_flat_{feature_str}_{args.n_splits}.npy"), corrs)
    np.save(os.path.join(args.results_dir, f"folds_correlation_map_flat_{feature_str}_{args.n_splits}.npy"), fold_corrs)
    ridge_logger.info(f"Saved flattened correlations to {args.results_dir}/correlation_map_flat_{feature_str}.npy")
    
    # Save valphas
    if args.optimize_alpha:
        result_file_valphas = os.path.join(args.results_dir, f"valphas_{feature_str}.npy")
        np.save(result_file_valphas, valphas)
        ridge_logger.info(f"Saved valphas to {result_file_valphas}")
    
    end_time = time.time()
    print("Total r2: %d" % sum(corrs * np.abs(corrs)))
    print(f"Analysis completed in {(end_time - start_time) / 60:.2f} minutes.")

# Run the script
if __name__ == "__main__":
    main()