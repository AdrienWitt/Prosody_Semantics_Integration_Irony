import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import concat_imgs
from nilearn import signal as nilearn_signal
from nilearn import image as nilearn_image
from nilearn.maskers import NiftiMasker
import pandas as pd
from nilearn.glm.first_level import compute_regressor

# Define paths
folder_fmri = r'D:\Preproc_Analyses\data_done'
files_type = 'swrMF'  # smoothed, warped, realigned — update prefix if needed
output_dir_fmri = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\Clean_Project\data\fmri'
os.makedirs(output_dir_fmri, exist_ok=True)

TR = 0.65  # Repetition time in seconds


def select_files(root_folder, files_type):
    participant_folders = glob.glob(os.path.join(root_folder, 'p*'))
    participant_files = {}
    for participant_folder in participant_folders:
        participant = participant_folder[-3:]
        run_folders = glob.glob(os.path.join(participant_folder, 'RUN*'))
        run_files = {}
        for run_folder in run_folders:
            run = run_folder[-4:]
            nii_files = glob.glob(os.path.join(run_folder, f'{files_type}*.nii'))
            run_files[run] = nii_files
        participant_files[participant] = run_files
    return participant_files


def load_dataframe(participant_path):
    file_list = [f for f in os.listdir(participant_path) if f.startswith('Resultfile_p')]
    dfs = {}
    for file_name in file_list:
        full_path = os.path.join(participant_path, file_name)
        df = pd.read_csv(full_path, sep='\t')
        key = file_name[-8:-4].upper()
        dfs[key] = df
    return dfs


def find_tissue_masks(participant_folder):
    """Return paths to WM (wc2cs) and CSF (wc3cs) masks in the structural folder."""
    struct_dirs = glob.glob(os.path.join(participant_folder, 'STRUCTURAL_*'))
    if not struct_dirs:
        return None, None
    struct_dir = struct_dirs[0]
    wm_files  = glob.glob(os.path.join(struct_dir, 'wc2cs*.nii'))
    csf_files = glob.glob(os.path.join(struct_dir, 'wc3cs*.nii'))
    if not wm_files or not csf_files:
        return None, None
    return wm_files[0], csf_files[0]


def extract_tissue_signals(fmri_img, wm_path, csf_path):
    """Extract mean WM and CSF signal per TR (resampled to functional space)."""
    wm_mask  = nilearn_image.math_img("img > 0.9", img=wm_path)
    csf_mask = nilearn_image.math_img("img > 0.9", img=csf_path)
    wm_mask_res  = nilearn_image.resample_to_img(wm_mask,  fmri_img, interpolation='nearest')
    csf_mask_res = nilearn_image.resample_to_img(csf_mask, fmri_img, interpolation='nearest')
    wm_signal  = NiftiMasker(mask_img=wm_mask_res,  standardize=False).fit_transform(fmri_img).mean(axis=1, keepdims=True)
    csf_signal = NiftiMasker(mask_img=csf_mask_res, standardize=False).fit_transform(fmri_img).mean(axis=1, keepdims=True)
    return wm_signal, csf_signal


def z_score_run(fmri):
    """Z-score each voxel's time series across the run."""
    mean_time = np.mean(fmri, axis=3, keepdims=True)
    std_time = np.std(fmri, axis=3, keepdims=True)
    std_time = np.where(std_time == 0, 1, std_time)
    return (fmri - mean_time) / std_time


participant_files = select_files(folder_fmri, files_type)

for participant, runs in participant_files.items():
    dfs = load_dataframe(os.path.join(folder_fmri, participant))
    subj_dir = os.path.join(output_dir_fmri, participant)
    os.makedirs(subj_dir, exist_ok=True)

    for run_number, run_files in runs.items():
        concatenated_img = concat_imgs(run_files)
        fmri = concatenated_img.get_fdata()
        affine = concatenated_img.affine
        header = concatenated_img.header

        # Z-score each voxel across time
        fmri_normalized = z_score_run(fmri)

        # Build confounds: motion params + WM/CSF signals
        rp_files = glob.glob(os.path.join(folder_fmri, participant, run_number, 'rp_*.txt'))
        if rp_files:
            mp = np.loadtxt(rp_files[0])          # (n_TRs, 6)
            mp_deriv = np.vstack([mp[:1], np.diff(mp, axis=0)])  # derivatives
            confounds = np.hstack([mp, mp**2, mp_deriv, mp_deriv**2])  # (n_TRs, 24)

            wm_path, csf_path = find_tissue_masks(os.path.join(folder_fmri, participant))
            if wm_path and csf_path:
                wm_signal, csf_signal = extract_tissue_signals(concatenated_img, wm_path, csf_path)
                confounds = np.hstack([confounds, wm_signal, csf_signal])  # (n_TRs, 8)
                print(f"Motion + WM/CSF regression applied for {participant} {run_number}")
            else:
                print(f"Motion regression applied for {participant} {run_number} (no WM/CSF masks found)")

            x, y, z, t = fmri_normalized.shape
            fmri_2d = fmri_normalized.reshape(-1, t).T  # (timepoints, voxels)
            fmri_cleaned = nilearn_signal.clean(fmri_2d, confounds=confounds, standardize=None,
                                               high_pass=0.01, t_r=TR)
            fmri_normalized = fmri_cleaned.T.reshape(x, y, z, t)
        else:
            print(f"Warning: No motion parameter file found for {participant} {run_number}")

        df = dfs[run_number]
        df = df.rename(columns=lambda x: x.strip())
        frame_times = np.arange(0, fmri_normalized.shape[-1] * TR, TR)

        for _, row in df.iterrows():
            context = row['Context']
            statement = row['Statement']
            task = row['task']
            start_statement = row['Real_Time_Onset_Statement']
            end_statement = row['Real_Time_End_Statement']
            duration_statement = end_statement - start_statement
            end_evaluation = row['Real_Time_End_Evaluation']

            if np.isnan(end_evaluation):
                end_evaluation = row['Real_Time_Onset_Evaluation'] + 5

            # HRF regressor aligned to statement onset
            exp_condition = [np.array([start_statement]),
                             np.array([duration_statement]),
                             np.array([1.0])]
            hrf_regressor, _ = compute_regressor(
                exp_condition=exp_condition,
                hrf_model='glover',
                frame_times=frame_times,
                oversampling=16
            )

            # Extract window from statement onset to end of evaluation
            start_scan = max(0, min(round(start_statement / TR), fmri_normalized.shape[-1] - 1))
            end_scan = max(start_scan, min(round(end_evaluation / TR), fmri_normalized.shape[-1]))

            scans = fmri_normalized[..., start_scan:end_scan]
            hrf_weights = hrf_regressor[start_scan:end_scan, 0]
            hrf_weights = np.clip(hrf_weights, 0, None)

            if scans.shape[-1] == 0 or hrf_weights.sum() == 0:
                print(f"Skipping {participant}_{task}_{context[:-4]}_{statement[:-4]}: empty window or zero weights")
                continue

            hrf_weights = hrf_weights / hrf_weights.sum()
            weighted_scans = np.average(scans, axis=-1, weights=hrf_weights)

            filename = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement'
            nib.save(nib.Nifti1Image(weighted_scans, affine, header),
                     os.path.join(subj_dir, filename + ".nii.gz"))

print("Processing complete.")
