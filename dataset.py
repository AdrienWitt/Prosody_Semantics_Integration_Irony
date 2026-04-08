import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
import analysis_helpers


class WholeBrainDataset(Dataset):
    # Valid embedding types and whether they require context in the filename
    TEXT_EMBEDDING_TYPES = {
        "cross_attention":  "context_statement",
        "joint_encoding":   "context_statement",
        "statement_only":   "statement",
    }

    def __init__(self, participant_list, data_path, fmri_data_path, mask,
                 included_tasks=None, use_base_features=True,
                 use_text=True, use_audio=True,
                 embeddings_text_path=None, embeddings_audio_path=None,
                 text_embedding_type="cross_attention",
                 pca_threshold=0.50, use_pca=False, **kwargs):
        super().__init__()
        if text_embedding_type not in self.TEXT_EMBEDDING_TYPES:
            raise ValueError(f"text_embedding_type must be one of {list(self.TEXT_EMBEDDING_TYPES)}")
        self.data_path = data_path
        self.fmri_data_path = fmri_data_path
        self.participant_list = participant_list
        self.included_tasks = included_tasks or ["sarcasm", "irony", "prosody", "semantic", "tom"]
        self.use_base_features = use_base_features
        self.use_text = use_text
        self.use_audio = use_audio
        self.embeddings_text_path = embeddings_text_path
        self.embeddings_audio_path = embeddings_audio_path
        self.text_embedding_type = text_embedding_type
        self.pca_threshold = pca_threshold
        self.use_pca = use_pca

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.mask = mask
        self.fmri_cache = self.preload_fmri()
        self.data, self.fmri_data, self.ids_list, self.col_groups = self.create_data()

    def preload_fmri(self):
        fmri_cache = {}
        for participant in self.participant_list:
            participant_fmri_path = os.path.join(self.fmri_data_path, participant)
            if not os.path.exists(participant_fmri_path):
                continue
            for fmri_file in os.listdir(participant_fmri_path):
                if fmri_file.endswith('.npy'):
                    key = f"{participant}/{fmri_file}"
                    fmri_cache[key] = np.load(os.path.join(participant_fmri_path, fmri_file), mmap_mode='r')
        return fmri_cache

    def process_participant(self, participant):
        participant_data_path = os.path.join(self.data_path, participant)
        dfs = analysis_helpers.load_dataframe(participant_data_path)

        final_data = []
        fmri_data_list = []
        ids_list = []
        embeddings_text_list = []
        embeddings_audio_list = []
        task_counts = {task: 0 for task in self.included_tasks}
        sample_count = 0

        voxel_indices = np.where(self.mask.get_fdata().reshape(-1) > 0)[0]

        for df in dfs.values():
            df = df.rename(columns=lambda x: x.strip())
            for index, row in df.iterrows():
                task = row["task"]
                if task not in self.included_tasks:
                    continue

                task_counts[task] += 1
                sample_count += 1

                context = row["Context"]
                statement = row["Statement"]

                evaluation = row["Evaluation_Score"]
                age = row["age"]
                gender = row["genre"]
                fmri_file = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement.npy'
                fmri_path = f"{participant}/{fmri_file}"

                parts = row["Condition_name"].split("_")
                context_cond = parts[0]
                statement_cond = parts[1]

                fmri = self.fmri_cache.get(fmri_path)
                if fmri is None:
                    continue

                fmri_masked = fmri[:, voxel_indices]
                fmri_data_list.append(fmri_masked)

                if self.use_base_features:
                    final_data.append({
                        "context": context_cond,
                        "semantic": statement_cond[:2],
                        "prosody": statement_cond[-3:],
                        "task": task,
                        "evaluation": evaluation,
                        "age": age,
                        "gender": gender,
                        "participant": participant,
                    })

                ids_list.append(int(participant[1:]))

                if self.use_text and self.embeddings_text_path:
                    context_stem   = os.path.splitext(context)[0]
                    statement_stem = os.path.splitext(statement)[0]
                    if self.TEXT_EMBEDDING_TYPES[self.text_embedding_type] == "context_statement":
                        text_file = f"{context_stem}_{statement_stem}.npy"
                    else:  # statement_only
                        text_file = f"{statement_stem}.npy"
                    embeddings_text = np.load(os.path.join(self.embeddings_text_path, text_file))
                    embeddings_text_list.append(embeddings_text)

                if self.use_audio and self.embeddings_audio_path:
                    audio_file = statement.replace('.wav', '_opensmile.npy')
                    embeddings_audio = np.load(os.path.join(self.embeddings_audio_path, audio_file))
                    embeddings_audio_list.append(embeddings_audio)

        return (final_data, fmri_data_list, ids_list,
                embeddings_text_list, embeddings_audio_list,
                task_counts, sample_count)

    def create_data(self):
        final_data = []
        fmri_data_list = []
        ids_list = []
        embeddings_text_list = []
        embeddings_audio_list = []
        task_counts = {task: 0 for task in self.included_tasks}
        total_samples = 0

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_participant, self.participant_list))

        for (part_final, part_fmri, part_ids,
             part_text, part_audio,
             part_task_counts, part_count) in results:

            final_data.extend(part_final)
            fmri_data_list.extend(part_fmri)
            ids_list.extend(part_ids)
            embeddings_text_list.extend(part_text)
            embeddings_audio_list.extend(part_audio)
            for task, count in part_task_counts.items():
                task_counts[task] += count
            total_samples += part_count

        print(f"Loaded {total_samples} total samples")
        for task, count in task_counts.items():
            print(f" - {task}: {count} samples")

        col_groups = {'base_onehot': [], 'age': [], 'text': [], 'audio': []}

        # Create base DataFrame with raw (unscaled) features
        if self.use_base_features and final_data:
            df = pd.DataFrame(final_data)
            df.reset_index(drop=True, inplace=True)
            semantic_condition_cols  = ['context', 'semantic'] if self.use_text  else []
            prosodic_condition_cols = ['prosody']              if self.use_audio else []
            categorical_cols = ['task', 'gender', 'participant'] + semantic_condition_cols + prosodic_condition_cols
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            df['evaluation'] = df['evaluation'].fillna(df['evaluation'].median())
            df['evaluation'] = (df['evaluation'] - df['evaluation'].min()) / (df['evaluation'].max() - df['evaluation'].min())
            # age stays raw — will be scaled per fold by FoldPreprocessor
            col_groups['age'] = ['age']
            col_groups['base_onehot'] = [c for c in df.columns if c != 'age']
        else:
            df = pd.DataFrame(index=range(total_samples))

        # Text embeddings — concatenate raw, no scaling
        if self.use_text and embeddings_text_list:
            emb_df = pd.DataFrame(
                np.vstack(embeddings_text_list).squeeze(),
                columns=[f"emb_text_{i}" for i in range(np.vstack(embeddings_text_list).shape[-1])]
            )
            emb_df.reset_index(drop=True, inplace=True)
            df = pd.concat([df, emb_df], axis=1)
            col_groups['text'] = list(emb_df.columns)

        # Audio embeddings — concatenate raw, no scaling
        if self.use_audio and embeddings_audio_list:
            emb_df = pd.DataFrame(
                np.vstack(embeddings_audio_list).squeeze(),
                columns=[f"emb_audio_{i}" for i in range(np.vstack(embeddings_audio_list).shape[-1])]
            )
            emb_df.reset_index(drop=True, inplace=True)
            df = pd.concat([df, emb_df], axis=1)
            col_groups['audio'] = list(emb_df.columns)

        if not fmri_data_list:
            raise ValueError("No fMRI data was loaded. Check paths and file naming.")
        fmri_data = np.vstack(fmri_data_list)
        ids_array = np.array(ids_list, dtype=np.int32)

        return df, fmri_data, ids_array, col_groups

    def __getitem__(self, index):
        features = self.data.iloc[index].values.astype(np.float32)
        fmri = self.fmri_data[index].astype(np.float32)
        return {"features": features, "fmri_data": fmri}

    def __len__(self):
        return len(self.data)


class FoldPreprocessor:
    """
    Fits scalers and optionally PCA on training data only.
    Instantiate a fresh instance for every fold — never reuse across folds.
    """
    def __init__(self, col_groups, use_pca=False, pca_threshold=0.95):
        self.col_groups    = col_groups
        self.use_pca       = use_pca
        self.pca_threshold = pca_threshold
        self._scalers = {}
        self._pcas    = {}

    def fit_transform(self, df):
        return self._process(df, fit=True)

    def transform(self, df):
        return self._process(df, fit=False)

    def _process(self, df, fit):
        parts = []

        # One-hot base features — no scaling
        if self.col_groups['base_onehot']:
            parts.append(df[self.col_groups['base_onehot']].values.astype(np.float32))

        # Age — StandardScaler
        age_cols = self.col_groups['age']
        if age_cols and all(c in df.columns for c in age_cols):
            if fit:
                self._scalers['age'] = StandardScaler().fit(df[age_cols])
            parts.append(self._scalers['age'].transform(df[age_cols]).astype(np.float32))

        # Text and audio embeddings
        for group in ('text', 'audio'):
            if self.col_groups.get(group):
                parts.append(self._scale_or_pca(df, group, fit))

        return np.hstack(parts).astype(np.float32)

    def _scale_or_pca(self, df, group, fit):
        cols = self.col_groups[group]
        X = df[cols].values
        if fit:
            self._scalers[group] = StandardScaler().fit(X)
        X_sc = self._scalers[group].transform(X)
        if self.use_pca:
            if fit:
                n = int(self.pca_threshold) if self.pca_threshold >= 1 else self.pca_threshold
                self._pcas[group] = PCA(n_components=n).fit(X_sc)
                actual_var = np.sum(self._pcas[group].explained_variance_ratio_)
                n_comp = self._pcas[group].n_components_
                print(f"[FoldPreprocessor] {group}: {n_comp} components → {actual_var*100:.1f}% variance")
            return self._pcas[group].transform(X_sc).astype(np.float32)
        return X_sc.astype(np.float32)
