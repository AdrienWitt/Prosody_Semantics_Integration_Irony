import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torchaudio
import torch
import opensmile

MODEL_NAME = "almanach/camembertav2-base"


def _load_texts(path):
    texts = {}
    for fname in os.listdir(path):
        if fname.endswith('.txt'):
            scenario = fname.split('_')[-1].split('.')[0]
            with open(os.path.join(path, fname), "r") as f:
                texts.setdefault(scenario, []).append((fname, f.read().strip()))
    return texts


def _load_model():
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def embeddings_cross_attention(contexts_path, statements_path, output_dir, model, tokenizer):
    """
    Approach 1: separate encoding + manual cross-attention.
    Statement tokens are weighted by their dot-product similarity to the context embedding.
    Output: [emb_statement_weighted (768) | diff (768)] = 1536-dim
    """
    os.makedirs(output_dir, exist_ok=True)
    contexts = _load_texts(contexts_path)
    statements = _load_texts(statements_path)
    pairs_count = sum(len(contexts[s]) * len(statements[s]) for s in contexts if s in statements)
    print(f"[cross-attention] {pairs_count} pairs to generate")

    pair_count = 0
    for scenario, scenario_contexts in contexts.items():
        if scenario not in statements:
            continue
        for context_file, context_text in scenario_contexts:
            inputs_A = tokenizer(context_text, return_tensors="pt",
                                 truncation=True, max_length=512)
            with torch.no_grad():
                emb_A = model(**inputs_A).last_hidden_state.mean(dim=1)  # (1, 768)

            for statement_file, statement_text in statements[scenario]:
                inputs_B = tokenizer(statement_text, return_tensors="pt",
                                     truncation=True, max_length=512)
                with torch.no_grad():
                    emb_B_tokens = model(**inputs_B).last_hidden_state
                    attn_scores = torch.softmax(
                        torch.matmul(emb_B_tokens, emb_A.transpose(-1, -2)), dim=1
                    )
                    emb_B_weighted = (emb_B_tokens * attn_scores).sum(dim=1)
                    diff = emb_B_weighted - emb_A
                    embedding = torch.cat([emb_B_weighted, diff], dim=1).numpy()

                fname = f"{os.path.splitext(context_file)[0]}_{os.path.splitext(statement_file)[0]}.npy"
                np.save(os.path.join(output_dir, fname), embedding)
                pair_count += 1
                if pair_count % 10 == 0:
                    print(f"  {pair_count}/{pairs_count}")

    print(f"[cross-attention] Done — {pair_count} embeddings saved to {output_dir}")


def embeddings_joint_encoding(contexts_path, statements_path, output_dir, model, tokenizer):
    """
    Approach 2: joint encoding — context + statement passed together to CamemBERT.
    CamemBERT's 12-layer attention operates across both sequences.
    Only statement token representations are extracted and pooled.
    emb_A (for diff) is extracted from the same forward pass — context tokens before first [SEP].
    Output: [emb_statement_in_context (768) | diff (768)] = 1536-dim
    """
    os.makedirs(output_dir, exist_ok=True)
    contexts = _load_texts(contexts_path)
    statements = _load_texts(statements_path)
    pairs_count = sum(len(contexts[s]) * len(statements[s]) for s in contexts if s in statements)
    sep_id = tokenizer.sep_token_id
    print(f"[joint-encoding] {pairs_count} pairs to generate")

    pair_count = 0
    for scenario, scenario_contexts in contexts.items():
        if scenario not in statements:
            continue
        for context_file, context_text in scenario_contexts:
            for statement_file, statement_text in statements[scenario]:
                inputs_joint = tokenizer(context_text, statement_text,
                                         return_tensors="pt", truncation=True,
                                         max_length=512)
                with torch.no_grad():
                    hidden = model(**inputs_joint).last_hidden_state

                token_ids = inputs_joint['input_ids'][0].tolist()
                if sep_id not in token_ids:
                    print(f"Warning: no SEP token for {statement_file}, skipping")
                    continue
                sep_pos = token_ids.index(sep_id)

                # Extract context and statement from the same forward pass
                emb_A = hidden[:, 1:sep_pos, :].mean(dim=1)          # context tokens (excl. CLS)
                statement_hidden = hidden[:, sep_pos + 1:-1, :]       # statement tokens (excl. final SEP)

                if statement_hidden.shape[1] == 0:
                    print(f"Warning: empty statement tokens for {statement_file}, skipping")
                    continue

                emb_statement = statement_hidden.mean(dim=1)
                diff = emb_statement - emb_A
                embedding = torch.cat([emb_statement, diff], dim=1).numpy()

                fname = f"{os.path.splitext(context_file)[0]}_{os.path.splitext(statement_file)[0]}.npy"
                np.save(os.path.join(output_dir, fname), embedding)
                pair_count += 1
                if pair_count % 10 == 0:
                    print(f"  {pair_count}/{pairs_count}")

    print(f"[joint-encoding] Done — {pair_count} embeddings saved to {output_dir}")


def embeddings_statement_only(statements_path, output_dir, model, tokenizer):
    """
    Baseline: encode statement alone, no context.
    Output: [CLS-mean pooled statement (768)] = 768-dim
    """
    os.makedirs(output_dir, exist_ok=True)
    statements = _load_texts(statements_path)
    all_statements = [(f, t) for stmts in statements.values() for f, t in stmts]
    print(f"[statement-only] {len(all_statements)} statements to generate")

    for i, (statement_file, statement_text) in enumerate(all_statements):
        inputs = tokenizer(statement_text, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()

        fname = os.path.splitext(statement_file)[0] + ".npy"
        np.save(os.path.join(output_dir, fname), embedding)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(all_statements)}")

    print(f"[statement-only] Done — {len(all_statements)} embeddings saved to {output_dir}")


def create_audio_embeddings(audio_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    for filename in os.listdir(audio_path):
        if not filename.endswith('.wav'):
            continue
        y, sr = torchaudio.load(os.path.join(audio_path, filename))
        features = smile.process_signal(signal=y.squeeze().numpy(), sampling_rate=sr)
        np.save(os.path.join(output_dir, filename[:-4] + '_opensmile.npy'), features)
    print(f"Audio embeddings saved to {output_dir}")


# ── Paths ─────────────────────────────────────────────────────────────────────

contexts_path   = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text\contexts"
statements_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text\statements"

model, tokenizer = _load_model()  # load once, shared across both functions

embeddings_cross_attention(
    contexts_path, statements_path, model=model, tokenizer=tokenizer,
    output_dir=r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\Clean_Project\embeddings\text_cross_attention"
)

embeddings_joint_encoding(
    contexts_path, statements_path, model=model, tokenizer=tokenizer,
    output_dir=r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\Clean_Project\embeddings\text_joint_encoding"
)

embeddings_statement_only(
    statements_path, model=model, tokenizer=tokenizer,
    output_dir=r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\Clean_Project\embeddings\text_statement_only"
)

create_audio_embeddings(
    audio_path=r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\audio",
    output_dir=r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\Clean_Project\embeddings\audio_opensmile"
)
