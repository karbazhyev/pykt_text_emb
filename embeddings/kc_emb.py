def get_avg_kc_emb(embeddings, kc_map):
    # Example implementation
    avg_kc_emb = {}
    for kc, problem_ids in kc_map.items():
        kc_emb = torch.mean(embeddings[problem_ids], dim=0)
        avg_kc_emb[kc] = kc_emb
    return avg_kc_emb


def get_sample_kc_emb(embeddings, kc_map):
    # Example implementation
    pass

def get_stuffing_kc_emb(question_text_df, kc_map):
    # Example implementation
    pass

def get_kc_name_emb(kc_names, kc_map):
    # Example implementation
    pass