
def get_emb_from_model(text_df, model, emb_dim):

    if model == "openai":
        return get_openai_embeddings(text_df, model, emb_dim)
    elif model == "cohere":
        return get_cohere_embeddings(text_df)
    elif model == "bert-ots":
        return get_bert_ots_embeddings(text_df)
    elif model == "bert-ft":
        return get_bert_ft_embeddings(text_df)
    else:
        raise ValueError(f"Unknown model: {model}")

def get_openai_embeddings(text_df, model, emb_dim, api_ke_config_path):
    from openai import OpenAI
    import json
    import torch
    with open(api_ke_config_path, 'r') as f:
        api_key = json.load(f)['open_ai_api_key']

    client = OpenAI(api_key=api_key)
    print(model, emb_dim)
    response_embeddings = client.embeddings.create(input = text_df, model=model).data
    embeddings = torch.empty((0, emb_dim))

    for data in response_embeddings:
        emb = torch.tensor(data.embedding)
        emb = torch.reshape(emb, (1, emb_dim))
        embeddings = torch.cat((embeddings, emb), dim=0)

    return embeddings


def get_cohere_embeddings(text_df):
    # Implementation for Cohere embeddings
    pass

def get_bert_ots_embeddings(text_df):
    # Implementation for BERT-OTS embeddings
    pass

def get_bert_ft_embeddings(text_df):
    # Implementation for BERT-FT embeddings
    pass


