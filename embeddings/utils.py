
def get_emb_from_model(text_df, model):

    if model == "openai":
        return get_openai_embeddings(text_df)
    elif model == "cohere":
        return get_cohere_embeddings(text_df)
    elif model == "bert-ots":
        return get_bert_ots_embeddings(text_df)
    elif model == "bert-ft":
        return get_bert_ft_embeddings(text_df)
    else:
        raise ValueError(f"Unknown model: {model}")

def get_openai_embeddings(text_df):
    # Implementation for OpenAI embeddings
    pass

def get_cohere_embeddings(text_df):
    # Implementation for Cohere embeddings
    pass

def get_bert_ots_embeddings(text_df):
    # Implementation for BERT-OTS embeddings
    pass

def get_bert_ft_embeddings(text_df):
    # Implementation for BERT-FT embeddings
    pass


