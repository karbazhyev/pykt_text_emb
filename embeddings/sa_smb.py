import torch

def sa_mirror_emb(emb_tensor):
	# Example implementation
	emn_tensor_flipped = emb_tensor.flip(dims=[-1])
	stacked_tensor = torch.stack((emb_tensor, emn_tensor_flipped), dim=0)
	return stacked_tensor
def sa_stacked(emb_tensor, correct_emb, incorrect_emb):
    # Ensure correct_emb has the same batch size as emb_tensor
    correct_emb_expanded = correct_emb.unsqueeze(0).expand(emb_tensor.size(0), -1)
    # Concatenate along the last dimension
    stacked_tensor = torch.cat([emb_tensor, correct_emb_expanded], dim=1)
    return stacked_tensor



def get_mirror_embeddings(emb_tensor):
    import json
    import torch
    for data in emb_tensor:
        emb = torch.tensor([x * -1 for x in emb_tensor])
        # emb = torch.reshape(emb, (1, emb_dim))
        emb_tensor = torch.cat((emb_tensor, emb), dim=0)
    return emb_tensor

def get_mirror_embeddings(emb_tensor):
    import json
    import torch
    for data in emb_tensor:
        emb = torch.tensor([x * -1 for x in emb_tensor])
        # emb = torch.reshape(emb, (1, emb_dim))
        emb_tensor = torch.cat((emb_tensor, emb), dim=0)
    return emb_tensor

def get_add_word_embeddings(emb_tensor, word_emb):
    pass