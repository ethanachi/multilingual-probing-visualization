# Randomizes the encoder layers of a BERT model.
# Ethan Chi (ethanchi@stanford.edu), May 2020

from pytorch_pretrained_bert import BertModel
import sys
import torch

weights_location = "/sailhome/ethanchi/.pytorch_pretrained_bert/731c19ddf94e294e00ec1ba9a930c69cc2a0fd489b25d3d691373fae4c0986bd.4e367b0d0155d801930846bb6ed98f8a7c23e0ded37888b29caa37009a40c7b9"
if len(sys.argv) > 2:
    weights_location = sys.argv[2]

model = BertModel.from_pretrained(weights_location)
state_dict = model.state_dict()
state_keys = state_dict.keys()

for key in state_dict:
    if 'encoder' in key:
        print(key, state_dict[key].shape, state_dict[key].mean(), state_dict[key].var())
        state_dict[key].normal_(state_dict[key].mean(), state_dict[key].var())

print("All replaced.")
torch.save(state_dict, sys.argv[1])
