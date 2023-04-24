import torch
import spacy
# from vocabulary import *
from vocabulary import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input, Output, model = final_values()
fields, Input, Output = voc_py()

tg, model = training()

SRC = Input
TRG = Output


# model.load_state_dict(torch.load('D:\College\Sem-8\codebot1\model123.pt'), strict=False)
# print(model)

# def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50000):
    
#     model.eval()
        
#     if isinstance(sentence, str):
#         nlp = spacy.load('en')
#         tokens = [token.text.lower() for token in nlp(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]

#     tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
#     src_indexes = [src_field.vocab.stoi[token] for token in tokens]

#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
#     src_mask = model.make_src_mask(src_tensor)
    
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)

#     trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

#     for i in range(max_len):

#         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

#         trg_mask = model.make_trg_mask(trg_tensor)
        
#         with torch.no_grad():
#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
#         pred_token = output.argmax(2)[:,-1].item()
        
#         trg_indexes.append(pred_token)

#         if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
#             break
    
#     trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
#     return trg_tokens[1:], attention