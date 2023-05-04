from data_tokenizer import *
from vocabulary import *
from loss_function import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy

train_df, val_df = train_validation()
fields, Input, Output = voc_py()
criterion = maskNLLLoss
print("success")

TRG_PAD_IDX, model = training()

def make_trg_mask(trg):
                
        trg_pad_mask = (trg != TRG_PAD_IDX).unsqueeze(1).unsqueeze(2)
                
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
                    
        trg_mask = trg_pad_mask & trg_sub_mask
                
        return trg_mask

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    n_totals = 0
    print_losses = []
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # print(batch)
        loss = 0
        src = batch.Input.permute(1, 0)
        trg = batch.Output.permute(1, 0)
        trg_mask = make_trg_mask(trg)
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
            
        mask_loss, nTotal = criterion(output, trg, trg_mask)
        
        mask_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal
       
    return sum(print_losses) / n_totals


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    n_totals = 0
    print_losses = []
    
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):

            src = batch.Input.permute(1, 0)
            trg = batch.Output.permute(1, 0)
            trg_mask = make_trg_mask(trg)

            output, _ = model(src, trg[:,:-1])

            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            mask_loss, nTotal = criterion(output, trg, trg_mask)

            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
        
    return sum(print_losses) / n_totals


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

SRC = Input
TRG = Output

model.load_state_dict(torch.load('D:\College\Sem-8\codebot1\model.pt'),strict=False)

print(model)

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

# def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
#     assert n_rows * n_cols == n_heads
    
#     fig = plt.figure(figsize=(30,50))
    
#     for i in range(n_heads):
        
#         ax = fig.add_subplot(n_rows, n_cols, i+1)
        
#         _attention = attention.squeeze(0)[i].cpu().detach().numpy()

#         cax = ax.matshow(_attention, cmap='bone')

#         ax.tick_params(labelsize=12)
#         ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
#                            rotation=45)
#         ax.set_yticklabels(['']+translation)

#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.show()

# # Converting english sentence to python code
# def eng_to_python(Source):
#   Source=Source.split(" ")
#   translation, attention = translate_sentence(Source, SRC, TRG, model, device)

#   print(f'Python Code: \n')
#   print()
#   py_code = untokenize(translation[:-1]).decode('utf-8')
#   return py_code