from data_tokenizer import *
from vocabulary import *
from encoder import *
from decoder import *



global fields, train_data, valid_data
# Check if a GPU is available
def run_on_gpu():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

def code_con():

    token_sample = tokenize_python(dps[1]['solution'])

    tokenized_sample = augment_tokenize_pycode(dps[1]['solution'])
    # print(tokenized_sample)
    print(token_sample)
    print(untokenize(tokenized_sample).decode('utf-8'))

    # token_sample = tokenize_python(dps[1]['solution'])

    fields, train_data, valid_data, Input, Output = voc_py()

    Input.build_vocab(train_data, min_freq = 0)
    Output.build_vocab(train_data, min_freq = 0)

    print(Output.vocab)
    print(train_data[1].Output)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)




if __name__ == "__main__":
    run_on_gpu()
    code_con()
    TRG_PAD_IDX, model = training()
    print(f'The model has {count_parameters(model):,} trainable parameters')
    # demo()
    
    
    


