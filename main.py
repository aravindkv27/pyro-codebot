from data_tokenizer import *
from vocabulary import *
from encoder import *

    # Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def code_con():


    token_sample = tokenize_python(dps[5]['solution'])
    # print(token_sample)

    # print(untokenize(token_sample).decode('utf-8'))


    tokenized_sample = augment_tokenize_pycode(dps[1]['solution'])
    # print(tokenized_sample)

    print(untokenize(tokenized_sample).decode('utf-8'))

    token_sample = tokenize_python(dps[5]['solution'])
    # print(token_sample)

    # train_df, val_df = train_validation()

    # print(train_df.shape)
    # print(val_df.shape)

    train_data, valid_data, Input, Output = voc_py()

    Input.build_vocab(train_data, min_freq = 0)
    Output.build_vocab(train_data, min_freq = 0)

    print(Output.vocab)
    print(train_data[0].Output)

code_con()