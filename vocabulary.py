from data_tokenizer import *

train_df, val_df = train_validation()

# print(train_df.shape)
# print(val_df.shape)

def voc_py():

    SEED = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    Input = data.Field(tokenize = 'spacy',
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=True)

    Output = data.Field(tokenize = augment_tokenize_pycode,
                        init_token='<sos>', 
                        eos_token='<eos>', 
                        lower=False)

    fields = [('Input', Input),('Output', Output)]

    train_example = []
    val_example = []

    train_expansion_factor = 100
    for j in range(train_expansion_factor):
        for i in range(train_df.shape[0]):
            try:
                ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
                train_example.append(ex)
            except:
                pass

    for i in range(val_df.shape[0]):
        try:
            ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
            val_example.append(ex)
        except:
            pass       

    train_data = data.Dataset(train_example, fields)
    valid_data =  data.Dataset(val_example, fields)

    return train_data, valid_data, Input, Output