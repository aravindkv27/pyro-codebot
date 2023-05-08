from data_tokenizer import *
from encoder import *

# from training_model import training_model

train_df, val_df = train_validation()


# Firstly, the seed value of 1234 is set for the random number generator using random.seed(SEED) 
# and PyTorch's manual seed is also set to this value using torch.manual_seed(SEED). 
# The torch.cuda.manual_seed(SEED) sets the seed value for CUDA devices if available.

def voc_py():

    SEED = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    global Input, Output

    #  the Input field is defined to preprocess the input data.
    Input = data.Field(tokenize = 'spacy',
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=True)

    # The Output field is defined to preprocess the output data
    Output = data.Field(tokenize = augment_tokenize_pycode,
                        init_token='<sos>', 
                        eos_token='<eos>', 
                        lower=False)

    fields = [('Input', Input),('Output', Output)]

    train_example = []
    val_example = []

    # Data preparation step for ML model. The input and Output data are loaded
    # prepared for training and validation.
    tef = 100
    for j in range(tef):
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
    Input.build_vocab(train_data, min_freq = 0)
    Output.build_vocab(train_data, min_freq = 0)

    return fields, Input, Output

def training():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = len(Input.vocab)
    OUTPUT_DIM = len(Output.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 16
    DEC_HEADS = 16
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                device)
    
    # print(len(Output.vocab.__dict__['freqs']))

    SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
    global TRG_PAD_IDX
    TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]
    global model
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    return TRG_PAD_IDX, model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):

    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def test_model():

    model.apply(initialize_weights)
    LEARNING_RATE = 0.0005

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    return optimizer

def final_values():
   
   input1 = Input
   output1 = Output
   model1 = model

   return input1, output1, model1