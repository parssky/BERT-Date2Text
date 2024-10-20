

# Model Card for BERT-Text2Date

## Model Overview

**Model Name:** BERT-Text2Date
**Model Type:** BERT (Encoder-only architecture)  
**Language:** Persian

**Description:**  
This model is designed to process and generate Persian dates in both formal (YYYY-MM-DD) and informal formats. It utilizes a dataset that includes various representations of dates, allowing for effective training in understanding and predicting Persian date formats.

## Dataset

**Dataset Description:**  
The dataset consists of two types of dates: formal and informal. It is generated using two main functions:

- **`convert_year_to_persian(year)`**: Converts years to Persian format, currently supporting the year 1400.
- **`generate_date_mappings_with_persian_year(start_year, end_year)`**: Generates dates for a specified range, considering the number of days in each month.

**Data Formats:**

- **Informal Dates:** Various formats like “روز X ماه سال” and “اول/دوم/… ماه سال”.
- **Formal Dates:** Stored in YYYY-MM-DD format.

**Example Dates:**

- بیست و هشتم اسفند هزار و چهار صد و ده, 1410-12-28
- 1 فروردین 1400, 1400-01-01

**Data Split:**

- **Training Set:** 80% (19272 samples)
- **Validation Set:** 10% (2409 samples)
- **Test Set:** 10% (2409 samples)

## Model Architecture

**Architecture Details:**  
The model is built using an encoder-only architecture, consisting of:

- **Layers:** 4 Encoder layers
- **Parameters:**
    - `vocab_size`: 25003
    - `context_length`: 32
    - `emb_dim`: 256
    - `n_heads`: 4
    - `drop_rate`: 0.1

**Parameter Count:** 14,933,931

```
Transformer( (embedding): Embedding(25003, 256) (positional_encoding): Embedding(32, 256) (en): TransformerEncoder( (layers): ModuleList( (0-3): 4 x TransformerEncoderLayer( (self_attn): MultiheadAttention( (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=False) ) (linear1): Linear(in_features=256, out_features=512, bias=False) (dropout): Dropout(p=0.1, inplace=False) (linear2): Linear(in_features=512, out_features=256, bias=False) (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True) (dropout1): Dropout(p=0.1, inplace=False) (dropout2): Dropout(p=0.1, inplace=False) ) ) ) (fc_train): Linear(in_features=256, out_features=25003, bias=True) )
```

**Tokenizer:**  
The model uses a Persian tokenizer named “بلبل زبان” available on Hugging Face, with a vocabulary size of 25,000 tokens.

## Training

**Training Process:**

- **Batch Size:** 2048
- **Epochs:** 60
- **Learning Rate:** 0.00005
- **Optimizer:** AdamW
- **Weight Decay:** 0.2
- **Masking Technique:** The formal part of the date is masked to facilitate learning.

**Performance Metrics:**

- **Training Loss:** Reduced from 10.3 to 0.005 over 60 epochs.
- **Validation Loss:** Reduced from 10.1 to 0.010.
- **Test Accuracy:** 66% (exact match required).
- **Perplexity:** 1.01

## Inference

**Inference Code:**  
The model can be loaded along with the tokenizer using the provided `Inference.ipynb` file. Three functions are implemented:

1. **Convert Token IDs to Text**
```python
def text_to_token_ids(text, tokenizer):

	encoded = tokenizer.encode(text)
	
	encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension

return encoded_tensor
```

2. **Convert Text to Token IDs**
```python
def token_ids_to_text(token_ids, tokenizer):

	flat = token_ids.squeeze(0) # remove batch dimension

return tokenizer.decode(flat.tolist())
```

3. **`predict_masked(input)`**: Takes an input to predict the masked date.
```python
def predict_masked(model,tokenizer,input,deivce):

	model.eval()
	
	inputs_masked = input + " " + "[MASK][MASK][MASK][MASK]-[MASK][MASK]-[MASK][MASK]"
	
	input_ids = tokenizer.encode(inputs_masked)
	
	input_ids = torch.tensor(input_ids).to(deivce)
	
	with torch.no_grad():
	
	logits = model(input_ids.unsqueeze(0))
	
	logits = logits.flatten(0, 1)
	
	probs = torch.argmax(logits,dim=-1,keepdim=True)
	
	token_ids = probs.squeeze(1)
	
	answer_ids = token_ids[-11:-1]

return token_ids_to_text(answer_ids,tokenizer)
```

And use:
```python
predict_masked(model,tokenizer,"12 آبان 1402","cuda")
```
Output: 
```
'1402-08-12'
```
## Limitations

- The model currently only supports Persian dates for the year 1400-1410, with potential for expansion.
- Performance may vary with dates outside the training dataset.

## Intended Use

This model is intended for applications requiring date recognition and generation in Persian, such as natural language processing tasks, chatbots, or educational tools.

## Acknowledgements

- Special thanks to the developers of the “بلبل زبان” tokenizer and the contributors to the dataset.
