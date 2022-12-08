# Tokenizer
[_Tokenizer_]() is the base class of all the _Tokenizer_ objects in this toolkit. 
It on-the-fly transforms text data between strings and tensors.  

For data storage and visualization, the text data should be in the form of strings which is not friendly for model forward calculation. 
For model forward calculation, the text data is better to be in the form of vectors (`torch.tensor` or `numpy.ndarray`).


👆[Back to the handbook page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)

## Table of Contents
1. [**Tokenizer Library**]()
2. [**API Documents**]()


## Tokenizer Library
```
/speechain
    /tokenizer
        /abs.py         # Abstract class of Tokenizer. Base of all Tokenizer implementations.
        /char.py        # Tokenizer implementation of the character tokenizer.
        /subword.py     # Tokenizer implementation of the subword tokenizer by SentencePiece package.
```

## API Document
1. [speechain.tokenizer.abs.Tokenizer.\_\_init__]()
2. [speechain.tokenizer.abs.Tokenizer.tokenizer_init_fn]()
3. [speechain.tokenizer.abs.Tokenizer.tensor2text]()
4. [speechain.tokenizer.abs.Tokenizer.text2tensor]()


### speechain.tokenizer.abs.Tokenizer.\_\_init__(self, token_vocab, **tokenizer_conf)
* **Description:**  
    This function registers some shared member variables for all _Tokenizer_ subclasses: 
    1. `self.idx2token`: the mapping Dict from the token index to token string.
    2. `self.token2idx`: the mapping Dict from the token string to token index.
    3. `self.vocab_size`: the number of tokens in the given vocabulary.
    4. `self.sos_eos_idx`: the index of the joint <sos/eos> token used as the beginning and end of a sentence.
    5. `self.ignore_idx`: the index of the blank token used for either CTC blank modeling or ignored token for encoder-decoder ASR&TTS models.
    6. `self.unk_idx`: the index of the unknown token.
* **Arguments:**
  * _**token_vocab:**_ str  
    The path where the token vocabulary is placed.
  * _****tokenizer_conf:**_  
    The arguments used by `tokenizer_init_fn()` for your customized _Tokenizer_ initialization.

### speechain.tokenizer.abs.Tokenizer.tokenizer_init_fn(self, **tokenizer_conf)
* **Description:**  
    This hook interface function initializes the customized part of a _Tokenizer_ subclass if had.  
    This interface is not mandatory to be overridden.
* **Arguments:**
  * _****tokenizer_conf:**_  
    The arguments used by `tokenizer_init_fn()` for your customized _Tokenizer_ initialization.  
    For more details, please refer to the docstring of your target _Tokenizer_ subclass.

### speechain.tokenizer.abs.Tokenizer.tensor2text(self, tensor)
* **Description:**  
    This functions decodes a text tensor into a human-friendly string.  
    The default implementation transforms each token index in the input tensor to the token string by `self.idx2token`. 
    If the token index is `self.unk_idx`, an asterisk (*) will be used to represent an unknown token in the string.  
    This interface is not mandatory to be overridden. If your _Tokenizer_ subclass uses some third-party packages to decode the input tensor rather than the built-in `self.idx2token`, 
    please override this function.
* **Arguments:**
  * _**tensor:**_ torch.LongTensor  
    1D integer torch.Tensor that contains the token indices of the sentence to be decoded.
* **Return:**  
    The string of the decoded sentence.

### speechain.tokenizer.abs.Tokenizer.text2tensor(self, text)
* **Description:**  
    This functions encodes a text string into a model-friendly tensor.  
    This interface is mandatory to be overridden.
* **Arguments:**
  * _**text:**_ str  
    The input text string to be encoded
* **Return:** torch.LongTensor  
    The tensor of the encoded sentence
