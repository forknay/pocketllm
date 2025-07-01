# Pocket LLM
Low scale LLM for autocompletion on open corpus database (currently Tiny Shakespeare, 1.1M tokens)
<br>
Change max_length in the generate function to decide how many tokens to generate (context length will stay block_length regardless), toggle nb_iter if you want to train
<br>
**Character level tokenizer:**
<br>
Training Loss: 1.33 
<br>
Validation Loss: 1.55
<br>
**Byte-Pair Encoding tokenizer:**
<br>
Training Loss: 1.56
<br>
Validation Loss: 3.34 
<br>

# To-DO
- Add device support (cuda, load move all params to cuda)

- Add weight decay

- Add Multi-Latent Attention

- Gradient Accumulation

- Gradient clipping

- RoPE

<br>

## Resources

[Andrej Karparthy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6045s)
[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
[DeepSeek-V3 Github](https://github.com/deepseek-ai/DeepSeek-V3)
(and many many youtube videos & medium articles)
