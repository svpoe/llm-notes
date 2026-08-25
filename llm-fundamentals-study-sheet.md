# LLM Fundamentals — Study Sheet

## Step 1) Tokenization — BPE, subword units, fixed vocabulary

- We need to turn text into numbers for the NN
- **Byte-Pair Encoding (BPE)** breaks text into subword tokens from a fixed, pre-built vocabulary (50k-100k tokens)
    - deterministic, built once before training, no changes after
- Each token gets an ID. The token ID that tokenization produces is just an index; the ID is what gets looked up in the embedding table. What that index actually *means* numerically is entirely up to the learned embedding matrix.

## Step 2) Embedding

- Each token ID gets converted into a vector (dense vector) via lookup into a trainable embedding matrix
    - embedding matrix randomly initialized, improved throughout entire pretraining, SFT and alignment via loss calculation and backprop
- Every instance of a given token has an identical vector (just plausible meaning w/o context, needing to be improved)

## Step 3) Self-attention — tokens finally share context

**Query, key, value**
- Each token is projected into query, key, value vectors by matrix multiplication separately with three trainable matrices `W_Q`, `W_K`, `W_V`
- Each token's query Q is compared to all tokens' keys K to get a relevance score. The score for all keys that come after the query's position are overwritten to negative infinity. Then the score goes through softmax to become usable weights. Softmaxed scores are dot-producted with the value vectors to produce the usable context-aware vector
- This is done by multiple heads at once. Outputs are combined via concatenation. Final projection (an ordinary linear layer) occurs when the concatenated context-aware vectors (each a different perspective/opinion of sorts) is multiplied by a learned weight matrix `W_O`. Now we have the final output of the multi-head attention block
- Repeat in multiple layers (multi-head attention block + feedforward MLP)
    - "attention is communication, feedforward is computation"
- Note: because this is decoder-only, causal masking is used to ensure tokens only see themselves and earlier tokens (essential for honest token prediction)

## Step 4) Turn final context vector into a token prediction

**Classification**
- Vector gets projected (via linear layer) into logits
- Logits softmaxed into a probability distribution over all possible next tokens (classification w/ tens of thousands of classes)

---

**That's the complete path:** text → tokens → embeddings → attention-built contextual meaning → next-token classification → trained via a three-stage pipeline → deployed via sequential, cached, sampled generation.

---

## Follow-up concepts

### Prompting — zero/few-shot, chain-of-thought, system prompts

- Text the user provides to the context window
- Does not change weights
- **Zero/few-shot** — directly ask, or give a couple examples
- **Chain-of-thought** — has the model write out reasoning steps first, since those tokens become context later tokens can attend to, improving complex answers
- **System prompts** — instructions about overall persona

### Tool use / function calling

- **Tool description sent:** you call the LLM model host's API, passing in a list of available tools (each w/ name and description) and a JSON schema describing its arguments
    - This tool list is placed into the model's context (along with the conversation so far)
- **Model makes tool call:** the model doesn't explicitly choose a tool, but instead when it does next-token prediction, it has learned to produce tokens representing a call
- **Host application executes** the called tool
    - Parses out the tool name and arguments and executes a function or API call
- **Results are fed back** to the model as new context
- **Model generates again** with the tool's result in context
    - Generates a new tool call or provides a natural-language answer to the user

**When does tool calling behavior become an agent?**
- When the model can chain many tool calls together across multiple steps before producing a final answer

### RAG — embeddings for retrieval, vector search, the retrieval pipeline

Use when we need additional context, like info on company private documents.

- Embed the documents: break each into chunks and run through an embedding model (specialized NN). Text chunks get converted into vectors (similar to token embeddings, but now an embedding represents the meaning of a text chunk)
- Text chunk meaning embeddings get stored in a vector database
    - Optimized for fast similarity search (approximate nearest neighbor algorithms)
- When a user sends a query, embed it the same way. Then use cosine similarity to find the top-k most relevant chunks to retrieve
- Insert the retrieved relevant text into the model's context window alongside the user's question (with instructions like "answer using the following context")
- This helps hallucination! The model isn't asked to recall from just its trained weights. Instead it reads and summarizes text directly in the context

---

## Papers to refer to 

- **Transformer architecture / self-attention (Step 3):**
  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. arXiv:1706.03762. https://arxiv.org/abs/1706.03762

- **Rotary position embeddings (RoPE), mentioned re: positional encoding:**
  Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. https://arxiv.org/abs/2104.09864

- **Byte-Pair Encoding / subword tokenization (Step 1):**
  Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. Proceedings of ACL 2016. https://arxiv.org/abs/1508.07909

- **RLHF / alignment via human preference (SFT → alignment pipeline):**
  Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). *Training Language Models to Follow Instructions with Human Feedback*. arXiv:2203.02155. https://arxiv.org/abs/2203.02155

- **DPO, the simpler alternative to RLHF mentioned for alignment:**
  Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290. https://arxiv.org/abs/2305.18290

- **RAG (Retrieval-Augmented Generation section):**
  Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
