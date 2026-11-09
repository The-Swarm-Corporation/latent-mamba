# Latent Mamba: Architecture Notes

These are the design notes behind Latent Mamba, from the original Claude conversation where the architecture was worked out. The interactive architecture diagram is here:

**[Claude conversation and architecture diagram](https://claude.ai/chat/a327398f-fd14-42ac-928a-e84e1de7229e?artifact=6098850e-24e4-4ba1-80aa-fee723e678e0)**

> **Note:** the implementation described at the end of these notes, originally `latent_mamba.py`, is now [`latent_mamba/main.py`](../latent_mamba/main.py). It has no `--selftest` or `--demo` flags. `python latent_mamba/main.py` runs `self_test()` and then `run_demo()`, and you can also import and call either one directly. The PyTorch self-test has since been run and passes. See the [README](../README.md) for usage.

---

## The question

> How can we enable the Mamba architecture to think in latents adaptively? Essentially, how can we build Latent Mamba?

The core idea is to make Mamba feed its own final hidden state back in as the next input, without emitting a token, and to let a small halting head decide how many of these silent steps to take. Mamba is unusually well suited to this. Each latent step only updates a fixed-size state, so thinking longer costs no extra memory.

## Why Mamba is a natural host

The method to borrow is Coconut. Instead of decoding the last hidden state into a word, Coconut feeds it straight back into the model as the next input embedding, in continuous space. In a Transformer, every one of those thoughts adds an entry to the KV cache. In Mamba, a thought just updates the SSM state and the conv state, so memory per step stays constant. ([Coconut](https://arxiv.org/abs/2412.06769))

The loop also adds expressivity. Mamba's recurrence over time is linear in the state, which limits what one forward pass can track. When the output of the full network, with all its MLPs and gates, is fed back as the next input, the thinking loop becomes a deep nonlinear RNN over thought steps. That is where extra reasoning capacity can come from.

**Mamba collapses "thinking in depth" into "thinking in time."** The latent-reasoning literature describes two axes. Vertical depth controls how much the model can reason at each step. Horizontal continuity controls whether that reasoning carries forward in the latent state or has to be rebuilt at each position. Recurrent-depth models like Huginn take the vertical route. They use a prelude that embeds tokens into latent space and a core block that is iterated to refine the latent state, followed by a coda. ([Latent reasoning survey](https://arxiv.org/abs/2507.06203), [Huginn](https://arxiv.org/abs/2502.05171))

If you do that with Mamba and give each core iteration its own SSM state, you hit a problem at inference. When token t exits after 3 iterations and token t−1 ran 8, the states for iterations 4 to 8 are missing. If instead you let all iterations share one state, depth recurrence becomes the same thing as inserting r latent steps between tokens along the sequence. So the cleanest Mamba-native design is horizontal: a variable number of self-generated latent micro-steps between tokens, all writing into the same state.

## Architecture

The design needs four additions to a pretrained Mamba-2:

1. **Thought projector.** The final hidden state, after the last norm, lives in output space, not input-embedding space. Mamba computes Δ, B and C from its input, so off-distribution inputs push these projections into odd regimes. Pass the hidden state through RMSNorm and a linear layer (or a small MLP) before feeding it back.
2. **Thinking-mode embedding.** Add a learned vector to every thought input. Without it, every latent step multiplies the state by exp(ΔA) and the prompt slowly decays away. The mode flag lets the model learn a small Δ while thinking, so it keeps the prompt and integrates gently.
3. **Full state carry.** Carry both the SSM state and the causal-conv state (a window of about 4). Thoughts pass through the conv just like tokens, and forgetting the conv state is a common bug.
4. **Halting head.** A small MLP reads the hidden state (or a pooled summary of the SSM state) and outputs P(halt). This is what makes the model adaptive.

The core loop at inference:

```python
def think(self, cache, h, max_steps=32, eps=0.01):
    # cache: per-layer (conv_state, ssm_state) after the prompt, fixed size
    p_continue = 1.0
    for n in range(max_steps):
        x = self.thought_proj(h) + self.mode_emb   # hidden state -> input space
        h = self.backbone.step(x, cache)           # updates conv + SSM state in place
        lam = torch.sigmoid(self.halt_head(h))     # P(halt now | not yet halted)
        p_continue = p_continue * (1 - lam)
        if p_continue < eps:
            break
    return h, cache, n + 1                         # then decode the answer from here
```

## Training recipe

**Stage 1: CoT fine-tune.** Fine-tune the base Mamba on chain-of-thought data with `<bot>`/`<eot>` markers.

**Stage 2: Coconut curriculum.** At stage k, the first k reasoning steps of the CoT are replaced with k × c continuous thoughts. The curriculum matters: training Coconut without the multi-stage curriculum significantly hurts performance. One change I'd make is to sample c randomly per example rather than fixing it, so the model works at any thought count. This mirrors Huginn, where the number of recurrent iterations is sampled randomly during training so the model generalizes to different compute budgets at test time. ([Coconut](https://arxiv.org/abs/2412.06769), [Huginn](https://arxiv.org/abs/2502.05171))

The mechanics of each training example:

1. Run the prompt with the parallel scan.
2. Run the n thought steps one at a time, carrying the state forward.
3. Run the answer with the parallel scan, starting from the final state.

The thought steps need to be differentiable. Use the chunked scan with an initial state (Mamba-2's kernels accept one), or a plain PyTorch reference step. Fused inference step kernels generally aren't built for backprop. With n ≤ 32, this sequential stretch is cheap. If you need much longer loops, backprop through only the last k steps (truncated backprop), as Huginn does.

**Stage 3: Halting.** Three options, roughly in order of cost:

- **Zero-shot convergence exit.** Stop when consecutive steps stop changing anything. Huginn varies iterations per token using a convergence test such as the KL divergence between successive steps, and RD-VLA stops adaptively once its latent state converges. This needs no extra training and is a good baseline. ([Huginn](https://arxiv.org/abs/2502.05171), [RD-VLA](https://arxiv.org/abs/2602.07845))
- **PonderNet/ACT-style loss.** Define p_n = λ_n ∏_{j<n}(1−λ_j). Minimize Σ_n p_n · CE(answer | state after n steps) + β·KL(p ‖ geometric prior). To keep it affordable, evaluate the answer loss at a few sampled values of n rather than every n.
- **RL with a compute penalty.** Use GRPO with reward = correct − λ·steps. This is the most direct way to get "think more only when needed."

## Pitfalls to watch for

- **Thoughts collapse into pause tokens.** The model may learn to ignore the content of its thoughts. Test this by replacing the thoughts with zeros or noise. If accuracy holds, the loop isn't doing real work.
- **Prompt decay over long loops.** Watch accuracy versus step count. If it peaks and then drops, the state is forgetting the prompt. Mamba–attention hybrids help here, because attention layers can look back at the prompt's KV entries. Thoughts can be kept out of the KV cache so memory stays constant.
- **State tracking limits.** If your tasks need parity- or permutation-style tracking, consider letting the state transition take negative eigenvalues. Diagonal SSMs restricted to positive decay struggle with these.

## Prior work to borrow from

- **Huginn** (Geiping et al., NeurIPS 2025) is the reference for training random-depth recurrence. It needs no specialized training data and was trained from scratch at 3.5B parameters on 800B tokens. ([paper](https://arxiv.org/abs/2502.05171))
- **TRM with a Mamba-2 hybrid** (Feb 2026) swaps Mamba-2 blocks into a tiny recursive reasoning model with a Q-halt head. It explicitly leaves open whether the recursive loop can be absorbed into Mamba-2's own state updates, which is exactly the design above. ([paper](https://arxiv.org/abs/2602.12078))
- **A community 2.8B Mamba** (unreviewed), `mamba-2.8b-latent`. It was fine-tuned to reason in its hidden state, using spacer tokens as internal clock cycles, plus a halting head. It reports about 2 loops on average for HellaSwag versus 5.9 for ARC-Challenge. Useful as a reference implementation, but treat the numbers cautiously. ([Hugging Face](https://huggingface.co/batteryphil/mamba-2.8b-latent))
- **An SSM halting paper** (arXiv 2603.04180). Instead of a dedicated halting module, it adds an energy penalty during training, and halting emerges from how that penalty interacts with the SSM's recurrent state. ([paper](https://arxiv.org/abs/2603.04180))
- **SST V2** shows how to train nonlinear latent recurrence in parallel. It streams latent states across the sequence and uses extra passes per position for deliberation. Worth a look if sequential thought steps become a training bottleneck. ([paper](https://arxiv.org/abs/2605.00206))

## Architecture diagram

The interactive version is in the [Claude conversation](https://claude.ai/chat/a327398f-fd14-42ac-928a-e84e1de7229e?artifact=6098850e-24e4-4ba1-80aa-fee723e678e0). In it, the pretrained Mamba-2 is purple and the added modules are teal. Here is a text version:

```
prompt ... <bot>                                                 <eot> answer
     |                                                             |
     v                                                             v
token embedding --> Mamba-2 backbone --> final hidden h --> halting head
                      ^  (fixed-size conv + SSM state)          |      |
                      |                                     continue  halt
                      |                                         |      |
                      +----------- thought projector <----------+      v
                                     (+ mode embedding)              LM head
```

The halting head is the only thing that decides between thinking and talking. Each pass through the right-hand loop is one silent thought, and it updates only the fixed-size state, so memory stays flat however long the model thinks. When the head says halt, the LM head emits a token, and that token goes back in through the embedding as usual.

## Implementation

The implementation is a single typed PyTorch file with full docstrings, and it covers every box in the diagram. The SSM is pure PyTorch, with no CUDA kernels.

- **Mamba-2 mixer.** For prompts and answers it uses the chunked SSD scan, starting from the stored state. Each thought uses an exact single-token step. All state handling is functional, so the cache can be branched and gradients flow through every thought.
- **The new modules.**
  - `ThoughtProjector` starts as plain Coconut (an identity mapping at token-embedding scale) plus a mode embedding initialised to zero.
  - `HaltingHead` starts out outputting the prior halting probability at every step.
- **The training recipe.**
  - `latent_loss` covers the language-CoT stage (zero thoughts) and the Coconut curriculum stages. Thought counts can differ per example.
  - `ponder_loss` is the PonderNet objective. It scores the answer from all N thought steps in one batched pass.
  - `generate` supports learned, convergence and fixed halting. Halted examples are frozen while the rest of the batch keeps thinking. Optional `ablate="zeros"`/`"noise"` replaces the thought content, to check whether the thoughts actually matter.
- **A built-in demo.** `run_demo()` trains a small model through all stages on a pointer-chasing task. It reports accuracy and thoughts used by hop count, with no-thought and ablated baselines alongside.

The numerical core was first checked with a line-for-line NumPy port. The chunked scan matched the step-by-step recurrence to about 1e-14, across multiple chunks, padding and a non-zero initial state. Prefill followed by token-by-token continuation matched a single full pass. The halting distribution summed to 1 and equalled the geometric prior at initialisation. `self_test()` repeats these checks in PyTorch, together with gradient and generation tests.

## Sources

- [Coconut: Training LLMs to Reason in a Continuous Latent Space (arXiv 2412.06769)](https://arxiv.org/abs/2412.06769)
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (NeurIPS 2025)](https://arxiv.org/abs/2502.05171)
- [Recurrent-Depth VLA (arXiv 2602.07845)](https://arxiv.org/abs/2602.07845)
- [Tiny Recursive Reasoning with Mamba-2 Attention Hybrid (arXiv 2602.12078)](https://arxiv.org/abs/2602.12078)
- [Architectural Proprioception in SSMs (arXiv 2603.04180)](https://arxiv.org/abs/2603.04180)
- [State Stream Transformer V2 (arXiv 2605.00206)](https://arxiv.org/abs/2605.00206)
- [batteryphil/mamba-2.8b-latent (Hugging Face)](https://huggingface.co/batteryphil/mamba-2.8b-latent)
- [A Survey on Latent Reasoning (arXiv 2507.06203)](https://arxiv.org/abs/2507.06203)
