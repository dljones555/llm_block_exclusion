# llm_block_exclusion
Use headers (type, affordance) separate and POS tags in training. Load as bit layouts separate from vectors in inference as tensors. Apply block-level exclusion before attention to only use compatible blocks with any matching tokens. Use L2 caching and custom Triton kernels.

MERF - Meta Exclusion Routing Framework
Copyright (C) 2026, David Jones / fullcircle.ai

Attention is compute & energy ineffienct with 90%+ irrevelant recomputed token pairings that reveal semantic, pattern emergence and relationships in inference. In human biology, thoughts and neurotransmittes broadcast but only bind to the right type of selective receptors and ligands. This is exlcusion by type. In quantum physics, we have notions of where particle should be based on properties, rules pointing the right, and statistical distribution. It's not brute force, it's exlcusion. 

We need exclusion with attention.

This is achieved through extracting POS types at training and creating headers of these types and related affordances to use as exclusion filters in hierarchal memory on GPU's without altering GPU token architecture as is. Despite new GPU/TPU gains, most infra hardware is Hopper or Blackwell grade. We can still do better in software. The research and techniques to be implemented and synthesized are there. It's like a Tetris puzzle and putting these in place. Improving sparse tasks is the goal. GPU compute for naturally sparse domains like coding languages and equations excels. Natural language, not so much.

This can be achieved in software with possible shifts to silicon. It's like Windows ontop DOS and PC's (our first gen of mass AI and software equivalent with GPU's, transformers and brute force) as we shift to a more pure, sophisticated version of AI. Precendents include GUI's evolving on Macs, then NextSTEP and Jobs return towards a better MacOS and eventual iOS with underlying POSIX compliance towards a pure, elegant, universal system. Think other shifts like JavaScript to jQuery and finally TypeScript / React/web components or the evolution of UNIX to Linux.

We are in an early transitional 1st mass phase of AI that can be optimized/maximized in software as better forms are built out.

Plan:

- MVE on Colab GPU with Triton kernel with dummy syntethic data and scores. Vet idea vs traditional attention.
- MVE vet block exclusion: for a given token with type/affordances include that block. It's GPU warp friendly.

Larger POC:

- In training, do aux loss POS tagging 
- Save POS tagging as JSON file in model
- Create POS to enum/bitmask functions
- Create POS to afforance and bitmask functions
- Kernel: Load header in hierarchal memory token_id/ptr, type, affordance, energy, age
- Kernel: Load headers as hierarchal memory apart from tokens to use for exclusion w/ bitmask
- Implement exclusion before attention to match and route to the right tokens
+ Load token headers in cache or SDRAM / SKMEM in attention
+ Implement exclusion filter based on type & affordance bitmask filtering for sparsity, compute/efficiency improvement
- Write exlcusion routing fabric API with functions: match, route, bind, update, etc.
- Kernel: Implement monarchy tiling
- Use case validation: QA, docs, agent tasks
- Share results, try for funding and others to work with

1/10/2026

POS list: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
X Grok talk: POS/afforance encoding
spacy or stanza as library and model also HF
