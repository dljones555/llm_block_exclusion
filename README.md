# llm_block_exclusion
Use headers (type, affordance) separate and POS tags in training. Load as bit layouts separate from vectors in inference as tensors. Apply block-level exclusion before attention to only use compatible blocks with any matching tokens. Use L2 caching and custom Triton kernels.
