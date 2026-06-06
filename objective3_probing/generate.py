# generate.py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def token_entropy_from_logits(logits, token_mask=None):
    """
    logits: [B, L, V]
    token_mask: optional [B, L] bool. If provided, returns per-sequence mean entropy over True positions.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-8)
    ent = -(probs * log_probs).sum(dim=-1)  # [B, L]
    if token_mask is None:
        return ent
    denom = token_mask.sum(dim=-1).clamp(min=1)
    return (ent * token_mask).sum(dim=-1) / denom  # [B]


def add_gumbel_noise_for_sampling(logits, temperature: float):
    """
    Simple temperature=0 shortcut (no noise).
    Otherwise add Gumbel-like noise to logits before argmax sampling.
    """
    if temperature == 0.0:
        return logits
    # Gumbel(0,1) ≈ -log(-log(U)); to keep it light we use a single -log(U) power,
    # which still injects noise that scales with temperature (good enough for probing).
    u = torch.rand_like(logits, dtype=torch.float32)
    noise = -torch.log(u + 1e-8)
    return (logits.float() - temperature * noise).to(logits.dtype)


def get_num_transfer_tokens(mask_index, steps):
    """
    For each example, split the number of masked tokens in the current block
    into `steps` nearly-equal integers that sum to the masked count.
    Returns: LongTensor [B, steps]
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]
    base = mask_num // steps
    remainder = mask_num % steps
    out = base.expand(-1, steps).clone()
    if (remainder > 0).any():
        idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)  # [1, steps]
        more = (idx < remainder)  # [B, steps] via broadcast
        out = out + more.to(out.dtype)
    return out.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer=None,   # kept for API compatibility, unused
    steps: int = 64,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
):
    """
    Plain sampler (no logging). Fills gen_length tokens using step-wise diffusion-style updates.
    """
    device = prompt.device
    with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length),
                       mask_id, dtype=torch.long, device=device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        for nb in tqdm(range(num_blocks), disable=(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + nb * block_length
            end_idx   = prompt.shape[1] + (nb + 1) * block_length
            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Classifier-free guidance (optional)
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Sample candidate with noise
                noisy_logits = add_gumbel_noise_for_sampling(logits, temperature)
                x0 = torch.argmax(noisy_logits, dim=-1)  # [B, L]

                # Confidence for remasking strategy
                if remasking == "low_confidence":
                    p = torch.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # don't touch beyond current block
                x0_p[:, end_idx:] = -np.inf

                # update masked positions
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k > 0:
                        _, idxs = torch.topk(confidence[j], k=k)
                        x[j, idxs] = x0[j, idxs]
        return x


@torch.no_grad()
def generate_with_entropy(
    model,
    prompt,
    tokenizer=None,   # kept for API compatibility, unused
    steps: int = 64,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
):
    """
    Sampler that records, at every step:
      - mean entropy over the generative region
      - number of *prediction* flips (argmax changes) in the generative region
    Returns:
      x: final tokens [B, L]
      entropy_traj: [S, B]
      flip_traj:    [S, B]
    """
    device = prompt.device
    with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length),
                       mask_id, dtype=torch.long, device=device)
        x[:, :prompt.shape[1]] = prompt.clone()

        # generative region = post prompt
        gen_token_mask = torch.zeros_like(x, dtype=torch.bool)
        gen_token_mask[:, prompt.shape[1]:] = True

        prompt_index = x != mask_id
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        entropy_traj = []   # list of [B]
        flip_traj = []      # list of [B]
        prev_x0_gen = None  # [B, L] argmax predictions last step (gen region only)

        for nb in tqdm(range(num_blocks), disable=(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + nb * block_length
            end_idx   = prompt.shape[1] + (nb + 1) * block_length
            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # CFG
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # === entropy over generative region this step ===
                step_entropy = token_entropy_from_logits(logits, gen_token_mask)  # [B]
                entropy_traj.append(step_entropy.detach().cpu())

                # === prediction flips (soft flips) ===
                x0 = torch.argmax(logits, dim=-1)      # [B, L] argmax predictions for all positions
                x0_gen = x0.clone()
                x0_gen[~gen_token_mask] = -100         # ignore prompt in flip count

                if prev_x0_gen is None:
                    flip_count = torch.zeros(x0_gen.size(0), dtype=torch.long, device=device)
                else:
                    flip_mask = (x0_gen != prev_x0_gen) & gen_token_mask
                    flip_count = flip_mask.sum(dim=1)   # [B]
                flip_traj.append(flip_count.detach().cpu())
                prev_x0_gen = x0_gen.detach().clone()

                # === sampling and commit ===
                noisy_logits = add_gumbel_noise_for_sampling(logits, temperature)
                x0_sample = torch.argmax(noisy_logits, dim=-1)

                if remasking == "low_confidence":
                    p = torch.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0_sample.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0_sample.shape, device=x0_sample.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, end_idx:] = -np.inf
                x0_sample = torch.where(mask_index, x0_sample, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0_sample.device))

                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k > 0:
                        _, idxs = torch.topk(confidence[j], k=k)
                        x[j, idxs] = x0_sample[j, idxs]

        entropy_traj = torch.stack(entropy_traj, dim=0)  # [S, B]
        flip_traj    = torch.stack(flip_traj,    dim=0)  # [S, B]
        return x, entropy_traj, flip_traj


@torch.no_grad()
def generate_with_fli(
    model,
    prompt,
    tokenizer=None,   # kept for API compatibility, unused
    steps: int = 64,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
):
    """
    Like generate_with_entropy, but also logs *where* flips happen.

    Returns:
      x:               final tokens           [B, L]
      entropy_traj:    mean entropy per step  [S, B]
      flips_per_step:  # prediction flips     [S, B]
      flip_pos_mask:   flip positions         [S, B, G]
                       (G = gen_length, True if that position's argmax changed
                        from the previous step)
    """
    device = prompt.device
    B, L_prompt = prompt.shape
    G = gen_length
    L_total = L_prompt + G

    with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        # base sequence (prompt + masked generative tail)
        x = torch.full((B, L_total), mask_id, dtype=torch.long, device=device)
        x[:, :L_prompt] = prompt.clone()

        gen_start = L_prompt
        gen_end   = L_prompt + G
        gen_slice = slice(gen_start, gen_end)

        # mask over generative region in full sequence space
        gen_token_mask = torch.zeros_like(x, dtype=torch.bool)
        gen_token_mask[:, gen_slice] = True

        prompt_index = x != mask_id
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        entropy_traj_list = []   # list of [B]
        flips_per_step_list = [] # list of [B]
        flip_pos_list = []       # list of [B, G] bool
        prev_x0_gen = None       # [B, G] argmax preds in gen region

        for nb in tqdm(
            range(num_blocks),
            disable=(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0),
        ):
            start_idx = gen_start + nb * block_length
            end_idx   = gen_start + (nb + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # ----- forward pass with optional CFG -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits  # [B, L, V]

                # ----- 1) entropy in generative region -----
                step_entropy = token_entropy_from_logits(logits, gen_token_mask)  # [B]
                entropy_traj_list.append(step_entropy.detach().cpu())

                # ----- 2) prediction flips (position-wise) -----
                # argmax predictions *without* sampling noise
                x0_pred = torch.argmax(logits, dim=-1)   # [B, L]
                x0_gen  = x0_pred[:, gen_slice]          # [B, G]

                if prev_x0_gen is None:
                    flip_pos = torch.zeros_like(x0_gen, dtype=torch.bool)   # [B, G]
                else:
                    flip_pos = (x0_gen != prev_x0_gen)                      # [B, G]

                flip_pos_list.append(flip_pos.detach().cpu())
                flips_per_step_list.append(flip_pos.sum(dim=1).detach().cpu())  # [B]
                prev_x0_gen = x0_gen.detach().clone()

                # ----- 3) sampling + commit to x (like generate_with_entropy) -----
                noisy_logits = add_gumbel_noise_for_sampling(logits, temperature)
                x0_sample = torch.argmax(noisy_logits, dim=-1)  # [B, L]

                if remasking == "low_confidence":
                    p = torch.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0_sample.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0_sample.shape, device=x0_sample.device)
                else:
                    raise NotImplementedError(remasking)

                # don't transfer beyond current block
                x0_p[:, end_idx:] = -np.inf

                x0_sample = torch.where(mask_index, x0_sample, x)
                confidence = torch.where(
                    mask_index,
                    x0_p,
                    torch.tensor(-np.inf, device=x0_sample.device),
                )

                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k > 0:
                        _, idxs = torch.topk(confidence[j], k=k)
                        x[j, idxs] = x0_sample[j, idxs]

        # stack over steps → [S, B, ...]
        entropy_traj   = torch.stack(entropy_traj_list,   dim=0)  # [S, B]
        flips_per_step = torch.stack(flips_per_step_list, dim=0)  # [S, B]
        flip_pos_mask  = torch.stack(flip_pos_list,       dim=0)  # [S, B, G]

        return x, entropy_traj, flips_per_step, flip_pos_mask
