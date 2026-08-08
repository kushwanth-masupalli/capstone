# Phase 0 — Environment Setup (Local GPU)

Your team is training on local GPU machines rather than a shared Colab/Kaggle
environment. That's fine, but it means the usual "everyone has the same
runtime" assumption doesn't hold automatically — each person's driver, CUDA
version, and OS may differ. Do these steps individually, not once for the
whole team.

## Steps

1. **Clone the repo** (already created) and `cd` into it.

2. **Create a virtual environment** — agree as a team on ONE Python version
   first (e.g. 3.10 or 3.11), then:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # venv\Scripts\activate on Windows
   ```

3. **Check your GPU/driver before installing anything:**
   ```bash
   nvidia-smi
   ```
   Note the "CUDA Version" in the top-right of the output — this is the
   *maximum* CUDA version your driver supports.

4. **Install torch matched to your CUDA version** (pick the closest build
   `<=` what `nvidia-smi` reported, from https://pytorch.org/get-started/locally/).
   Example for CUDA 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   **This command may differ between team members' machines — that's expected
   and OK.** What must be identical is the *torch version number itself*
   (e.g. everyone on torch 2.3.x), not the CUDA build suffix.

5. **Install the rest of the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the environment checker:**
   ```bash
   python check_env.py
   ```
   Every team member must see `torch.cuda.is_available(): True` and a
   successful "GPU matmul sanity check: OK" before writing any training code.

7. **Freeze your environment:**
   ```bash
   pip freeze > requirements.txt
   ```
   Commit this. When someone's code "randomly" breaks in three weeks, this
   file is what saves you. If your frozen torch version differs from a
   teammate's, flag it in your team channel now, not after Phase 3 is half-built.

8. **Confirm folder structure** matches (already scaffolded in this repo):
   ```
   project/
     data/                 # gitignored
     notebooks/
     src/
       data/  classifier/  retrieval/  history/
       generation/  hallucination/  explainability/  eval/
     demo/
     checkpoints/          # gitignored
     docs/
   ```

## Checkpoint (do not proceed to Phase 1 until all pass)

- [ ] Every team member ran `check_env.py` and got `torch.cuda.is_available(): True`
- [ ] Every team member is on the same Python version
- [ ] Every team member is on the same torch version (CUDA build suffix can differ)
- [ ] `requirements.txt` is committed
- [ ] Folder structure exists and is pushed to the repo

## Local-GPU specific gotchas

- **VRAM differences across machines**: if your GPUs have different VRAM
  (e.g. one teammate has 8GB, another has 24GB), batch sizes that work for
  one person may OOM for another. Agree on a conservative default batch size
  now (e.g. 16) and let people with more VRAM increase it locally without
  committing that change — put batch size in a config file, not hardcoded.
- **Driver mismatches causing silent CPU fallback**: `pip install torch`
  without the `--index-url` flag sometimes silently installs a CPU-only
  build. Always verify with `check_env.py`, don't assume the install worked.
- **OS differences**: if some teammates are on Windows and others on
  Linux/Mac, watch for path separator bugs (`/` vs `\`) later in Phase 2's
  data loading code — use `pathlib.Path` everywhere instead of string
  concatenation to sidestep this entirely.