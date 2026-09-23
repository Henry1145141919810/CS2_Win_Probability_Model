# PARCC Betty — consolidated guide for this project

Replaces the `e:/CLAUDE_CONTEXT.md` reference in `cluster_runbook.md` (that file did not transfer with
the repo). Sources: the PARCC site pages listed at the end (read 2026-09-13), plus what this project has
learned running its jobs. Companion runbooks: [cluster_runbook.md](cluster_runbook.md) (TCN),
[BETTY_benchmark_guide.md](BETTY_benchmark_guide.md), [BETTY_defuse_guide.md](BETTY_defuse_guide.md).

## 1. What Betty is

Penn's university-wide HPC/AI cluster, run by the Penn Advanced Research Computing Center (PARCC).
Hardware: 31 DGX B200 nodes (8 × B200 GPUs each, 180 GB HBM3e per GPU, 112 CPU cores, 2 TB RAM);
64 AMD Genoa EPYC 9374F standard-memory CPU nodes (64 cores, 384 GB); 10 large-memory CPU nodes
(1152 GB). Storage: VAST (home + projects) and Ceph. Scheduler: Slurm (plus Run:AI for some GPU
workflows). Modules: Lmod (`module avail`, `module load anaconda3`).

## 2. Logging in (two-of-three factors)

Betty accepts any two of: **Kerberos ticket** (PennKey password via `kinit`), **registered SSH key**,
**Duo** (push / text / token). Prerequisites: Duo enrolment, and either the campus network or the Penn
GlobalProtect VPN.

```bash
kinit <PennKey>@UPENN.EDU              # realm MUST be uppercase; ticket lasts 10 h
ssh <PennKey>@login.betty.parcc.upenn.edu   # -> Duo push on first factor pairing
```

Login nodes: `login01/02/03.betty.parcc.upenn.edu` behind the `login.betty` alias.
Recommended `~/.ssh/config` (PARCC):

```
Host *.parcc.upenn.edu
    VerifyHostKeyDNS yes
    GSSAPIAuthentication yes
    ControlMaster auto                 # optional multiplexing: one auth, many sessions
    ControlPath ~/.ssh/control:%h:%p:%r
```

Host-key fingerprints to verify on first connection:
- ED25519 `SHA256:talnzpFHiLmQR0xFrC8ZaPdQ9LxfDMb/iamK2pbBd7I`
- ECDSA `SHA256:rFIuRhUoQP+YApM/qvY1D/EOmxccqGEcOX9rFcOiO6s`
- RSA `SHA256:gQPU5K0JY7/0gTwEmQG9dCF+eCrfqqEXvbDzmVzJZ3c`

Skip Duo on every login: generate `ssh-keygen -t ed25519`, register once with
`ssh-copy-id <PennKey>@login.betty.parcc.upenn.edu`; then key + Kerberos ticket satisfies 2-of-3.
Gotcha from this project: a locally activated conda env can shadow the system `kinit`; deactivate it
first (not an issue inside WSL, which has no conda).

**Windows:** PARCC's supported route is WSL2 + Ubuntu with `krb5-user` (alternatives: MIT Kerberos for
Windows + MobaXterm/SecureCRT with domain `UPENN.EDU`, GSSAPI = MIT Kerberos, login `pennkey@UPENN.EDU`).
This machine's WSL setup is in §7.

## 3. Rules that get accounts suspended (code of conduct)

- **No compute on login nodes.** They are for job submission/monitoring, editing, light compilation
  (`make -j4` is the stated ceiling). No `python train.py`, no MATLAB, no heavy `rsync`/`scp`.
  Use `srun`/`sbatch`/`salloc` (or `interact`) for anything else.
- **Large data goes through Globus**, not scp on a login node. (Project convention: scp is fine
  under ~1 GB; the 84 MB training table was scp'd.)
- **No account sharing**, even within the group. Never bypass the scheduler to reach a compute node.
- **Publications must acknowledge the allocation** and be reported in the ColdFront portal
  (https://coldfront.parcc.upenn.edu, "Login with PennKey"). Paper v12 already carries PARCC's
  official wording.
- New accounts and project memberships take up to an hour to propagate.

## 4. Storage

| Path | Purpose | Notes |
|---|---|---|
| `/vast/home/<letter>/<PennKey>` (`~`) | configs, code, conda envs (`~/envs/cs2-rwp`) | 50 GB; code only |
| `/vast/projects/ajw/wharton/cs2-rwp` | **this project**: repo clone, `data/`, `logs/`, `checkpoints/`, `outputs/` | Prof. Wyner's allocation (`ajw`), 1 TB |
| local NVMe on compute nodes | scratch during a job | 800 GB, not persistent |

Quota/usage: `parcc_quota.py`, `parcc_du.py /vast/projects/ajw/wharton/cs2-rwp`. The login MOTD prints a
storage table.

## 5. Slurm on Betty

Partitions this project uses: `b200-mig45` (one MIG slice of a B200; smoke tests and all holdout
jobs), `dgx-b200` (a full B200; large sweeps), `genoa-std-mem` (CPU; env builds). Check availability
with `parcc_sfree.py`, your QOS limits with `parcc_sqos.py`, past usage with
`parcc_sreport.py --user <PennKey>`, failures with `parcc_sdebug.py --job <id>`.

**CPU-per-GPU filter (since 2026-08-03):** `sbatch` rejects a wrong ratio with
`CPUS_PER_GPU_MISMATCH`. Required `--cpus-per-task` per GPU: **6 on b200-mig45**, 14 on b200-mig90,
28 on dgx-b200; memory is capped at 8 GB per CPU (so 48 GB with 6 CPUs). All `jobs/*.sh` already
comply.

Template used by every job in `jobs/` (see `jobs/tcn_holdout.sh`):

```bash
#SBATCH --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=00:30:00
#SBATCH --output=logs/%x_%j.out          # relative to the directory you run sbatch from
module load anaconda3 && source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"; cd "$PROJ"   # submit from the repo root
```

Interactive debugging (20 min on a MIG slice):

```bash
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=00:20:00 --pty bash
```

Environment: built once on a compute node by `sbatch jobs/setup_env.sh` (conda env at `~/envs/cs2-rwp`,
Python 3.11, torch cu128 for the B200, polars/numpy/scikit-learn/pyarrow). Never `pip install --user`.
Project habit: always smoke-test with `--limit-matches 20 --epochs 3` before a full job; long loops
checkpoint every 10 iterations.

## 6. Moving data

- Small files (< ~1 GB): `scp file <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/data/`
  (the SSH multiplexing block makes repeated scp free of re-authentication).
- Large: Globus web app (https://globus.org → University of Pennsylvania → PennKey), collections
  "vast parcc" / "ceph parcc"; or `globus-cli` from WSL (installed by the setup script) with Globus
  Connect Personal on the laptop.
- What currently lives on Betty (as of the last session there, 2026-08-31): the repo clone,
  `data/training_dataset.parquet` (defuse re-parse version), `data/test_dataset_2026_lag2025.parquet`,
  `data/test_dataset_2026_defuse.parquet`, `data/trajectory_dataset.parquet`, deep OOF/holdout parquets
  in `outputs/`. Verify with `ls -la data outputs` after login; re-sync from the local supplement if
  the schema guard in the deep scripts complains.

## 7. This laptop's WSL setup (done / to do)

Done on 2026-09-13 (needed admin, UAC accepted): `wsl --install --no-distribution` → WSL 2.7.14
installed, `VirtualMachinePlatform` enabled. **A reboot is required before any distro can run.**

After the reboot, run once from the repo root (no admin needed):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wsl_parcc.ps1 -LinuxUser <PennKey>
```

It registers Ubuntu 24.04 without the interactive first run and executes
`scripts/setup_ubuntu_parcc.sh` as root, which installs `krb5-user`, `openssh-client`, `rsync`, `git`,
`tmux`, `python3`, `pipx`, `globus-cli`; writes `/etc/krb5.conf` for `UPENN.EDU` (KDCs
`kerberos1-4.upenn.edu`, confirmed by DNS SRV records); creates a Linux user named after your PennKey with
passwordless sudo as the default WSL user with systemd; writes the PARCC `~/.ssh/config` with a `betty`
alias; generates an ed25519 key; and adds shell aliases `kb` (kinit), `betty` (ssh), `bettyproj` (ssh
straight into the project dir), `bettymux` (persistent master connection).

First interactive use (once): `wsl` → `kb` → `betty` (Duo push) → `ssh-copy-id betty`. Afterwards a
ticket plus the key logs in without Duo.

## Sources

- [Logging In](https://parcc.upenn.edu/training/getting-started/logging-in/) ·
  [Windows Setup](https://parcc.upenn.edu/training/getting-started/logging-in/windows-setup/) ·
  [Looking Around](https://parcc.upenn.edu/training/getting-started/looking-around/) ·
  [PARCC Tools](https://parcc.upenn.edu/training/getting-started/parcc-tools/) ·
  [Betty system page](https://parcc.upenn.edu/systems/betty/) ·
  [Code of Conduct](https://parcc.upenn.edu/about/code-of-conduct/) ·
  [Globus Data Transfer](https://parcc.upenn.edu/training/storage/globus-data-transfer/) ·
  [Getting Started index](https://parcc.upenn.edu/training/getting-started/)
