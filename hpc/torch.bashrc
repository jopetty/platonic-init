# Torch HPC shell helpers

export EDITOR=vim
export VISUAL=vim

alias ll='ls -lh'
alias la='ls -lah'
alias ta='tmux attach -t torch'
alias tls='tmux ls'
alias tn='tmux new -s torch'
alias sq='squeue -u "$USER"'
alias so='sacct -u "$USER" --format=JobID,JobName%30,Partition,State,Elapsed,MaxRSS,ExitCode'
alias si='sinfo'
alias py='/ext3/venvs/platonic-init/bin/python'
alias repo='cd /scratch/$USER/platonic-init'

sj() {
  scontrol show job "$1"
}

sqw() {
  watch -n 5 "squeue -u $USER"
}

slurm-log() {
  tail -f "$1"
}

torch-shell() {
  singularity exec --nv \
    --overlay /scratch/$USER/uv-env/uv-python.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash
}
