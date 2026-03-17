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
alias py='/scratch/$USER/venvs/platonic-init/bin/python'
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
    /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash
}
