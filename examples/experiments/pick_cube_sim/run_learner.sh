export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=first_run \
    --demo_path=/home/jyang159/projects/hil-serl/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_20_demos_2025-01-22_18-19-38.pkl \
    --learner \