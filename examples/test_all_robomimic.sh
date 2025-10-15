# test alll robomimic  task in debug mode
for task in lift_ph_state lift_mh_state can_ph_state can_mh_state square_ph_state square_mh_state transport_ph_state transport_mh_state tool_hang_ph_state; do
    uv run examples/train_robomimic.py -cn exps/debug.yaml task=$task
done

for task in lift_ph_image lift_mh_image can_ph_image can_mh_image square_ph_image square_mh_image transport_ph_image transport_mh_image tool_hang_ph_image; do
    uv run examples/train_robomimic.py -cn exps/debug.yaml task=$task
done
