# Source your ROS2 workspace first (contains robot_interfaces and control nodes)
source /home/user/qingyu/robot/Rokae_Ros2_ws_25/install/setup.bash

# Run the Rokae ROS2->HTTP bridge used by FrankaEnv-compatible JAX pipelines
python rokae_ros2_server.py \
    --flask_url=127.0.0.1 \
    --flask_port=5000 \
    --joint_positions_topic=/joint_positions \
    --end_pos_topic=/EndPos_Topic \
    --end_vel_topic=/end_velocity \
    --end_control_topic=/EndControl_Topic \
    --force_sensor_topic=/ForceSensor_Topic \
    --control_mode=3 \
    --reset_control_mode=4 \
    --home_pose=0.41254,-0.04786,0.29984,-3.1415926,0.00087,-0.36408
