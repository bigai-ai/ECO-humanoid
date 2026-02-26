import Startups.memory_manager as MM
import time
import numpy as np
import csv
freq = 100
sum_energy = 0
t0 = time.time()
truth_data = []
des_data = []
f = open('/home/bigeast/humanoid-gym/humanoid/scripts/gazebo/'+time.strftime("%Y%m%d_%H%M%S_")+'test_real'+'.csv', 'w', encoding='utf-8', newline='')
#head = [f'q{i}' for i in range(10)] + [f'dq{i}' for i in range(10)]  + [f'tau{i}' for i in range(10)] + [f'tau_des{i}' for i in range(10)]
head = ['t'] + ['step', 'left_foot_position_z', 'energy', 'sum_energy', 'px', 'py', 'pz', 
                    'rx', 'ry', 'rz',
                    'vx', 'vy', 'vz', 
                    'vbb_x', 'vbb_y', 'vbb_z', 
                    'wx', 'wy', 'wz']+ [f'q{i}' for i in range(10)] + [f'dq{i}' for i in range(10)]  + [f'tau{i}' for i in range(10)] + [f'tau_des{i}' for i in range(10)]

writer = csv.writer(f)
writer.writerow(head)

        
def log_data(writer, time, step, foot_world_pos, energy, sum_energy, p_wb, v_wb, R_wb, w_bb, v_bb, q, dq, truth_torques, des_torques):
    # data = q
    # data += dq

    data = [time]
    data += [step]
    data += [p_wb[2] - foot_world_pos]
    data += [energy]
    data += [sum_energy]
    data += list(p_wb)
    data += list(R_wb)
    data += list(v_wb)
    data += list(v_bb)
    data += list(w_bb)
    data += list(q)
    data += list(dq)
    data += list(truth_torques)
    data +=  list(des_torques)

    writer.writerow(data)
step = 0    
sum_sum_energy = 0
while  time.time() - t0 < 50.0:
    t = time.time()
    leg_data = MM.LEG_STATE.get()
    leg_command = MM.LEG_COMMAND.get()
    
    estimation_data = MM.ESTIMATOR_STATE.get()
    p_wb  = estimation_data['body_position']
    v_wb  = estimation_data['body_velocity']
    R_wb  = estimation_data['body_rot_matrix']
    w_bb  = estimation_data['body_ang_rate']
    foot_world_pos = estimation_data['left_foot_position'][2]
    v_bb = R_wb.T @ v_wb
    #print(leg_data['joint_torques'])
    energy = np.sum(np.abs(np.array(leg_data['joint_velocities']*leg_data['joint_torques'])))
    sum_energy += energy
    log_data(writer, time.time(), step, foot_world_pos, energy, sum_energy, p_wb, v_wb, R_wb, w_bb, v_bb, leg_data['joint_positions'], leg_data['joint_velocities'], leg_data['joint_torques'], leg_command['goal_torques'])

    while time.time() - t < 1. / freq:
        pass
    print(step, sum_energy, time.time() - t0)
    step += 1
