import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt
import math

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.dt = 1/60.0
sim_params.use_gpu_pipeline = False
sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load ball asset
asset_root = "resources/robots/bruce/"
asset_file = "bruce.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

a = gym.get_asset_rigid_shape_properties(asset)
for idx, shape in enumerate(a):
    shape.filter = idx
if gym.set_asset_rigid_shape_properties(asset, a):
    print("set filter done!")

env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# set random seed
# np.random.seed(17)

env = gym.create_env(sim, env_lower, env_upper, 1)
pose = gymapi.Transform()
pose.r = gymapi.Quat.from_euler_zyx(-math.pi*0.5, 0, 0)
pose.p = gymapi.Vec3(0, 0.9, 0)
handler = gym.create_actor(env, asset, pose, None, -1, 1)



cam_pos = gymapi.Vec3(1.5, 0.5, 2)
cam_target = gymapi.Vec3(-10, -2.5, -13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))



while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)