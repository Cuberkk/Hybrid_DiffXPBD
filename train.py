import numpy as np
import warp as wp
import argparse
from utils.DiffXPBD import DiffXPBDTapeFramework3D_Warp

STOP_CONDITION = ["convergence", "thresholding", "both"]
OPTIMIZED_METHOD = [None, "simultaneous", "alternating"]
OPTIMIZE_SUBJECT = ["Youngs_Modulus", "Applied_Force", "Force_Amplification", "Damping_Amplification"]
OPTIMIZER = [None, "Adam"]

def main(total_time=50., 
         dt = 1.e-2,
         trajectory_type="kps", # "kps" or "mesh_pts"
         optimize_type = "sim2real", # "sim2sim" or "sim2real"
        
         mesh_file_path = f"mesh/ad_gripper_less_stiff.msh",
         contact_pos = "mid", # up, mid, down
         sim_target_path = "data/Youngs_9500000_time_200_dt_0.01_sweep_20_damping_amp_1.00e+00.npz",
         constant_applied_force = (-5.0, 0.0, 0.0),
         series_force_mode_switch = True,
         gripper_side = 'left', # 'right' or 'left'

         youngs_init = 95. * 1e6,
         force_amplification = 1.0,
         damping_amplification=1.0, 
         sweep_count=20,

         kp_init_path = "data/PosEffs_all_pts_left.json",
         
         max_epochs = 20,
         optimized_subject_indices = [0],
         stop_condition_index = 2,
         convergence_patience = 5,
         relative_change_threshold = 1.0e-3,
         optimized_style_idx = 0,
         alternating_epochs = 100,
         optimized_lr = [2.e-1, 1.e-2],
         thresholding_eps = 1.0e0,
         optimizer_idx = 0
         ):
    
    wp.init()
    print("CUDA available:", wp.is_cuda_available())
    print("Preferred device:", wp.get_preferred_device())
    DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"

    total_steps = int(total_time / dt)
    real_target_kp_path = f"data/kps_trajectories/{contact_pos}_contact/synced_kps.csv"
    real_applied_force_path = f"data/kps_trajectories/{contact_pos}_contact/synced_forces.csv"

    optimized_subjects = [OPTIMIZE_SUBJECT[i] for i in optimized_subject_indices]
    optimized_str = ""
    for i in range(len(optimized_subjects)):
        optimized_str += f"{optimized_subjects[i]}"
        if i != len(optimized_subjects) - 1:
            optimized_str += "_"

    print(f"\nSimulation total time: {total_time}s, Time step: {dt}s, Total steps: {int(total_time/dt)}")
    print(f"Initial Young's Modulus: {youngs_init}, Force Amplification: {force_amplification}, Damping Amplification: {damping_amplification}, Sweep Count: {sweep_count}")
    print(f"Optimization type: {optimize_type}, Trajectory type: {trajectory_type}")
    print(f"Contact position: {contact_pos}, Gripper side: {gripper_side}")
    print(f"Optimized subjects: {optimized_subjects}, Learning rates: {optimized_lr}")
    print(f"Max epochs: {max_epochs}")
    if optimized_style_idx == 2:
        print(f"Alternating epochs: {alternating_epochs}")
    print("\nInitializing solver...")

    if contact_pos == "up":
        applied_z = 65.47258853
    elif contact_pos == "mid":
        applied_z = 45.20056126
    elif contact_pos == "down":
        applied_z = 24.92853398

    solver = DiffXPBDTapeFramework3D_Warp(
        trajectory_type=trajectory_type,
        optimize_type = optimize_type,
        mesh_path=mesh_file_path,
        real_target_kps_path=real_target_kp_path,
        sim_target_npz=sim_target_path,
        target_unit="mm",
        dt=dt,
        gravity_vec=(0.,  -0., 0.),
        mass_total=0.00729,
        poisson_ratio=0.48,
        compliance_modulation=1e-2,
        
        fixed_center=[(0.71246, 0, 4.03762)],
        fixed_rotation=[(0., 10.007068, 0.)],
        fixed_extents=[(20, 10, 5.5)],
        
        applied_center=[(18.85565662, 0, applied_z)],
        applied_extents=[(2.5, 10, 2)],
        constant_applied_force = constant_applied_force,
        show_force_arrow = True,
        series_force_path=real_applied_force_path,
        series_force_mode=series_force_mode_switch,
        

        camera_pos=(0., 180., 45.2006) if gripper_side == 'right' else (0., -180., 45.2006),
        camera_front=(0.0, -1.0, 0.0) if gripper_side == 'right' else (0.0, 1.0, 0.0),
        background_color=(0., 0., 0.),
        text_color = (0, 0, 0, 255) if gripper_side == 'right' else (255, 255, 255, 255),

        youngs_init = youngs_init,
        force_amplification = force_amplification,
        damping_amplification=damping_amplification, 
        sweep_count=sweep_count,

        position_effector_path = kp_init_path,

        device=DEVICE,)

    print("\nStarting training...\n")
    solver.train(project_name = f"{optimize_type}_{trajectory_type}_{contact_pos.capitalize()}_contact_Time_{int(total_time)}s_Max_Epochs_{int(max_epochs)}_Optimized_{optimized_str}",
                contact_pos = contact_pos,
                stop_condition = STOP_CONDITION[stop_condition_index],
                convergence_patience = convergence_patience,
                relative_change_threshold = relative_change_threshold,
                optimized_method = OPTIMIZED_METHOD[optimized_style_idx],
                alternating_epochs = alternating_epochs,
                max_epochs = max_epochs, 
                lr = optimized_lr, 
                eps = thresholding_eps, 
                optimize_subject = optimized_subjects,
                total_steps = total_steps,
                optimizer = OPTIMIZER[optimizer_idx])

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify training parameters for DiffXPBD model optimization")
    parser.add_argument("-t", "--simulate_time", type=float, default=100., help="Simulation total time (s)")
    parser.add_argument("-dt", "--simulate_time_step", type=float, default=1.e-2, help="Simulation Time step (s)")
    parser.add_argument("-optcombo", "--optimize_combo", type=str, default="sim2real", choices=["sim2sim", "sim2real"], help="Optimization combination: sim2sim or sim2real")

    parser.add_argument("-cnt", "--contact_pos", type=str, default="mid", choices=["up", "mid", "down"], help="Contact position on the gripper")

    parser.add_argument("-gs", "--gripper_side", type=str, default="left", choices=["left", "right"], help="Side of the gripper")

    parser.add_argument("-G", "--youngs_modulus", type=float, default=95. * 1e6, help="Initial Young's Modulus for the simulation")
    parser.add_argument("-famp", "--force_amplification", type=float, default=1.0, help="Initial Force amplification factor")
    parser.add_argument("-damp", "--damping_amplification", type=float, default=1.0, help="Initial Damping amplification factor")
    parser.add_argument("-sc", "--sweep_count", type=int, default=20, help="Number of sweeps for optimization")

    parser.add_argument("-e", "--max_epochs", type=int, default=20, help="Total epochs to train (not additional epochs).")
    parser.add_argument("-optidx", "--optimized_subject_indices", nargs='+', type=int, default=[0], help="Indices of subjects to optimize from the list: 0-Young's Modulus, 1-Applied Force, 2-Force Amplification, 3-Damping Amplification")
    parser.add_argument("-stpidx", "--stop_condition_index", type=int, default=2, help="Index of stop condition: 0-Convergence, 1-Thresholding, 2-Both")
    parser.add_argument("-cpts", "--convergence_patience", type=int, default=5, help="Number of epochs to wait for convergence")
    parser.add_argument("-rct", "--relative_change_threshold", type=float, default=1.0e-3, help="Relative change threshold for convergence")
    parser.add_argument("-optst", "--optimized_style_idx", type=int, default=0, help="Style of optimization for visualization: 0-None, 1-Simultaneous, 2-alternating")
    parser.add_argument("-altepochs", "--alternating_epochs", type=int, default=100, help="Number of epochs for each subject in alternating optimization")
    parser.add_argument("-lr", "--learning_rates", nargs='+', type=float, default=[2.e-1, 1.e-2], help="Learning rates for the optimized subjects (order should match optimized_subject_indices)")
    parser.add_argument("-tld_eps", '--thresholding_eps', type=float, default=1.0e0, help="Epsilon value for thresholding stop condition")
    parser.add_argument("-opteridx", "--optimizer_index", type=int, default=0, help="Index of the optimizer to use: 0-None, 1-Adam")

    args = parser.parse_args()

    main(total_time=args.simulate_time, 
         dt=args.simulate_time_step,
         optimize_type=args.optimize_combo,

         contact_pos=args.contact_pos,
         gripper_side=args.gripper_side,

         youngs_init=args.youngs_modulus,
         force_amplification=args.force_amplification,
         damping_amplification=args.damping_amplification,
         sweep_count=args.sweep_count,

         max_epochs=args.max_epochs,
         optimized_subject_indices=args.optimized_subject_indices,
         stop_condition_index=args.stop_condition_index,
         convergence_patience=args.convergence_patience,
         relative_change_threshold=args.relative_change_threshold,
         optimized_style_idx=args.optimized_style_idx,
         alternating_epochs=args.alternating_epochs,
         optimized_lr=args.learning_rates,
         thresholding_eps=args.thresholding_eps,
         optimizer_idx=args.optimizer_index)