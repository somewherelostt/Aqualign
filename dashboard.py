import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.ocean_field import OceanField
from src.particle_simulator import ParticleSimulator
from src.optimizer import RouteOptimizer
import os

# Page Config
st.set_page_config(page_title="Aqualign Dashboard", page_icon="🌊", layout="wide")

st.title("🌊 Aqualign: Differentiable Ocean Cleanup")
st.markdown("""
**Interactive Simulation**: Compare Random Patrol vs. Gradient-Optimized Cleanup.
Adjust parameters in the sidebar and click **Run Simulation**.
""")

# Sidebar Controls
st.sidebar.header("Simulation Settings")

num_vessels = st.sidebar.slider("Number of Vessels", 1, 5, 3)
time_horizon = st.sidebar.slider("Time Horizon (Hours)", 24, 96, 72)
opt_iterations = st.sidebar.slider("Optimization Iterations", 50, 500, 200)
learning_rate = st.sidebar.number_input("Learning Rate", 0.01, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Higher iterations yield better routes but take longer to compute.")

# Load Data
@st.cache_resource
def load_field():
    if not os.path.exists("data/gulf_stream.npz"):
        st.error("Data file not found! Run `python data/generate_data.py` first.")
        return None
    return OceanField(device='cpu')

field = load_field()

if field:
    # Run Simulation
    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Simulating {num_vessels} vessels for {time_horizon} hours..."):
            
            device = 'cpu'
            dt = 1.0
            steps = time_horizon
            
            # --- Setup ---
            # Initial Vessel Positions (Left side)
            initial_vessels = torch.tensor([
                [10.0, 10.0],
                [10.0, 25.0],
                [10.0, 40.0],
                [10.0, 15.0],
                [10.0, 35.0]
            ][:num_vessels], device=device)
            
            # Debris Field (Random cluster)
            torch.manual_seed(42) # Fixed seed for consistency
            debris_center = torch.tensor([25.0, 25.0], device=device)
            debris_pos = debris_center + torch.randn(200, 2, device=device) * 15.0
            
            # --- 1. Random Patrol ---
            random_controls = torch.randn(steps, num_vessels, 2, device=device)
            v_pos = initial_vessels.clone()
            d_pos = debris_pos.clone()
            random_traj = [v_pos.numpy().copy()]
            
            random_collected_mask = torch.zeros(len(d_pos), dtype=torch.bool, device=device)
            CAPTURE_RADIUS = 1.0
            
            for t in range(steps):
                d_pos = ParticleSimulator.advect_particles(d_pos, field.get_velocity, dt)
                v_current = field.get_velocity(v_pos)
                thrust = 1.0 * torch.tanh(random_controls[t])
                v_pos = ParticleSimulator.step_vessels(v_pos, v_current, thrust, dt)
                random_traj.append(v_pos.numpy().copy())
                
                # Collection
                diff = v_pos.unsqueeze(1) - d_pos.unsqueeze(0)
                dists = torch.norm(diff, dim=2)
                min_dists, _ = dists.min(dim=0)
                random_collected_mask |= (min_dists < CAPTURE_RADIUS)
                
            random_count = random_collected_mask.sum().item()
            random_fuel = torch.sum(random_controls**2).item()
            
            # --- 2. Optimized Strategy ---
            optimizer = RouteOptimizer(field, num_vessels, steps, dt)
            status_text = st.empty()
            status_text.text("Optimizing routes...")
            
            # We can't easily capture the print output from tqdm here, so we just run it
            optimizer.run_optimization(initial_vessels, debris_pos, iterations=opt_iterations, lr=learning_rate)
            status_text.text("Optimization complete!")
            
            final_controls = optimizer.controls.detach()
            v_pos = initial_vessels.clone()
            d_pos = debris_pos.clone()
            opt_traj = [v_pos.numpy().copy()]
            
            opt_collected_mask = torch.zeros(len(d_pos), dtype=torch.bool, device=device)
            
            for t in range(steps):
                d_pos = ParticleSimulator.advect_particles(d_pos, field.get_velocity, dt)
                v_current = field.get_velocity(v_pos)
                thrust = 1.5 * torch.tanh(final_controls[t]) 
                v_pos = ParticleSimulator.step_vessels(v_pos, v_current, thrust, dt)
                opt_traj.append(v_pos.numpy().copy())
                
                diff = v_pos.unsqueeze(1) - d_pos.unsqueeze(0)
                dists = torch.norm(diff, dim=2)
                min_dists, _ = dists.min(dim=0)
                opt_collected_mask |= (min_dists < CAPTURE_RADIUS)
                
            opt_count = opt_collected_mask.sum().item()
            opt_fuel = torch.sum(final_controls**2).item()
            
            # --- Visualization ---
            col1, col2 = st.columns(2)
            
            # Metrics
            with col1:
                st.subheader("Random Patrol 🐢")
                st.metric("Debris Collected", f"{random_count} / 200", delta=None)
                st.metric("Fuel Usage", f"{random_fuel:.1f}", delta=None)
                
            with col2:
                st.subheader("Aqualign AI ⚡")
                st.metric("Debris Collected", f"{opt_count} / 200", delta=f"{opt_count - random_count}")
                st.metric("Fuel Usage", f"{opt_fuel:.1f}", delta=f"{random_fuel - opt_fuel:.1f}", delta_color="inverse")
            
            # Plot
            st.subheader("Trajectory Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Background Flow
            x = np.linspace(0, 50, 20)
            y = np.linspace(0, 50, 20)
            X, Y = np.meshgrid(x, y)
            grid_pos = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(-1, 2)
            vel = field.get_velocity(grid_pos).detach().numpy()
            U = vel[:, 0].reshape(20, 20)
            V = vel[:, 1].reshape(20, 20)
            
            ax.quiver(X, Y, U, V, color='gray', alpha=0.3)
            
            # Final Debris (Snapshot at T_final is tricky with static plot, showing final pos)
            # Actually, showing paths is better
            
            random_traj = np.array(random_traj) # (T, N, 2)
            opt_traj = np.array(opt_traj)
            
            # Plot Random Paths
            for i in range(num_vessels):
                ax.plot(random_traj[:, i, 0], random_traj[:, i, 1], 'r--', alpha=0.5, label='Random' if i==0 else "")
                
            # Plot Optimized Paths
            for i in range(num_vessels):
                ax.plot(opt_traj[:, i, 0], opt_traj[:, i, 1], 'b-', linewidth=2, label='Aqualign' if i==0 else "")
                
            # Plot Debris (Final positions)
            ax.scatter(d_pos[:, 0].numpy(), d_pos[:, 1].numpy(), c='k', s=5, alpha=0.3, label='Debris (Final)')
            
            ax.set_title(f"Optimized vs Random Trajectories ({time_horizon}h)")
            ax.legend()
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
            
            st.pyplot(fig)
