import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.ocean_field import OceanField
from src.particle_simulator import ParticleSimulator
from src.optimizer import RouteOptimizer
import os

# --- Visual Styling ---
st.set_page_config(
    page_title="Aqualign | Ocean Cleanup",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Pro" feel
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to bottom right, #000428, #004e92);
        color: white;
    }
    /* Headers */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(0deg, #00f260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #e0e0e0 !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 20, 30, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/tsunami.png", width=60)
    st.title("Aqualign")
    st.markdown("_Differentiable Ocean Cleanup_")
    
    st.markdown("---")
    
    st.subheader("⚙️ Simulation Settings")
    num_vessels = st.slider("Fleet Size", 1, 5, 3, help="Number of cleanup vessels to deploy.")
    time_horizon = st.slider("Time Horizon (Hours)", 24, 96, 72, help="Simulation duration.")
    
    with st.expander("Advanced Parameters"):
        opt_iterations = st.slider("Iterations", 50, 500, 200)
        learning_rate = st.number_input("Learning Rate", 0.01, 1.0, 0.1)
        st.caption("Adjust gradient descent hyperparameters.")

    st.markdown("---")
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        run_sim = True
    else:
        run_sim = False
        
    st.markdown("### About")
    st.info(
        "Aqualign uses **differentiable physics** to optimize vessel routes, "
        "surfing ocean currents to maximize debris collection."
    )

# --- Main Layout ---
col_hero, col_logo = st.columns([4, 1])
with col_hero:
    st.title("Order from Chaos")
    st.markdown("### Autonomous Optimization of Cleanup Trajectories")

# --- Logic ---
@st.cache_resource
def load_field():
    if not os.path.exists("data/gulf_stream.npz"):
        st.error("Data file `data/gulf_stream.npz` not found! Generate it first.")
        return None
    return OceanField(device='cpu')

field = load_field()

if run_sim and field:
    with st.status("Running Differentiable Simulation...", expanded=True) as status:
        st.write("🌊 Loading Ocean Vector Field...")
        device = 'cpu'
        dt = 1.0
        steps = time_horizon
        
        # Initial State
        initial_vessels = torch.tensor([[10.0, 10.0 + i*5.0] for i in range(num_vessels)], device=device)
        
        torch.manual_seed(42)
        debris_center = torch.tensor([25.0, 25.0], device=device)
        debris_pos = debris_center + torch.randn(200, 2, device=device) * 15.0
        
        st.write("🐢 Simulating Random Patrol Baseline...")
        # Random Patrol
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
            
            diff = v_pos.unsqueeze(1) - d_pos.unsqueeze(0)
            dists = torch.norm(diff, dim=2)
            min_dists, _ = dists.min(dim=0)
            random_collected_mask |= (min_dists < CAPTURE_RADIUS)
            
        random_count = random_collected_mask.sum().item()
        random_fuel = torch.sum(random_controls**2).item()
        
        st.write("⚡ Optimizing Aqualign Trajectories...")
        # Deep Optimization
        optimizer = RouteOptimizer(field, num_vessels, steps, dt)
        optimizer.run_optimization(initial_vessels, debris_pos, iterations=opt_iterations, lr=learning_rate)
        
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
        
        status.update(label="Simulation Complete!", state="complete", expanded=False)

    # --- Results Dashboard ---
    st.markdown("---")
    
    # 1. KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("Random Debris", f"{random_count}", help="Debris collected by random patrol")
    with kpi2:
        st.metric("Aqualign Debris", f"{opt_count}", delta=f"{opt_count - random_count}", help="Debris collected by Aqualign")
    with kpi3:
        st.metric("Aqualign Fuel", f"{opt_fuel:.0f}", delta=f"{random_fuel - opt_fuel:.0f}", delta_color="inverse", help="Fuel units consumed")
    with kpi4:
         efficiency_gain = ((opt_count / max(1, random_count)) - 1) * 100
         st.metric("Efficiency Gain", f"+{efficiency_gain:.1f}%", delta="High Impact")

    # 2. Main Visualization
    st.subheader(f"Trajectory Analysis (T={time_horizon}h)")
    
    # Dark Theme Plot with Neon colors
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0e1117') # Match Streamlit dark theme if default
    ax.set_facecolor('#0e1117')
    
    # Ocean Field (Subtle)
    x = np.linspace(0, 50, 25)
    y = np.linspace(0, 50, 25)
    X, Y = np.meshgrid(x, y)
    grid_pos = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(-1, 2)
    vel = field.get_velocity(grid_pos).detach().numpy()
    U = vel[:, 0].reshape(25, 25)
    V = vel[:, 1].reshape(25, 25)
    
    ax.streamplot(X, Y, U, V, color='#ffffff26', linewidth=0.5, density=0.8, arrowsize=0.5)
    
    # Trajectories
    random_traj = np.array(random_traj)
    opt_traj = np.array(opt_traj)
    
    # We plot the full path
    for i in range(num_vessels):
        # Random - Dashed Red/Orange
        ax.plot(random_traj[:, i, 0], random_traj[:, i, 1], color='#ff4b4b', linestyle='--', linewidth=1.5, alpha=0.6, label='Random Patrol' if i==0 else "")
        # Aqualign - Glowing Cyan
        ax.plot(opt_traj[:, i, 0], opt_traj[:, i, 1], color='#00e6e6', linewidth=2.5, alpha=0.9, label='Aqualign AI' if i==0 else "")
        # Start/End points
        ax.scatter(opt_traj[-1, i, 0], opt_traj[-1, i, 1], c='#00e6e6', s=40, zorder=5)

    # Debris (Scatter)
    ax.scatter(d_pos[:, 0].numpy(), d_pos[:, 1].numpy(), c='white', s=5, alpha=0.4, label='Debris')
    
    # Styling
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    ax.grid(False) # Clean look
    ax.axis('off') # Map style
    
    # Custom Legend
    leg = ax.legend(loc='upper right', facecolor='#1c2128', edgecolor='#30363d', fontsize='medium')
    for text in leg.get_texts():
        text.set_color("white")
        
    st.pyplot(fig, use_container_width=True)

else:
    # Empty State / Landing
    st.info("👈 Adjust settings in the sidebar and click **Run Simulation** to start.")
    
    # Placeholder infographic
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0.0)
    ax.axis('off')
    ax.text(0.5, 0.5, "Configure & Run", ha='center', va='center', color='white', fontsize=20, alpha=0.3)
    st.pyplot(fig)
