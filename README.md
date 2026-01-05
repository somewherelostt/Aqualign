# 🌊 Aqualign: Order from Chaos

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)

> *“We don’t fight the ocean; we learn to dance with it.”*

---

## 🌍 The Story

### The Problem: A Silent Catastrophe
Every single year, **8 million tons** of plastic vanish into our oceans. It doesn't just sit there; it migrates. Carried by chaotic, swirling currents—eddies, gyres, and jets—this debris becomes a moving target that is near-impossible to track.

Traditional cleanup methods are **reactive**. They rely on brute force: patrolling random sectors or using static predictions that fail the moment the wind shifts. It’s like trying to catch a feather in a tornado with a pair of tweezers. We are burning fuel to chase ghosts.

### The Solution: Differentiable Physics
**Aqualign** changes the rules of engagement. We stop treating the ocean as a black box and start treating it as a **differentiable function**.

By building a fully differentiable simulation of ocean dynamics, we allow our cleanup vessels to "look into the future." They don't just react to currents; they exploit them. Through gradient-based optimization, Aqualign discovers non-intuitive trajectories—vessels that **surf** the currents to intercept debris clusters before they disperse.

This isn't just prediction. It's **control**.

---

## ⚙️ The Tech

At the heart of Aqualign lies a seamless differentiable pipeline powered by **PyTorch**.

1.  **🌊 Differentiable Ocean (`src/ocean_field.py`)**:
    We model the ocean not as a static map, but as a continuous, differentiable vector field. Using bilinear interpolation on real or synthetic hydrographic data, we ensure that every point in the ocean allows for gradient flow.

2.  **🚀 Physics Engine (`src/particle_simulator.py`)**:
    We implemented a differentiable **Runge-Kutta 4 (RK4)** integrator. This allows us to simulate particle advection and vessel kinematics while maintaining the entire computational graph. We can backpropagate *through time*, calculating exactly how a small change in a vessel's thrust *now* will affect its distance to a piece of plastic *10 hours later*.

3.  **🧠 Gradient Descent Control (`src/optimizer.py`)**:
    Instead of reinforcement learning (which struggles with sample efficiency), we use direct gradient optimization. The loss function is a delicate balance:
    $$ \mathcal{L} = -\text{DebrisCollected} + \lambda \cdot \text{FuelConsumed} $$
    The vessel "learns" to minimize this loss, naturally discovering energy-efficient paths that maximize impact.

---

## 🏔️ The Challenges

Building Aqualign was a journey into the mathematical deep end.

-   **Chaos & Gradients**: Ocean dynamics are famously chaotic (butterfly effect). Backpropagating through 72 hours of fluid dynamics often led to exploding or vanishing gradients. We had to carefully tune our time steps and integrator stability to keep the "gradient signal" clean.

-   **Sparse Rewards**: The ocean is vast, and debris is tiny. A vessel might travel for hours without seeing a single piece of plastic. To fix this, we implemented **soft capture functions** (Gaussian kernels) that provide smooth, continuous feedback, guiding the vessel toward high-probability zones even when it gathers nothing.

-   **The Energy Trade-off**: Aggressive cleanup burns fuel. We had to find the "Goldilocks" penalty for fuel consumption ($\lambda$)—too high, and the vessel stays at port; too low, and it wastes energy chasing outliers.

---

## 📊 The Impact

We benchmarked Aqualign against standard industry patrol strategies on the synthetic "Double Gyre" dataset. The results speak for themselves:

| Strategy | Debris Collected (200 total) | Fuel Usage (Units) | Efficiency Rating |
| :--- | :---: | :---: | :---: |
| 🐢 **Random Patrol** | 6 | 450.4 | Low |
| ⚡ **Aqualign** | **9** | **396.7** | **High** |

**+50% Efficiency**. **-12% Carbon Footprint**.
By working *with* the ocean, we achieve more with less.

---

## 🚀 Get Started

Experience the optimization yourself.

### 1. Installation
Cloning the repo and setting up the environment takes seconds.

```bash
# Clone the repository
git clone https://github.com/somewherelostt/Aqualign.git
cd Aqualign

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate the Ocean
Create a synthetic ocean environment (Double Gyre model).
```bash
python data/generate_data.py
```

### 3. Run the Simulation
Watch Aqualign optimize the fleet in real-time.
```bash
python main.py
```
*Results will be saved to `visualizations/comparison.png`*

---

## 🤝 Acknowledgements
Built with ❤️ for the **Tesseract Hackathon**. Inspired by the potential of Simulation Intelligence to heal our planet. Special thanks to the open-source community behind PyTorch and SciPy.
