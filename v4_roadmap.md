# V3 Architecture Blueprint: The Adaptive Conductor
**Concept:** True AI fabric optimization requires controlling *both* spatial distribution (where traffic goes) and temporal injection (when traffic enters the fabric). 

### 1. Upgrading the Simulator Physics (The Edge Buffer)
To mimic modern Priority Flow Control (PFC) and prevent TCP tail-drops, we must hold data at the edge of the network until it is safe to transmit.

*   **The Edge Queue:** Every Leaf switch gets a new attribute: `leaf_backlog` (a queue storing the data of all flows originating from that rack).
*   **Flow Initialization:** When the `TrafficGenerator` creates a 75GB Elephant flow, it does *not* instantly hit the wire. It goes into the `leaf_backlog`.
*   **The Transmission Gate:** Data is only drained from the `leaf_backlog` into the network based on an **Admitted Rate** (set by the agent). 
*   **TCP Interaction:** TCP still runs. A flow's actual sending rate becomes: `min(TCP_cwnd, Agent_Admitted_Rate, Flow_Hardware_Bandwidth)`.

### 2. Upgrading the RL Interface (Gym Wrapper)
We must give the RL agent the tools to act as a Toll Booth Operator.

*   **The New Observation Space (Input):**
    *   The GNN must know how much traffic is waiting. Add `normalized_leaf_backlog` (e.g., `current_backlog / max_buffer_capacity`) to the **Node Features** of every Leaf.
*   **The New Action Space (Output):**
    *   Previously: `[Spine1_Weight, Spine2_Weight, Spine3_Weight, Spine4_Weight]` (Size: 4 per leaf).
    *   Now: `[Admission_Rate, Spine1_Ratio, Spine2_Ratio, Spine3_Ratio, Spine4_Ratio]` (Size: 5 per leaf).
    *   `Admission_Rate` is a float `[0.0, 1.0]`. If it outputs `0.5`, the Leaf only injects data at 50% of its uplink capacity for that step.
*   **The New Reward Function (Pure FCT):**
    *   *Delete* throughput, fairness, hotspots, and drop penalties. 
    *   **New Reward:** `Reward = - (Number_of_Active_Unfinished_Flows)`
    *   *Why this is genius:* Every simulation step a flow sits in the backlog OR traverses the network, it costs the agent -1 point. To maximize its score, the agent *must* finish flows as fast as physically possible. If it admits too much traffic $\rightarrow$ TCP drops $\rightarrow$ flows slow down $\rightarrow$ massive negative reward. If it admits too little $\rightarrow$ flows sit in the backlog $\rightarrow$ massive negative reward. It forces the agent to find the perfect equilibrium.

### 3. The Validation Pipeline (How to build it safely)

Do not jump straight to the GNN. Build it in these verification stages:

**Stage A: The Sanity Check (The "Dumb" Agent)**
*   Write a dummy agent that always outputs `Admission_Rate = 1.0` and equal Spine ratios.
*   *Validation:* The simulator should behave exactly as it did in V2 (TCP collapse, high drops), proving your underlying TCP math is still intact.

**Stage B: The Heuristic Conductor (Proving the Physics)**
*   Write a simple Python threshold rule: *"If any Spine is > 90% utilized, set Admission_Rate = 0.5. Else, set Admission_Rate = 1.0. Route to the Least Loaded Spine."*
*   *Validation:* Run this against ECMP. You should immediately see **Zero Drops**, **Zero Retransmissions**, and a **Lower Tail FCT** than ECMP. This mathematically proves that Admission Control works.

**Stage C: The GNN-PPO Orchestrator (The Final Brain)**
*   Update the PyTorch Geometric policy to output the extra `Admission_Rate` head.
*   Train the model using the new `-num_active_flows` reward.
*   *Validation:* The GNN should learn to dynamically throttle Leaves that are participating in massive bursts, while perfectly spraying their flowlets across the Spines, achieving the absolute theoretical minimum FCT.
