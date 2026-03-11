import paramiko
import time
import logging
import csv
from datetime import datetime

# --- CONFIGURATION ---
GNS3_VM_IP = "192.168.170.128"
GNS3_VM_USER = "gns3"
GNS3_VM_PASS = "gns3"

# Exact container name from your docker ps output
LEAF_1_CONTAINER = "GNS3.Leaf-1.2322fc6b-91e6-4c55-8f33-0da8cb3c817f"

# Thresholds
TRAFFIC_THRESHOLD_MBPS = 700.0
COOLDOWN_PERIOD = 5  # Seconds to wait before allowing another switch to prevent flapping

# Logging Setup
log_filename = f"network_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
csv_filename = f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def get_tx_bytes(ssh_client, container_name, interface):
    """Fetch raw bytes from the Linux kernel via Docker."""
    cmd = f"docker exec {container_name} cat /sys/class/net/{interface}/statistics/tx_bytes"
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    output = stdout.read().decode().strip()
    return int(output) if output.isdigit() else 0

def set_path_preference(ssh_client, target_path):
    """
    Action: Dynamically updates BGP weights on Leaf-1.
    target_path 1 -> Prefer Spine-1 (eth1)
    target_path 2 -> Prefer Spine-2 (eth2)
    """
    w1 = 500 if target_path == 1 else 100
    w2 = 500 if target_path == 2 else 100
    
    # We build a multi-line vtysh command
    vtysh_cmd = (
        f"vtysh -c 'conf t' "
        f"-c 'router bgp 65101' "
        f"-c 'neighbor 10.1.1.2 weight {w1}' "
        f"-c 'neighbor 10.1.2.2 weight {w2}' "
        f"-c 'end'"
    )
    
    ssh_client.exec_command(vtysh_cmd)
    logging.info(f"!!! ACTION TRIGGERED: Routing preference moved to SPINE-{target_path} (W1:{w1}, W2:{w2})")

# --- MAIN ENGINE ---
try:
    # 1. Connect to GNS3 VM
    vm_client = paramiko.SSHClient()
    vm_transport = vm_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    vm_client.connect(GNS3_VM_IP, username=GNS3_VM_USER, password=GNS3_VM_PASS)
    logging.info("Connected to GNS3 Control Plane. Starting Adaptive Logic...")

    # 2. Setup CSV logging
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Eth1_Mbps", "Eth2_Mbps", "Current_Path"])

        # Initial State
        prev_eth1 = get_tx_bytes(vm_client, LEAF_1_CONTAINER, "eth1")
        prev_eth2 = get_tx_bytes(vm_client, LEAF_1_CONTAINER, "eth2")
        prev_time = time.time()
        
        current_active_path = 1 # Start assuming Path 1 (ECMP default)
        last_switch_time = 0

        while True:
            time.sleep(1)
            
            # A. Telemetry Gathering
            curr_eth1 = get_tx_bytes(vm_client, LEAF_1_CONTAINER, "eth1")
            curr_eth2 = get_tx_bytes(vm_client, LEAF_1_CONTAINER, "eth2")
            curr_time = time.time()
            
            elapsed = curr_time - prev_time
            mbps1 = ((curr_eth1 - prev_eth1) * 8) / (1000000 * elapsed)
            mbps2 = ((curr_eth2 - prev_eth2) * 8) / (1000000 * elapsed)
            
            # B. Log to Console and File
            status_msg = f"S1 (eth1): {mbps1:7.2f} Mbps | S2 (eth2): {mbps2:7.2f} Mbps | Active: {current_active_path}"
            print(status_msg)
            writer.writerow([datetime.now().isoformat(), mbps1, mbps2, current_active_path])
            f.flush() # Force write to disk

            # C. Heuristic Control Logic (The "Brain")
            now = time.time()
            if (now - last_switch_time) > COOLDOWN_PERIOD:
                
                # Rule: If Path 1 is congested (>700) and Path 2 is empty (<50)
                if mbps1 > TRAFFIC_THRESHOLD_MBPS and mbps2 < 50 and current_active_path == 1:
                    logging.warning(f"Congestion detected on Spine-1 ({mbps1:.2f} Mbps). Switching to Spine-2.")
                    set_path_preference(vm_client, 2)
                    current_active_path = 2
                    last_switch_time = now

                # Rule: If Path 2 is congested (>700) and Path 1 is empty (<50)
                elif mbps2 > TRAFFIC_THRESHOLD_MBPS and mbps1 < 50 and current_active_path == 2:
                    logging.warning(f"Congestion detected on Spine-2 ({mbps2:.2f} Mbps). Switching to Spine-1.")
                    set_path_preference(vm_client, 1)
                    current_active_path = 1
                    last_switch_time = now

            # Update baselines
            prev_eth1, prev_eth2, prev_time = curr_eth1, curr_eth2, curr_time

except KeyboardInterrupt:
    logging.info("Manual stop requested. Exiting.")
except Exception as e:
    logging.error(f"System Error: {e}")
finally:
    vm_client.close()