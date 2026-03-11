import paramiko
import time

# --- CONFIGURATION ---
GNS3_VM_IP = "192.168.170.128"
GNS3_VM_USER = "gns3"
GNS3_VM_PASS = "gns3"

# Exact names from your 'docker ps' output
# Note: Leaf-1 is the 'Source' router where we want to see the traffic split
LEAF_1_CONTAINER = "GNS3.Leaf-1.2322fc6b-91e6-4c55-8f33-0da8cb3c817f"

def get_tx_bytes(ssh_client, container_name, interface):
    """Reaches into the docker container and grabs total bytes sent on a specific port."""
    cmd = f"docker exec {container_name} cat /sys/class/net/{interface}/statistics/tx_bytes"
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    output = stdout.read().decode().strip()
    return int(output) if output.isdigit() else 0

# --- MAIN LOOP ---
try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(GNS3_VM_IP, username=GNS3_VM_USER, password=GNS3_VM_PASS)
    
    print(f"Monitoring Traffic Split on Leaf-1...")
    print(f"Path 1 (via Spine-1) | Path 2 (via Spine-2)")
    print("-" * 45)

    # Initial Baseline
    # eth1 is Leaf-1 -> Spine-1
    # eth2 is Leaf-1 -> Spine-2
    prev_eth1 = get_tx_bytes(client, LEAF_1_CONTAINER, "eth1")
    prev_eth2 = get_tx_bytes(client, LEAF_1_CONTAINER, "eth2")
    prev_time = time.time()

    while True:
        time.sleep(1)
        
        curr_eth1 = get_tx_bytes(client, LEAF_1_CONTAINER, "eth1")
        curr_eth2 = get_tx_bytes(client, LEAF_1_CONTAINER, "eth2")
        curr_time = time.time()
        
        elapsed = curr_time - prev_time
        
        # Mbps calculation
        mbps1 = ((curr_eth1 - prev_eth1) * 8) / (1000000 * elapsed)
        mbps2 = ((curr_eth2 - prev_eth2) * 8) / (1000000 * elapsed)
        
        print(f"Port eth1: {mbps1:6.2f} Mbps | Port eth2: {mbps2:6.2f} Mbps")
        
        prev_eth1, prev_eth2, prev_time = curr_eth1, curr_eth2, curr_time

except KeyboardInterrupt:
    print("\nStopping.")
    client.close()