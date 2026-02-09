import socket
import concurrent.futures
import ipaddress
import subprocess

# get local ip
def get_local_ip():
    try:
        # Hack to get local IP connecting to internet
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def scan_host(ip, ports=[80, 554, 37777, 8080]):
    found = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1) # Fast timeout
        result = sock.connect_ex((str(ip), port))
        if result == 0:
            found.append(port)
        sock.close()
    return str(ip), found

def scan_network():
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")
    
    # Assume /24 subnet
    network_prefix = ".".join(local_ip.split(".")[:3])
    ips = [f"{network_prefix}.{i}" for i in range(1, 255)]
    
    print(f"Scanning {network_prefix}.1 - 254 for CP Plus Cameras (Ports 554, 37777)...")
    
    found_devices = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(scan_host, ips)
        
    for ip, ports in results:
        if ports:
            print(f"[FOUND] Device at {ip} | Open Ports: {ports}")
            found_devices.append((ip, ports))
            
    if not found_devices:
        print("No devices found. Are you sure the camera is on this WiFi?")
    else:
        print("\n--- Potential Cameras ---")
        for ip, ports in found_devices:
            if 554 in ports or 37777 in ports:
                 print(f"--> RECOMMENDED: {ip} (Has RTSP/CP-Plus Port)")
            elif 8080 in ports:
                 print(f"--> Phone? {ip} (Has Port 8080)")

if __name__ == "__main__":
    scan_network()
