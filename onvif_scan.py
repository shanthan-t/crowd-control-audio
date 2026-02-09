import socket
import uuid

def ws_discovery():
    # WS-Discovery Probe Message
    msg = \
    '<?xml version="1.0" encoding="UTF-8"?>' \
    '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" ' \
    'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" ' \
    'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" ' \
    'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">' \
    '<e:Header>' \
    '<w:MessageID>uuid:{0}</w:MessageID>' \
    '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>' \
    '<w:Action a:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>' \
    '</e:Header>' \
    '<e:Body>' \
    '<d:Probe>' \
    '<d:Types>dn:NetworkVideoTransmitter</d:Types>' \
    '</d:Probe>' \
    '</e:Body>' \
    '</e:Envelope>'.format(uuid.uuid4())

    # Multicast Address for WS-Discovery
    MCAST_GRP = '239.255.255.250'
    MCAST_PORT = 3702

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.settimeout(3)
    
    print(f"Sending ONVIF Probe to {MCAST_GRP}:{MCAST_PORT}...")
    try:
        s.sendto(msg.encode(), (MCAST_GRP, MCAST_PORT))
        
        while True:
            try:
                data, addr = s.recvfrom(65535)
                print(f"[FOUND] Device at {addr[0]}")
                # Simple extraction of XAddrs (URLs)
                decoded = data.decode(errors='ignore')
                if "XAddrs" in decoded:
                    start = decoded.find("http://")
                    end = decoded.find(" ", start) if " " in decoded[start:] else decoded.find("<", start)
                    print(f"    Service URL: {decoded[start:end]}")
            except socket.timeout:
                print("Scan complete.")
                break
            except Exception as e:
                print(f"Error reading: {e}")
                break
    except Exception as e:
        print(f"Socket error: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    ws_discovery()
