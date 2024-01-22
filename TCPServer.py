import socket
import struct
import traceback
import threading
import os

import Debug

from NetworkManager import NetworkUtils
from SessionManager import Session


class TCPServer:
    
    connection_pool = []
    
    def __init__(self, session: Session, server_host = '', server_port = 9090) -> None:
        self.session = session
        self.network_utils = NetworkUtils()
        self.network_utils.add_host(socket.gethostbyname(socket.gethostname()))
        self.socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket_connection.bind((server_host, server_port))
            Debug.Log('Server bind to port ' + str(server_port))
        except socket.error as msg:
            Debug.Log('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            return
        self.socket_connection.listen(10)
        Debug.Log('Start listening...')
        self.socket_connection.settimeout(3.0)
        threading.Thread(target=self.listen_for_connection, daemon=True).start()
    
    def close_connection(self):
        self.socket_connection.close()
        self.network_utils.del_host()
        while (self.connection_pool.count != 0):
            self.connection_pool.pop()
    
    def get_connection_list(self):
        return self.connection_pool
    
    def listen_for_connection(self):
        while True:
            try:
                conn, addr = self.socket_connection.accept() # Blocking, wait for incoming connection
                threading.Thread(target=self.listen_for_data, daemon=True, args=(conn, addr)).start()
                self.connection_pool.append(addr[0])
            except Exception:
                continue
            
    def listen_for_data(self, conn, addr):
        Debug.Log('Connected with ' + addr[0] + ':' + str(addr[1]))
        if not(os.path.exists(self.session.session_folder + "/" + addr[0])):
            os.mkdir(self.session.session_folder + "/" + addr[0])
        
        final_data = bytearray(0)
        current_header = ""
        current_data_length = 0

        while True:
            # Receiving and transmitting from client
            try:
                data = conn.recv(512*512*4)
                
                #while len(self.transmit_queue) > 0:
                #    conn.send(self.transmit_queue.pop(0))
                
                if len(data)==0:
                    continue
                if current_header != "": # Means there are data that is not yet received
                    final_data += bytearray(data)
                    if current_data_length + 1 > len(final_data):
                        continue
                    else:
                        if current_header == "c":
                            current_header = ""
                            process_data = final_data if len(final_data) <= current_data_length + 1 else final_data[0:current_data_length]
                            self.session.new_frame(process_data, addr[0])
                            current_data_length = 0
                            final_data.clear()
                            continue
                        
                try:
                    header = data[0:1].decode('utf-8')
                    Debug.Log('-------------------------- Header: ' + header + ' --------------------------', category="Conn")
                except:
                    Debug.Log('Header is not UTF8', category="Conn")
                    continue

                if header == 'c':
                    current_data_length = 0
                    final_data.clear()
                    # Retrieve Depth Sensor Data
                    depth_length = struct.unpack(">i", data[9:13])[0]
                    ab_length = struct.unpack(">i", data[13:17])[0]
                    pointcloud_length = struct.unpack(">i", data[17:21])[0]
                    if (depth_length != 184320 or ab_length != 92160 or pointcloud_length <= 0):
                        continue
                    final_data += bytearray(data)
                    current_header = 'c'
                    current_data_length = depth_length + ab_length + pointcloud_length + 21
                    Debug.Log("Length of expected data " + str(current_data_length), category="Conn")
                    if len(data) < current_data_length: # Insufficient data, continue receive
                        continue
                    else:
                        current_header = ""
                        self.session.new_frame(data, addr[0])
                        current_data_length = 0
                        final_data.clear()
            except ConnectionResetError:
                self.connection_pool.remove(addr[0])
                traceback.print_exc()
                return
            except:
                traceback.print_exc()
                continue
                