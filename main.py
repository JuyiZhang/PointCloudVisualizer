
import Debug
from multiprocessing import Process, Manager
from multiprocessing.managers import DictProxy
import time
import threading



if __name__ == "__main__":
    import TCPServer
    print("\n")
    Debug.Log("Session initiated", True)
    Debug.Log("Session begins at: " + time.strftime("%Y/%m/%d %H:%M:%S, UTC %z"), True)
    Debug.Log("Debug mode is now set to " + str(Debug.debug_mode), True)
    Debug.Log("Begin Debugging...")
    print("\n")
    Debug.Log("Starting MuTA Backend...")
    session_timestamp = time.time()
    session_folder = "data/Session_" + str(int(session_timestamp))
    manager = Manager()
    device_timestamp_dict = manager.dict()
    Process(target=TCPServer.listen_for_connection_async, args=(session_folder, device_timestamp_dict)).start()
    
    import app
    import SessionManager
    app.session = SessionManager.Session(session_timestamp)
    device_timestamp_dict_global = {}
    def listen_for_new_frame(device_timestamp_dict: DictProxy, session):
        while True:
            device_addresses = device_timestamp_dict.keys()
            for i in range(0,len(device_addresses)):
                dev_addr = device_addresses[i]
                if not(dev_addr in device_timestamp_dict_global) or device_timestamp_dict[dev_addr] != device_timestamp_dict_global[dev_addr]:
                        device_timestamp_dict_global[dev_addr] = device_timestamp_dict[dev_addr]
                        app.session.new_frame(device_timestamp_dict[dev_addr],dev_addr)
    threading.Thread(target=listen_for_new_frame, args=(device_timestamp_dict,app.session)).start()
    
    app.app.run()