import time

debug_mode = True
filter_category = ["Frame", "Conn", "Detection Miscellaneous"]
fp = open("log/Session_"+str(int(time.time()))+".log","a")

def Log(log_content, force_print = False, category = "0"):
    if (debug_mode or force_print):
        if not(category in filter_category):
            print("[" + time.strftime("%H:%M:%S") + "] {" + category + "} " + log_content)
            fp.write("[" + time.strftime("%H:%M:%S") + "] {" + category + "} " + log_content + "\n")
            
        
print("\n")
Log("Session initiated", True)
Log("Session begins at: " + time.strftime("%Y/%m/%d %H:%M:%S, UTC %z"), True)
Log("Debug mode is now set to " + str(debug_mode), True)
Log("Begin Debugging...")
print("\n")