import time

debug_mode = True
filter_category = ["Conn", "Frame", "Detection Miscellaneous"]
fp = open("log/Session_"+str(int(time.time()))+".log","a")

def Log(log_content, force_print = False, category = "0"):
    if (debug_mode or force_print):
        if not(category in filter_category):
            print("[" + time.strftime("%H:%M:%S") + "] {" + category + "} " + log_content)
            fp.write("[" + time.strftime("%H:%M:%S") + "] {" + category + "} " + log_content + "\n")
            
        
