import requests

url = "https://mutaw.azurewebsites.net/"

def add_host(hostip):
    
    print("Adding Host IP: " + hostip)
    request_url = url + "add_host"
    request_content = {
        "op": "addm", # Add or Modify Host
        "hostname": hostip
    }
    return requests.post(request_url,json=request_content)

def del_host():
    
    request_url = url + "add_host"
    request_content = {
        "op": "delete", # Delete Host
    }
    return requests.post(request_url, json=request_content)