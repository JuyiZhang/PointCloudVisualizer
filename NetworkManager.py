import requests
import Debug

class NetworkUtils:
    
    def __init__(self, url = "https://mutaw.azurewebsites.net/") -> None:
        self.url = url
    
    # Adding host to server when online
    def add_host(self, hostip):
        
        Debug.Log("Adding Host IP: " + hostip)
        request_url = self.url + "add_host"
        request_content = {
            "op": "addm", # Add or Modify Host
            "hostname": hostip
        }
        return requests.post(request_url,json=request_content)

    # Deleting host to server when offline
    def del_host(self):
        
        request_url = self.url + "add_host"
        request_content = {
            "op": "delete", # Delete Host
        }
        return requests.post(request_url, json=request_content)

