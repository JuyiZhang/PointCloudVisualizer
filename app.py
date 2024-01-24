from flask import Flask, render_template

import json
import SessionManager
import Debug

session: SessionManager.Session = None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/device-location")
def get_device_location():
    device_location = session.get_all_pose()
    
    return json.dumps(device_location)

@app.route("/observed-person")
def get_observed_person_location():
    observed_person_location = session.get_observed_list()
    print(observed_person_location)
    return json.dumps(observed_person_location)

@app.route("/set-main-device", methods=['POST'])
def set_main_device():
    return "Not Implemented"

    