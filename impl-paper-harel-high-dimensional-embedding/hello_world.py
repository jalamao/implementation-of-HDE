from flask import Flask
import snap
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"