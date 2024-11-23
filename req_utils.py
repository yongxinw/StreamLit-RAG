import requests
import hashlib
import time

def generate_signature(timestamp, secret_key):
    data = f"{timestamp}{secret_key}".encode('utf-8')
    return hashlib.md5(data).hexdigest().upper()