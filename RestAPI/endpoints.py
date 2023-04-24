import logging
from flask_pymongo import pymongo
from flask import jsonify, request
import pandas as pd

def project_api_routes(endpoints):

    @endpoints.route('register', method=['POST'])
    def register():
        resp = {}
        