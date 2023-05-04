import logging
from flask_pymongo import pymongo
from flask import jsonify, request
import pandas as pd
from main import input_function


con_string = "mongodb+srv://aravindkv27:Aravind01@sfc.pvzgik1.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(con_string)
db = client.get_database('pyrodb')
user_collection = pymongo.collection.Collection(db, 'user_cred')
print("MongoDB connected Successfully")

code={}



def project_api_routes(endpoints):

    @endpoints.route('/signup', methods=['POST'])
    def register_user():
        resp = {}
        try:
            req_body = request.json
            # req_body = req_body.to_dict()
            user_collection.insert_one(req_body)            
            print("User Data Stored Successfully in the Database.")
            status = {
                "statusCode":"200",
                "statusMessage":"User Data Stored Successfully in the Database."
            }
        except Exception as e:
            print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
        resp["status"] =status
        return resp

    @endpoints.route('/read-users',methods=['GET'])
    def read_users():
        resp = {}
        try:
            users = user_collection.find({})
            print(users)
            users = list(users)
            status = {
                "statusCode":"200",
                "statusMessage":"User Data Retrieved Successfully from the Database."
            }
            output = [{'email' : user['email'], 'pass' : user['pass']} for user in users]   #list comprehension
            resp['data'] = output
        except Exception as e:
            print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
        resp["status"] =status
        return resp

    @endpoints.route('/home',methods=['POST', 'GET'])
    def home():
        resp = {}
        try:
            req = request.get_json()

            # print()
            english_input = req['engSent']
            print(req)
            
            code = input_function(english_input)
            # print(code)
            resp['data']=code
           
        except Exception as e:
            # print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
            
            resp["data"] =status
        # return jsonify(code)
        print(resp)
        return resp

        
    # @endpoints.route('/get_data',methods=['GET'])
    # def get_data():
    #     return pmts
    return endpoints