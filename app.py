from flask import Flask, request, render_template
import requests
import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configurations for Mappls API (replace with your credentials)
CLIENT_ID = '33OkryzDZsJRintCil75COPhvAcncmDK-4uqhF317GYY089TvqKY5qOIL0A4l4ZNlg0drDk26s2IQZrcpqxtWA=='  # Replace with your actual client ID
CLIENT_SECRET = 'lrFxI-iSEg_KHhNvMcOs6BTb5WFellX4Rfr6Bfpgf2Me8iQkvi1HI3cn4ydf9I8vjNuYU0yb0IDYWZhik4ztIu5mxsRNovNv'  # Replace with your actual client secret
PROFILE = 'f71022b4196166a460abf9190b460595'  # Replace with your actual profile if needed

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def ml_model(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r

# Load the ML model (Haversine function) from the pickle file
# with open('ML_model.pkl', 'rb') as file:
#     ml_model = pickle.load(file)

# Objective function for dual annealing
def evaluate(new_service_center, customers_df, service_centers_df):
    new_lat, new_lon = new_service_center
    total_distance = 0
    for i, row in customers_df.iterrows():
        cust_lat, cust_lon = row['latitude'], row['longitude']
        new_distance = ml_model(new_lat, new_lon, cust_lat, cust_lon)
        existing_min_distance = row['minimum']
        total_distance += min(new_distance, existing_min_distance)
    return total_distance

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and optimization using customer bounds
@app.route('/upload', methods=['POST'])
def upload_file():
    customer_file = request.files['customer_file']
    service_center_file = request.files['service_center_file']

    # Save files
    customer_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(customer_file.filename))
    service_center_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(service_center_file.filename))
    customer_file.save(customer_filepath)
    service_center_file.save(service_center_filepath)

    # Load data
    customers_df = pd.read_excel(customer_filepath)
    service_centers_df = pd.read_excel(service_center_filepath)

    # Calculate minimum distance
    customers_df['minimum'] = customers_df.apply(
        lambda row: min([ml_model(row['latitude'], row['longitude'], sc_lat, sc_lon) 
                         for sc_lat, sc_lon in zip(service_centers_df['latitude'], service_centers_df['longitude'])]), axis=1)
    
    # Get lat/lon bounds
    lat_bounds = (customers_df['latitude'].min(), customers_df['latitude'].max())
    lon_bounds = (customers_df['longitude'].min(), customers_df['longitude'].max())

    # Run dual annealing
    result = dual_annealing(evaluate, bounds=[lat_bounds, lon_bounds], args=(customers_df, service_centers_df),maxiter=100)
    optimal_new_center = np.round(result.x, 6)
    total_existing_distance = customers_df['minimum'].sum()

    # Number of customers benefited
    num_benefited = len(customers_df[customers_df.apply(
        lambda row: ml_model(optimal_new_center[0], optimal_new_center[1], row['latitude'], row['longitude']) < row['minimum'], axis=1)])

    # Render the results on the same page
    return render_template('index.html',
                           optimal_latitude=optimal_new_center[0],
                           optimal_longitude=optimal_new_center[1],
                           total_distance_after=result.fun,
                           total_distance_before=total_existing_distance,
                           num_beneficial_customers=num_benefited)

# Route to handle file upload and optimization using custom bounds
@app.route('/upload_custom_bounds', methods=['POST'])
def upload_custom_bounds():
    customer_file = request.files['customer_file']
    service_center_file = request.files['service_center_file']

    # Save files
    customer_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(customer_file.filename))
    service_center_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(service_center_file.filename))
    customer_file.save(customer_filepath)
    service_center_file.save(service_center_filepath)

    # Load data
    customers_df = pd.read_excel(customer_filepath)
    service_centers_df = pd.read_excel(service_center_filepath)

    # Calculate minimum distance
    customers_df['minimum'] = customers_df.apply(
        lambda row: min([ml_model(row['latitude'], row['longitude'], sc_lat, sc_lon) 
                         for sc_lat, sc_lon in zip(service_centers_df['latitude'], service_centers_df['longitude'])]), axis=1)

    # Get user-defined lat/lon bounds
    lat_lower = float(request.form['lat_lower'])
    lat_upper = float(request.form['lat_upper'])
    lon_lower = float(request.form['lon_lower'])
    lon_upper = float(request.form['lon_upper'])

    # Run dual annealing with custom bounds
    result = dual_annealing(evaluate, bounds=[(lat_lower, lat_upper), (lon_lower, lon_upper)], args=(customers_df, service_centers_df),maxiter=100)
    optimal_new_center = np.round(result.x, 6)
    total_existing_distance = customers_df['minimum'].sum()

    # Number of customers benefited
    num_benefited = len(customers_df[customers_df.apply(
        lambda row: ml_model(optimal_new_center[0], optimal_new_center[1], row['latitude'], row['longitude']) < row['minimum'], axis=1)])

    # Render the results on the same page
    return render_template('index.html',
                           manual_optimal_latitude=optimal_new_center[0],
                           manual_optimal_longitude=optimal_new_center[1],
                           manual_total_distance_after=result.fun,
                           manual_total_distance_before=total_existing_distance,
                           manual_beneficial_customers=num_benefited)

# Route to calculate distance before/after and number of customers benefitted for manually input lat/lon
@app.route('/calculate_distance', methods=['POST'])
def calculate_distance():
    customer_file = request.files['customer_file']
    service_center_file = request.files['service_center_file']

    # Save files
    customer_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(customer_file.filename))
    service_center_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(service_center_file.filename))
    customer_file.save(customer_filepath)
    service_center_file.save(service_center_filepath)

    # Load data
    customers_df = pd.read_excel(customer_filepath)
    service_centers_df = pd.read_excel(service_center_filepath)

    # Calculate minimum distance
    customers_df['minimum'] = customers_df.apply(
        lambda row: min([ml_model(row['latitude'], row['longitude'], sc_lat, sc_lon) 
                         for sc_lat, sc_lon in zip(service_centers_df['latitude'], service_centers_df['longitude'])]), axis=1)

    # Get new latitude and longitude
    new_lat = float(request.form['new_lat'])
    new_lon = float(request.form['new_lon'])

    # Total distance before
    total_existing_distance = customers_df['minimum'].sum()

    # Calculate total distance after using the new service center location
    total_new_distance = customers_df.apply(
        lambda row: min(ml_model(new_lat, new_lon, row['latitude'], row['longitude']), row['minimum']), axis=1).sum()

    # Calculate the number of customers benefited
    num_benefited = len(customers_df[customers_df.apply(
        lambda row: ml_model(new_lat, new_lon, row['latitude'], row['longitude']) < row['minimum'], axis=1)])

    # Render the results on the same page
    return render_template('index.html',
                           new_latitude=new_lat,
                           new_longitude=new_lon,
                           total_distance_after_new=total_new_distance,
                           total_distance_before_new=total_existing_distance,
                           num_beneficial_customers_new=num_benefited)

# Route to calculate the nearest service centers using Mappls API
@app.route('/find_nearest_service_center', methods=['POST'])
def calculate_nearest_centers():
    customer_file = request.files['customer_file']
    service_center_file = request.files['service_center_file']
    
    # Save uploaded files
    customer_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(customer_file.filename))
    service_center_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(service_center_file.filename))
    customer_file.save(customer_filepath)
    service_center_file.save(service_center_filepath)
    
    # Load customer and service center data
    customer_df = pd.read_excel(customer_filepath)
    service_center_df = pd.read_excel(service_center_filepath)
    
    customer_addresses = customer_df['addresses'].tolist()
    service_center_addresses = service_center_df['addresses'].tolist()
    
    # Get access token and geocode customer and service center addresses
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    
    service_centers_eloc, service_center_eloc_dict = get_eloc(service_center_addresses, access_token)
    customers_eloc, customer_eloc_dict = get_eloc(customer_addresses, access_token)
    
    distance_matrix = get_distance_and_time(access_token, service_centers_eloc, customers_eloc)
    nearest_centers = find_nearest_service_center(distance_matrix, len(customers_eloc), len(service_centers_eloc))
    
    results = []
    for idx, customer_address in enumerate(customer_addresses):
        nearest_service_center_idx, nearest_distance = nearest_centers[idx]
        nearest_service_center_eloc = service_centers_eloc[nearest_service_center_idx]
        nearest_service_center_address = service_center_addresses[nearest_service_center_idx]
        results.append({
            'customer_address': customer_address,
            'customer_eloc': customers_eloc[idx],
            'nearest_service_center_eloc': nearest_service_center_eloc,
            'nearest_service_center_address': nearest_service_center_address,
            'distance': nearest_distance
        })
    
    return render_template('index.html', results=results)

# Mappls API functions for eLoc retrieval and distance calculation
def get_eloc(address_list, access_token):
    eloc_list = []
    eloc_dict = {}
    for address in address_list:
        if access_token:
            response = get_lat_long(address, access_token)
            temp = response.json()
            if 'copResults' in temp:
                eloc = temp['copResults']['eLoc']
                eloc_list.append(eloc)
                eloc_dict[address] = eloc
            else:
                print(f"Failed to retrieve eLoc for {address}")
    return eloc_list, eloc_dict

def get_lat_long(address, access_token):
    url = f'https://atlas.mappls.com/api/places/geocode'
    headers = {'Authorization': f'bearer {access_token}'}
    params = {'address': address}
    return requests.get(url, headers=headers, params=params)

def get_access_token(client_id, client_secret):
    url = "https://outpost.mappls.com/api/security/oauth/token"
    params = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
    response = requests.post(url, params=params)
    token_data = response.json()
    return token_data.get('access_token')

# def get_distance_and_time(access_token, sp_eloc, cust_eloc):
#     eloc_string = ";".join(sp_eloc + cust_eloc)
#     service_center_count = len(sp_eloc)
#     sources_param = ";".join([str(i) for i in range(service_center_count)])
#     dest_param = ";".join([str(i) for i in range(service_center_count, len(sp_eloc) + len(cust_eloc))])
    
#     url = f"https://atlas.mappls.com/api/directions/v1/route"
#     headers = {'Authorization': f'bearer {access_token}'}
#     params = {'sources': sources_param, 'destinations': dest_param}
#     response = requests.get(url, headers=headers, params=params)
#     return response.json()

def get_distance_and_time(access_token, sp_eloc,cust_eloc):
    res=""
    res1=""
    n=0
    for i in sp_eloc:
      res = res+f"{i};"
      res1 = res1+f"{n};"
      n+=1
    for i in cust_eloc:
      res = res+f"{i};"
    print(res[:-1])
    url = f"https://apis.mapmyindia.com/advancedmaps/v1/f71022b4196166a460abf9190b460595/distance_matrix/driving/{res[:-1]}?rtype=0&region=ind&sources={res1[:-1]}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "rtype": "0",
        "region": "ind"
    }
    response = requests.get(url, headers=headers)#, params=params)
    return response.json()


def find_nearest_service_center(distance_matrix, customer_count, service_center_count):
    distances = distance_matrix['results']['distances']
    distances = list(zip(*distances))
    nearest_centers = []
    
    for i in range(service_center_count, service_center_count + customer_count):
        customer_distances = distances[i][:service_center_count]
        nearest_service_center = min(range(service_center_count), key=lambda x: customer_distances[x])
        nearest_centers.append((nearest_service_center, customer_distances[nearest_service_center]))
    
    return nearest_centers

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
