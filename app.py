from flask import Flask, request, render_template
import requests
import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing
import pickle
import os
from werkzeug.utils import secure_filename
from flask import send_file
import io
from datetime import datetime


app = Flask(__name__)

# Configurations for Mappls API (replace with your credentials)
CLIENT_ID = '33OkryzDZsJRintCil75COPhvAcncmDK-4uqhF317GYY089TvqKY5qOIL0A4l4ZNlg0drDk26s2IQZrcpqxtWA=='  # Replace with your actual client ID
CLIENT_SECRET = 'lrFxI-iSEg_KHhNvMcOs6BTb5WFellX4Rfr6Bfpgf2Me8iQkvi1HI3cn4ydf9I8vjNuYU0yb0IDYWZhik4ztIu5mxsRNovNv'  # Replace with your actual client secret
PROFILE = 'f71022b4196166a460abf9190b460595'  # Replace with your actual profile if needed

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# def ml_model(lat1, lon1, lat2, lon2):
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     r = 6371  # Earth radius in kilometers
#     return c * r

def ml_model(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    base_distance = c * r

    # Apply distance correction based on the range
    corrected_distance = apply_correction(base_distance)
    return corrected_distance

def apply_correction(distance):
    """
    Applies a correction to the Haversine distance based on predefined ranges.
    The correction is added as a percentage of the original Haversine distance.
    """
    if distance <= 0.5:
        # 0 - 500 meters -> 2-5% correction
        correction_factor = 0.05
    elif 0.5 < distance <= 1:
        # 500 meters - 1 km -> 5-7% correction
        correction_factor = 0.07
    elif 1 < distance <= 3:
        # 1 km - 3 km -> 7-10% correction
        correction_factor = 0.10
    elif 3 < distance <= 5:
        # 3 km - 5 km -> 10-12% correction
        correction_factor = 0.12
    elif 5 < distance <= 10:
        # 5 km - 10 km -> 12-15% correction
        correction_factor = 0.15
    elif 10 < distance <= 20:
        # 10 km - 20 km -> 15-18% correction
        correction_factor = 0.18
    else:
        # Greater than 20 km -> 18-20% correction
        correction_factor = 0.20

    # Apply the correction as a percentage of the base distance
    corrected_distance = distance * (1 + correction_factor)
    return corrected_distance

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

def generate_plot(customers_df, service_centers_df, optimal_new_center):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Combine existing and new service centers
    all_centers = service_centers_df[['latitude', 'longitude']].values.tolist()
    all_centers.append(optimal_new_center.tolist())

    # Calculate distance from each customer to all centers
    distance_matrix = []
    for _, row in customers_df.iterrows():
        cust_lat, cust_lon = row['latitude'], row['longitude']
        distances = [ml_model(sc_lat, sc_lon, cust_lat, cust_lon) for sc_lat, sc_lon in all_centers]
        distance_matrix.append(distances)

    distance_matrix = np.array(distance_matrix)
    nearest_indices = np.argmin(distance_matrix, axis=1)  # Index of nearest service center

    # Use updated colormap API
    color_palette = plt.colormaps.get_cmap('tab10')
    customer_colors = [color_palette(idx % 10) for idx in nearest_indices]

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(customers_df['longitude'], customers_df['latitude'], color=customer_colors, s=40, alpha=0.7)

    for i, center in enumerate(all_centers):
        plt.scatter(center[1], center[0], color=color_palette(i), marker='X', edgecolor='black', s=200,
                    label=f'Service Center {i+1}' if i < len(all_centers)-1 else 'New Optimal Center')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Customers and Service Centers (including new optimized center)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs('static', exist_ok=True)
    plot_filename = 'service_center_plot.png'
    plot_path = os.path.join('static', plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_filename

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

    # Total number of customers
    total_customers = len(customers_df)

    # Generate the plot and get the path
    plot_filename = generate_plot(customers_df, service_centers_df, optimal_new_center)

    # ✅ Create downloadable Excel with existing + new service center
    existing_centers = service_centers_df.copy()
    existing_centers['type'] = 'existing'

    new_center = pd.DataFrame([{
        'latitude': optimal_new_center[0],
        'longitude': optimal_new_center[1],
        'type': 'new'
    }])

    combined_centers = pd.concat([existing_centers, new_center], ignore_index=True)

    # Save to file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        combined_centers.to_excel(writer, sheet_name='Service_Centers', index=False)
    output.seek(0)

    download_filename = f"combined_service_centers.xlsx"
    download_path = os.path.join('static', download_filename)
    with open(download_path, 'wb') as f:
        f.write(output.read())

    # Render the results on the same page
    return render_template('index.html',
                           optimal_latitude=optimal_new_center[0],
                           optimal_longitude=optimal_new_center[1],
                           total_distance_after=result.fun,
                           total_distance_before=total_existing_distance,
                           total_customers_new=total_customers,
                           num_beneficial_customers=num_benefited,
                           plot_filename=plot_filename,
                           download_link=download_filename)

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

    # Total number of customers
    total_customers1 = len(customers_df)

    # Generate and save plot
    plot_filename = generate_plot(customers_df, service_centers_df, optimal_new_center)

    # ✅ Create downloadable Excel with existing + new service center
    existing_centers1 = service_centers_df.copy()
    existing_centers1['type'] = 'existing'

    new_center1 = pd.DataFrame([{
        'latitude': optimal_new_center[0],
        'longitude': optimal_new_center[1],
        'type': 'new'
    }])

    combined_centers1 = pd.concat([existing_centers1, new_center1], ignore_index=True)

    # Save to Excel in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        combined_centers1.to_excel(writer, sheet_name='Service_Centers', index=False)
    output.seek(0)

    # Save to file in static folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manual_download_filename = f"custom_bounds_centers_{timestamp}.xlsx"
    manual_download_path = os.path.join('static', manual_download_filename)
    with open(manual_download_path, 'wb') as f:
        f.write(output.read())

    # Render the results on the same page
    return render_template('index.html',
                           manual_optimal_latitude=optimal_new_center[0],
                           manual_optimal_longitude=optimal_new_center[1],
                           manual_total_distance_after=result.fun,
                           manual_total_distance_before=total_existing_distance,
                           total_customers_new1=total_customers1,
                           manual_beneficial_customers=num_benefited,
                           manual_plot_filename=plot_filename,
                           manual_download_link=manual_download_filename)

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

    # Total number of customers
    total_customers2 = len(customers_df)

    # Generate and save plot
    plot_filename = generate_plot(customers_df, service_centers_df, np.array([new_lat, new_lon]))

    # Render the results on the same page
    return render_template('index.html',
                           new_latitude=new_lat,
                           new_longitude=new_lon,
                           total_distance_after_new=total_new_distance,
                           total_distance_before_new=total_existing_distance,
                           total_customers_new2=total_customers2,
                           num_beneficial_customers_new=num_benefited,
                           new_plot_filename=plot_filename)

global_results = []  # Store results globally

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
    
    # global_results = results  # Store results globally
    global global_results  
    global_results.clear()  # Clear old results before adding new ones
    global_results.extend(results)  # Update with new results

    return render_template('index.html', results=results)

@app.route('/download-results')
def download_results():
    global global_results
    if not global_results:
        return "No results available for download", 400
    
    # Convert results to DataFrame
    df = pd.DataFrame(global_results)

    # Rename columns for better readability
    df = df.rename(columns={
        "customer_address": "Customer Address",
        "customer_eloc": "Customer eLoc",
        "nearest_service_center_eloc": "Nearest Service Center eLoc",
        "nearest_service_center_address": "Nearest Service Center Address",
        "distance": "Distance (km)"
    })

    # Convert distance from meters to kilometers
    df["Distance (km)"] = df["Distance (km)"] / 1000

    # Convert DataFrame to Excel
    excel_file = "nearest_service_centers.xlsx"
    df.to_excel(excel_file, index=False, engine='openpyxl')

    # Read the file and send as response
    with open(excel_file, "rb") as f:
        data = f.read()

    response = Response(data, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response.headers["Content-Disposition"] = f"attachment; filename={excel_file}"

    return response

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

# Route to calculate the nearest service centers using ML model (without API)
@app.route('/find_nearest_service_center_ml', methods=['POST'])
def find_nearest_center_ml():
    ml_customer_file = request.files['ml_customer_file']
    ml_service_center_file = request.files['ml_service_center_file']
    
    # Save uploaded files
    ml_customer_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(ml_customer_file.filename))
    ml_service_center_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(ml_service_center_file.filename))
    ml_customer_file.save(ml_customer_filepath)
    ml_service_center_file.save(ml_service_center_filepath)
    
    # Load customer and service center data
    ml_customer_df = pd.read_excel(ml_customer_filepath)
    ml_service_center_df = pd.read_excel(ml_service_center_filepath)
    
    ml_customer_addresses = ml_customer_df['addresses'].tolist()
    ml_customer_latitudes = ml_customer_df['latitude'].tolist()
    ml_customer_longitudes = ml_customer_df['longitude'].tolist()
    
    ml_service_center_addresses = ml_service_center_df['addresses'].tolist()
    ml_service_center_latitudes = ml_service_center_df['latitude'].tolist()
    ml_service_center_longitudes = ml_service_center_df['longitude'].tolist()
    
    ml_service_center_results = []
    for i, (cust_lat, cust_lon) in enumerate(zip(ml_customer_latitudes, ml_customer_longitudes)):
        min_distance = float('inf')
        nearest_service_center_idx = -1
        
        for j, (sc_lat, sc_lon) in enumerate(zip(ml_service_center_latitudes, ml_service_center_longitudes)):
            distance = ml_model(cust_lat, cust_lon, sc_lat, sc_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_service_center_idx = j
        
        ml_service_center_results.append({
            'customer_address': ml_customer_addresses[i],
            'customer_latitude': cust_lat,
            'customer_longitude': cust_lon,
            'nearest_service_center_address_using_ml': ml_service_center_addresses[nearest_service_center_idx],
            'nearest_service_center_latitude': ml_service_center_latitudes[nearest_service_center_idx],
            'nearest_service_center_longitude': ml_service_center_longitudes[nearest_service_center_idx],
            'distance_using_ml': min_distance
        })
    
    return render_template('index.html', ml_results=ml_service_center_results)

@app.route('/find_nearest_service_center_hybrid', methods=['POST'])
def find_nearest_service_center_hybrid():
    # Upload files
    hybrid_customer_file = request.files['hybrid_customer_file']
    hybrid_service_center_file = request.files['hybrid_service_center_file']

    # Save files
    cust_path = os.path.join(UPLOAD_FOLDER, secure_filename(hybrid_customer_file.filename))
    sc_path = os.path.join(UPLOAD_FOLDER, secure_filename(hybrid_service_center_file.filename))
    hybrid_customer_file.save(cust_path)
    hybrid_service_center_file.save(sc_path)

    # Load data
    customer_df = pd.read_excel(cust_path)
    service_center_df = pd.read_excel(sc_path)

    # Extract data
    customer_addresses = customer_df['addresses'].tolist()
    cust_lats = customer_df['latitude'].tolist()
    cust_lons = customer_df['longitude'].tolist()

    sc_addresses = service_center_df['addresses'].tolist()
    sc_lats = service_center_df['latitude'].tolist()
    sc_lons = service_center_df['longitude'].tolist()

    ml_results = []
    api_input = []

    # Step 1: Run ML predictions and split
    for i, (cust_lat, cust_lon) in enumerate(zip(cust_lats, cust_lons)):
        min_distance = float('inf')
        nearest_idx = -1

        for j, (sc_lat, sc_lon) in enumerate(zip(sc_lats, sc_lons)):
            dist = ml_model(cust_lat, cust_lon, sc_lat, sc_lon)
            if dist < min_distance:
                min_distance = dist
                nearest_idx = j

        if min_distance <= 100:
            ml_results.append({
                'customer_address': customer_addresses[i],
                'customer_latitude': cust_lat,
                'customer_longitude': cust_lon,
                'nearest_service_center_address': sc_addresses[nearest_idx],
                'nearest_service_center_latitude': sc_lats[nearest_idx],
                'nearest_service_center_longitude': sc_lons[nearest_idx],
                'final_distance': min_distance,
                'source': 'ML'
            })
        else:
            # Push for API processing
            api_input.append({
                'address': customer_addresses[i],
                'latitude': cust_lat,
                'longitude': cust_lon
            })

    # Step 2: Run API logic for customers >50km
    api_results = []
    if api_input:
        # Prepare files
        api_customer_df = pd.DataFrame(api_input)
        api_customer_df['addresses'] = api_customer_df['address']

        # Save temp Excel files
        api_cust_excel = 'temp_api_cust.xlsx'
        api_customer_df.to_excel(api_cust_excel, index=False)
        service_center_df.to_excel('temp_api_sc.xlsx', index=False)

        # Reuse original API logic
        access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
        sc_eloc, _ = get_eloc(sc_addresses, access_token)
        cust_eloc, _ = get_eloc(api_customer_df['addresses'].tolist(), access_token)

        distance_matrix = get_distance_and_time(access_token, sc_eloc, cust_eloc)
        nearest_centers = find_nearest_service_center(distance_matrix, len(cust_eloc), len(sc_eloc))

        for idx, entry in enumerate(api_input):
            nearest_idx, distance = nearest_centers[idx]
            api_results.append({
                'customer_address': entry['address'],
                'customer_latitude': entry['latitude'],
                'customer_longitude': entry['longitude'],
                'nearest_service_center_address': sc_addresses[nearest_idx],
                'nearest_service_center_latitude': sc_lats[nearest_idx],
                'nearest_service_center_longitude': sc_lons[nearest_idx],
                'final_distance': distance/1000,
                'source': 'API'
            })

    # Step 3: Combine ML + API results
    combined_results = ml_results + api_results

    # Step 4: Sort results to match original customer file order
    address_order = customer_df[['addresses', 'latitude', 'longitude']]
    final_df = pd.DataFrame(combined_results)
    merged_df = address_order.merge(final_df, left_on=['addresses', 'latitude', 'longitude'],
                                    right_on=['customer_address', 'customer_latitude', 'customer_longitude'],
                                    how='left')

    return render_template("index.html", hybrid_results=merged_df.to_dict(orient='records'))


# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
