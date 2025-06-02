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

def ml_model(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 
    base_distance = c * r
    corrected_distance = apply_correction(base_distance)
    return corrected_distance

def apply_correction(distance):
    if distance <= 0.5:
        correction_factor = 0.05
    elif 0.5 < distance <= 1:
        correction_factor = 0.07
    elif 1 < distance <= 3:
        correction_factor = 0.10
    elif 3 < distance <= 5:
        correction_factor = 0.12
    elif 5 < distance <= 10:
        correction_factor = 0.15
    elif 10 < distance <= 20:
        correction_factor = 0.17
    elif 20 < distance <= 75:
        correction_factor = 0.16
    elif 75 < distance <= 110:
        correction_factor = 0.20
    elif 110 < distance <= 170:
        correction_factor = 0.16  
    elif 170 < distance <= 210:
        correction_factor = 0.20
    elif 210 < distance <= 260:
        correction_factor = 0.25
    else:
        correction_factor = 0.30
    corrected_distance = distance * (1 + correction_factor)
    return corrected_distance

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
    import folium
    import os
    from folium.plugins import FloatImage
    from branca.element import Template, MacroElement

    # Combine existing and new service centers
    all_centers = service_centers_df[['latitude', 'longitude']].values.tolist()
    all_centers.append(optimal_new_center.tolist())

    # Calculate map bounds
    all_lats = list(customers_df['latitude']) + [lat for lat, _ in all_centers]
    all_lngs = list(customers_df['longitude']) + [lng for _, lng in all_centers]
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lng, max_lng = min(all_lngs), max(all_lngs)
    center_lat = (min_lat + max_lat) / 2
    center_lng = (min_lng + max_lng) / 2

    # Create folium map
    city_map = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles='OpenStreetMap')

    # Add customer points
    for idx, row in customers_df.iterrows():
        # Distance to optimal new center
        new_dist = ml_model(optimal_new_center[0], optimal_new_center[1], row['latitude'], row['longitude'])
        # Use orange for customers who benefit from new center
        color = 'orange' if new_dist < row['minimum'] else 'blue'

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Customer {idx+1}"
        ).add_to(city_map)

    # Add existing service centers
    for i, center in enumerate(all_centers[:-1]):
        folium.Marker(
            location=center,
            popup=f"Service Center {i+1}",
            icon=folium.Icon(color='green', icon='building')
        ).add_to(city_map)

    # Add optimal new center
    folium.Marker(
        location=optimal_new_center,
        popup='New Optimal Center',
        icon=folium.Icon(color='red', icon='star')
    ).add_to(city_map)

    # Add bounding rectangle
    folium.Rectangle(
        bounds=[(min_lat, min_lng), (max_lat, max_lng)],
        color='black',
        fill=True,
        fill_opacity=0.05
    ).add_to(city_map)

    # Add legend (custom HTML)
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 220px; height: 160px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white; padding: 10px; opacity: 0.9;
    ">
    <b>Legend</b><br>
    <i class="fa fa-map-marker fa-2x" style="color:green"></i> Existing Service Center<br>
    <i class="fa fa-star fa-2x" style="color:red"></i> New Optimal Center<br>
    <span style="color:orange;">●</span> Customer benefiting from new center<br>
    <span style="color:blue;">●</span> Other Customer<br>
    </div>
    {% endmacro %}
    """

    legend = MacroElement()
    legend._template = Template(legend_html)
    city_map.get_root().add_child(legend)

    # Save the map
    os.makedirs('static', exist_ok=True)
    plot_filename = 'service_center_map.html'
    plot_path = os.path.join('static', plot_filename)
    city_map.save(plot_path)

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

# global_results = []  # Store results globally

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
    
    # Get threshold from form
    # ml_threshold = float(request.form.get("ml_threshold", 100))  # default is 100 if not provided
    ml_threshold_raw = request.form.get("ml_threshold", "").strip()
    ml_threshold = float(ml_threshold_raw) if ml_threshold_raw else 100.0
    


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
    sc_names = service_center_df['service_centers'].tolist()
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

        if min_distance <= ml_threshold:
            ml_results.append({
                'customer_address': customer_addresses[i],
                'customer_latitude': cust_lat,
                'customer_longitude': cust_lon,
                'nearest_service_center_address': sc_addresses[nearest_idx],
                'nearest_service_center_name': sc_names[nearest_idx],
                'nearest_service_center_latitude': sc_lats[nearest_idx],
                'nearest_service_center_longitude': sc_lons[nearest_idx],
                'final_distance': min_distance,
                'source': 'ML'
            })
        else:
            api_input.append({
                'address': customer_addresses[i],
                'latitude': cust_lat,
                'longitude': cust_lon
            })

    # Step 2: Run API logic
    api_results = []
    if api_input:
        api_customer_df = pd.DataFrame(api_input)
        api_customer_df['addresses'] = api_customer_df['address']

        api_cust_excel = 'temp_api_cust.xlsx'
        api_customer_df.to_excel(api_cust_excel, index=False)
        service_center_df.to_excel('temp_api_sc.xlsx', index=False)

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
                'nearest_service_center_name': sc_names[nearest_idx],
                'nearest_service_center_latitude': sc_lats[nearest_idx],
                'nearest_service_center_longitude': sc_lons[nearest_idx],
                'final_distance': distance / 1000,
                'source': 'API'
            })

    # Step 3: Combine ML + API results
    combined_results = ml_results + api_results

    # Step 4: Sort results
    address_order = customer_df[['addresses', 'latitude', 'longitude']]
    final_df = pd.DataFrame(combined_results)
    merged_df = address_order.merge(final_df, left_on=['addresses', 'latitude', 'longitude'],
                                    right_on=['customer_address', 'customer_latitude', 'customer_longitude'],
                                    how='left')
    # ✅ Step 5: Save results to downloadable Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        merged_df.to_excel(writer, sheet_name='Hybrid_Results', index=False)
    output.seek(0)

    hybrid_download_filename = f"hybrid_results.xlsx"
    hybrid_download_path = os.path.join('static', hybrid_download_filename)
    with open(hybrid_download_path, 'wb') as f:
        f.write(output.read())

    return render_template("index.html",
                           hybrid_results=merged_df.to_dict(orient='records'),
                           hybrid_download_link=hybrid_download_filename)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True,port=3002)
