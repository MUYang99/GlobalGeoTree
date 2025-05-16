import pandas as pd
from pygbif import occurrences
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 'EE'  52
Europe = [
    'DE', 'AL', 'AX', 'AD', 'AT', 'BY', 'BE', 'BA', 'BG', 'HR', 'CZ', 'DK',
    'FO', 'FI', 'FR', 'GI', 'GR', 'HU', 'IS', 'IE', 'IT', 'LV', 'LI',
    'LT', 'LU', 'MT', 'MD', 'MC', 'ME', 'NL', 'MK', 'NO', 'PL', 'PT', 'RO',
    'RU', 'SM', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'UA', 'GB', 'VA', 'XK',
    'GG', 'JE', 'IM', 'SJ'
]

# 加载物种数据库 CSV 文件
file_path = './data/Tree_species_catalog.csv'
df = pd.read_csv(file_path)

# 国家进度文件
country_progress_file = './data/Geolocations/Europe/country_progress.csv'

# 加载国家的进度
if os.path.exists(country_progress_file):
    with open(country_progress_file, 'r') as f:
        last_country_code = f.read().strip()
    start_index = Europe.index(last_country_code) + 1 if last_country_code in Europe else 0
else:
    start_index = 0


def save_country_progress(country_code):
    """保存当前处理的国家代码到进度文件"""
    with open(country_progress_file, 'w') as f:
        f.write(country_code)


def save_progress(progress_file, species_key):
    """保存已处理物种的species_key到进度文件"""
    with open(progress_file, 'a') as f:
        f.write(f"{species_key}\n")


def get_dataset_info(dataset_key):
    url = f"https://api.gbif.org/v1/dataset/{dataset_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_species_geolocations(species_key, country_code, batch_size=300):
    offset = 0
    geolocations = []

    while True:
        if offset > 100000:  # 上限检查，根据 API 实际限制调整
            print(f"Offset exceeded limit for species_key: {species_key}")
            break

        occurrences_data = occurrences.search(
            taxonKey=species_key,
            hasCoordinate=True,
            basisOfRecord="HUMAN_OBSERVATION",
            year="2015,2024",
            hasGeospatialIssue=False,
            occurrenceStatus="present",
            limit=batch_size,
            country=country_code,
            offset=offset
        )

        if not occurrences_data['results']:
            break

        for result in occurrences_data['results']:
            if 'decimalLatitude' in result and 'decimalLongitude' in result:
                dataset_key = result['datasetKey']
                dataset_info = get_dataset_info(dataset_key)
                dataset_title = dataset_info.get('title', 'Unknown')

                lat = result['decimalLatitude']
                lon = result['decimalLongitude']

                if not any(geo['lat'] == lat and geo['lon'] == lon for geo in geolocations):
                    geolocations.append({
                        'lat': lat,
                        'lon': lon,
                        'location': result.get('country', 'Unknown'),
                        'event_year': result.get('eventDate', None).split("-")[0],
                        'source': dataset_title
                    })

        offset += batch_size

    return geolocations


def process_species(row, country_code, processed_species_keys, output_file_path, progress_file):
    species_key = row['species_key']

    if species_key in processed_species_keys:
        print(f"Skipping already processed species key: {species_key}")
        return

    if pd.notna(species_key):
        geolocations = get_species_geolocations(species_key, country_code)
        geolocation_data = []
        for location in geolocations:
            new_row = row.copy()
            new_row['latitude'] = location['lat']
            new_row['longitude'] = location['lon']
            new_row['location'] = location['location']
            new_row['year'] = location['event_year']
            new_row['source'] = location['source']
            geolocation_data.append(new_row)

        df_geolocations = pd.DataFrame(geolocation_data)

        if not df_geolocations.empty:
            print(f"Processing species key: {species_key}")
            df_geolocations.to_csv(output_file_path, mode='a', header=not os.path.exists(output_file_path), index=False)
            save_progress(progress_file, species_key)
        else:
            save_progress(progress_file, species_key)
        # print(f"Finished processing species key: {species_key}")
    else:
        print(f"Species key '{species_key}' is invalid or missing in the CSV file")


for country_code in Europe[start_index:]:
    print(f"[INFO] Processing {country_code}")

    # 进度文件路径
    progress_file = f'./data/Geolocations/Europe/{country_code}/progress_species_keys_{country_code}.csv'
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    if os.path.exists(progress_file):
        processed_species_keys = pd.read_csv(progress_file)['species_key'].tolist()
    else:
        with open(progress_file, 'w') as f:
            f.write("species_key\n")
        processed_species_keys = []

    output_file_path = f'./data/Geolocations/Europe/{country_code}/geolocations_{country_code}.csv'

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_species, row, country_code, processed_species_keys, output_file_path, progress_file)
            for _, row in df.iterrows()
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a species: {e}")

    # 更新国家进度
    save_country_progress(country_code)

print("All countries processed successfully.")
