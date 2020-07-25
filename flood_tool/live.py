"""Live and historical flood monitoring data from the Environment Agency API"""
import re
import os
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import flood_tool.tool as tool

__all__  = ['get_closest_station', 'get_station_rainfall_range',
            'get_today_prediction', 'get_all_stations',
            'get_station_mean', 'get_mean_class', 'get_history_stations_rainfall']

LIVE_URL = "http://environment.data.gov.uk/flood-monitoring/id/stations"
ARCHIVE_URL = "http://environment.data.gov.uk/flood-monitoring/archive/"
READINGS_URL = "http://environment.data.gov.uk/flood-monitoring/id/stations/%s/readings"

postcode_file = os.sep.join(("flood_tool", "resources","postcodes.csv"))
risk_file = os.sep.join(("flood_tool", "resources", "flood_probability.csv"))
values_file = os.sep.join(("flood_tool", "resources", "property_value.csv"))

def get_closest_station(latitude, longitude, distance=10):
    """Get the cloest station within certain distance from the given location using LIVE_URL

    Parameters
    ----------

    latitude : float
        the latitude of the location
    longitude : float
        the longitude of the location
    distance : int
        the radius to search, default 10
    
    Retruns
    -------

    DataFrame
        A dataframe with columns named Lat, Long and Station.
        Station stands for the id of the cloest station.
        ValueError raised when there is no stations within given distance.
    """
    lat = []
    lon = []
    station = []
    station_url = "%s?parameter=rainfall&lat=%s&long=%s&dist=%s" % (LIVE_URL, latitude, longitude, distance)
    api = requests.get(station_url).json()
    if api is None or len(api["items"]) == 0:
        raise ValueError
    for item in api["items"]:
        lat.append(item["lat"])
        lon.append(item["long"])
        station.append(item["notation"])
        break
    return pd.DataFrame({'Lat': lat, 'Long': lon, 'Station': station})

def get_station_rainfall_range(station_id, start_date, end_date):
    """Get the daily rainfall data of certain station during specific period.
    According to the API, the maximum data it would return is a 5-day full data.

    Parameters
    ----------

    station_id : str
        the id of station
    start_date : str
        start date of the period, format: YYYY-MM-DD
    end_date : str
        end date of the period, format: YYYY-MM-DD
    
    Retruns
    -------

    DataFrame
        A dataframe indexed by datetime with column named rainfall
    """
    regu = re.compile(r'(.*?)T')
    rainfall_values = []
    datetime_values = []
    reading_url = "%s?startdate=%s&enddate=%s" % (READINGS_URL % station_id, start_date, end_date)
    api = requests.get(reading_url).json()
    if api is None or "items" not in api:
        return None
    for item in api['items']:
        rainfall_values.append(item['value'])
        item['dateTime'] = regu.findall(item['dateTime'])
        datetime_values.append(item['dateTime'][0])
    station_rainfall_df = pd.DataFrame({'rainfall': rainfall_values}, index=datetime_values)
    station_rainfall_df.index.name = "datetime"
    return station_rainfall_df

def get_today_prediction(postcodes, show_plot=True):
    """Get the daily rainfall data prediction of the given postcodes
    based on the data in the past 30 days.

    Parameters
    ----------

    postcodes : a sequence of postcodes
        Ordered sequence of N postcode strings
    show_plot : bool
        A switch for showing the plots station by station.
    
    Retruns
    -------

    DataFrame
        A dataframe indexed by Postcode with columns
        Rain_Risk, Property_Risk, Flood_Prediction
    """
    tmp_postcodes = postcodes.copy()
    postcodes = []
    for postcode in tmp_postcodes:
        postcode = tool.trim_postcode(postcode)
        if len(postcode) == 7:
            postcodes.append(postcode)
    flood_Tool = tool.Tool(postcode_file, risk_file, values_file)
    positions = flood_Tool.get_lat_long(postcodes)
    probability_bands = flood_Tool.get_sorted_flood_probability(postcodes).reindex(postcodes)
    now = datetime.datetime.now()
    rain_risk = []
    rain_map = {"High": 0, "Medium": 1, "Low": 2, "Very Low": 3, "Zero": 4}
    prop_map = {"High": 0, "Medium": 1, "Low": 2, "Very Low": 3, "No Risk": 4, "Zero": 4}
    flood_risk_array = [[0, 1, 1, 2, 3], 
                        [1, 1, 1, 2, 3],
                        [1, 1, 2, 3, 3],
                        [2, 3, 3, 3, 4],
                        [4, 4, 4, 4, 4]]
    flood_risk_map = ["High", "Medium", "Low", "Very Low", "No Risk"]
    img_map = ["high_risk.png", "medium_risk.png", "low_risk.png", "very_low_risk.png", "no_risk.png"]
    if show_plot:
        fig = plt.figure()
    for i, position in enumerate(positions):
        try:
            closest_station = get_closest_station(position[0], position[1])
        except ValueError:
            print("unable to find station for %s" % postcodes[i])
            rain_risk.append('Zero')
            continue
        daily_rainfall_df = None
        for p5d in range(1, 7): # 6*5=30 days
            daily_rainfall_tmp_df = get_station_rainfall_range(closest_station['Station'][0], 
                str(now - datetime.timedelta(days=(p5d)*5+1)).split()[0], \
                    str(now - datetime.timedelta(days=(p5d-1)*5+1)).split()[0])
            if daily_rainfall_tmp_df is None:
                continue
            if daily_rainfall_df is None:
                daily_rainfall_df = daily_rainfall_tmp_df.copy()
            else:
                daily_rainfall_df = pd.concat([daily_rainfall_df, daily_rainfall_tmp_df])
        if daily_rainfall_df is None or len(daily_rainfall_df) == 0:
            continue
        daily_rainfall_df = daily_rainfall_df.groupby('datetime').sum()
        mean_value = sum(daily_rainfall_df.rainfall)/len(daily_rainfall_df)
        if (mean_value >= 0.2) and (mean_value < 5):
            rain = 'Very Low'
        elif (mean_value >= 5) and (mean_value < 10):
            rain = 'Low'
        elif(mean_value >= 10) and (mean_value < 15):
            rain = 'Medium'
        elif mean_value >= 15:
            rain = 'High'
        else:
            rain = 'Zero'
        rain_risk.append(rain)
        if show_plot:
            ax1 = fig.add_subplot(int((len(postcodes) - 1)/2+1)*2,2,2*i+1)
            flood_p = prop_map[probability_bands["Probability Band"][i]]
            img = plt.imread(os.sep.join(("flood_tool", "imgs", img_map[flood_risk_array[rain_map[rain]][flood_p]])))
            # ax1.imshow(img)
            ax1.bar(daily_rainfall_df.index, height = daily_rainfall_df.values.ravel())
            ax1.set_title(closest_station['Station'][0] + "-" + postcodes[i], rotation='vertical', x=-0.1, y=0.5)
            plt.xticks(rotation = 45)
            ax2 = fig.add_subplot(int((len(postcodes) - 1)/2+1)*2,2,2*i+2)
            ax2.imshow(img)
            ax2.axis('off')
    prediction_df = pd.DataFrame({"Rain_Risk": rain_risk, "Property_Risk": probability_bands["Probability Band"]}, index=postcodes)
    prediction_df.index.name = "Postcode"
    prediction_df.Rain_Risk = prediction_df.Rain_Risk.fillna("Zero").astype(str)
    prediction_df.Property_Risk = prediction_df.Property_Risk.fillna("Zero").astype(str)
    prediction_df["Flood_Prediction"] = prediction_df.apply(lambda row : flood_risk_map[flood_risk_array[rain_map[row.Rain_Risk]][prop_map[row.Property_Risk]]], axis=1)
    print(prediction_df)
    if show_plot:
        plt.xticks(rotation = 45)
        plt.show()
    return prediction_df

def get_all_stations(limit=20):
    """Get first certain number of stations

    Parameters
    ----------

    limit : int
        the number of stations to return
    
    Retruns
    -------

    DataFrame
        A dataframe with columns
        Lat, Long, Station
    """
    url  =  'https://environment.data.gov.uk/flood-monitoring/id/stations?parameter=rainfall'
    api = requests.get(url).json()
    lat = []
    lon = []
    station = []
    for i in api["items"]:
        try:
            lat.append(i["lat"])
            lon.append(i["long"])
            station.append(i["notation"])
        except:
            pass
    zippedlist = list(zip(lat,lon,station))
    return pd.DataFrame(zippedlist, columns = ['Lat','Long','Station'])[:limit]

def get_station_mean(notation, input_date):
    """Get mean of the rainfall from READINGS_URL on a certain date

    Parameters
    ----------

    notation : str
        the id of the station
    input_date : str
        the date parameter, format: YYYY-MM-DD
    
    Retruns
    -------

    float
        the mean of the rainfall from READINGS_URL on the given date
    """
    try:
        url = "https://environment.data.gov.uk/flood-monitoring/id/stations/"+notation+\
        "/readings?date=" + input_date
        print(url)
        api = requests.get(url).json()

        rainfall_values = []
        for single_api in api['items']:
            rainfall_values.append(single_api['value'])
        station_mean = np.sum(rainfall_values) / len(rainfall_values)
        return station_mean
    except:
        return 0.0

def get_mean_class(mean_value):
    """Map mean value to risk level based on preset thresholds

    Parameters
    ----------

    mean_value : float
        the mean value of certain station
    
    Returns
    -------
    
    str
        the risk level
    """
    if mean_value < 0.2:
        return 'Zero'
    elif (mean_value >= 0.2) and (mean_value < 5):
        return 'Very Low'
    elif (mean_value >= 5) and (mean_value < 10):
        return 'Low'
    elif (mean_value >= 10) and (mean_value < 15):
        return 'Medium'
    elif mean_value >= 15:
        return 'High'
    else:
        return 'Zero'

def get_history_stations_rainfall(input_date, limit=20):
    """Get a visualized history rainfall data of a certain number of stations on a certain date.

    Parameters
    ----------

    input_date : str
        the date parameter, format: YYYY-MM-DD
    limit : int
        the number of stations to return
    
    Retruns
    -------

    DataFrameGroupBy
        A DataFrameGroupBy object with datetime as the key 
    """
    stations = get_all_stations(limit=limit)
    stations["mean_value"] = stations.Station.apply(lambda x : get_station_mean(x, input_date))
    stations["mean_class"] = stations.mean_value.apply(get_mean_class)
    stations_grouped = stations.groupby("mean_class")
    import mplleaflet
    color_map = {"High": 'ms', 'Medium': 'rs', 'Low': 'bs', 'Very Low': 'ys', 'Zero': 'gs'}
    for k, v in stations_grouped:
        plt.plot(v["Long"], v["Lat"], color_map[k])
    mplleaflet.show()
    return stations_grouped