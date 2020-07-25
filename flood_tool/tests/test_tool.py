import numpy as np
import pandas as pd
import pytest
import os

import flood_tool.tool as Tool

postcode_file = os.sep.join(("flood_tool", "resources", "postcodes.csv"))
risk_file = os.sep.join(("flood_tool", "resources", "flood_probability.csv"))
values_file = os.sep.join(("flood_tool", "resources", "property_value.csv"))


@pytest.mark.lat_long
def test_lat_long():
    """Test the function get_lat_long in tool.py"""
    test_data = pd.read_csv(os.sep.join(("flood_tool", "tests", "test_data.csv")))
    test_data = test_data.astype({'Latitude': 'float64', 'Longitude': 'float64'})
    flood_Tool = Tool.Tool(postcode_file, risk_file, values_file)
    test_output = test_data[['Latitude', 'Longitude']].values
    o2 = flood_Tool.get_lat_long(test_data.Postcode)

@pytest.mark.prob
def test_flood_prob():
    """Test the function get_sorted_flood_probability in tool.py"""
    test_data = pd.read_csv(os.sep.join(("flood_tool", "tests", "test_data.csv")))
    flood_Tool = Tool.Tool(postcode_file, risk_file, values_file)
    assert flood_Tool.get_sorted_flood_probability(test_data.Postcode) == test_data.sort_values(['Postcode'])['Probability Band'].values

@pytest.mark.cost
def test_flood_cost():
    """Test the function get_flood_cost in tool.py"""
    test_data = pd.read_csv(os.sep.join(("flood_tool", "tests", "test_data.csv"))) #route
    flood_Tool = Tool.Tool(postcode_file, risk_file, values_file) #init input
    print(flood_Tool.get_flood_cost(test_data.Postcode))
    print(test_data['Total Value'].values*0.05)
    assert (flood_Tool.get_flood_cost(test_data.Postcode) == test_data['Total Value'].values*0.05).all()
    

