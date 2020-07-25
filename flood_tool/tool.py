"""Locator functions to interact with geographic data"""
import numpy as np
import pandas as pd
import flood_tool.geo as geo

__all__ = ['Tool', 'trim_postcode']

def trim_postcode(pc):
    """Convert a dirty postcode to a valid one but remain the same if invalid.
    
    Parameters
    ----------

    pc : str
        input postcode
    
    Returns
    -------

    str
        A valid postcode or the invalid one.
    """
    pc2 = pc.strip().upper().replace(" ", "")
    if len(pc2) > 7 or len(pc2) < 5:
        return pc
    return pc2[:-3] + " "*(7-len(pc2)) + pc2[-3:]

class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file=None, risk_file=None, values_file=None):
        """

        Reads postcode and flood risk files and provides a postcode locator service.

        Parameters
        ---------

        postcode_file : str, optional
            Filename of a .csv file containing geographic location data for postcodes.
        risk_file : str, optional
            Filename of a .csv file containing flood risk data.
        postcode_file : str, optional
            Filename of a .csv file containing property value data for postcodes.
        """
        self.postcodes = pd.read_csv(postcode_file, delimiter=',', names = ['Postcode', 'Latitude', 'Longitude'])
        self.postcodes.Postcode = self.postcodes.Postcode.fillna(0).astype(str).apply(trim_postcode)
        self.probability = pd.read_csv(risk_file)
        self.property_value = pd.read_csv(values_file)
        self.property_value.Postcode = self.property_value.Postcode.fillna(0).astype(str).apply(trim_postcode)


    def get_lat_long(self, postcodes):
        """Get an array of WGS84 (latitude, longitude) pairs from a list of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered sequence of N postcode strings

        Returns
        -------
       
        ndarray
            Array of Nx2 (latitude, longitdue) pairs for the input postcodes.
            Invalid postcodes return [`numpy.nan`, `numpy.nan`].
        """
        postcodes = pd.Series(postcodes).fillna(0).astype(str).apply(trim_postcode)
        cord = self.postcodes.loc[self.postcodes['Postcode'].isin(postcodes)] #check if the postcodes are in the database
        cord = cord.set_index('Postcode').reindex(postcodes) #order the array by input order
        cord = cord.to_numpy(dtype='float64') #change boolean into values 
        return cord
        
    

    def get_easting_northing_flood_probability(self, easting, northing):
        """Get an array of flood risk probabilities from arrays of eastings and northings.

        Flood risk data is extracted from the Tool flood risk file. Locations
        not in a risk band circle return `Zero`, otherwise returns the name of the
        highest band it sits in.

        Parameters
        ----------

        easting: numpy.ndarray of floats
            OS Eastings of locations of interest
        northing: numpy.ndarray of floats
            Ordered sequence of postcodes

        Returns
        -------
       
        numpy.ndarray of strs
            numpy array of flood probability bands corresponding to input locations.
        """
        prob_dict = {"High": 4, "Medium":3, "Low":2, "Very Low":1}
        self.probability["prob_level"] = self.probability.prob_4band.map(prob_dict)
        prob_df_filtered = self.probability[self.probability.prob_level > 0]
        prob_dict_inverse = {4: "High", 3: "Medium", 2: "Low", 1: "Very Low", 0: "Zero"}
        points = np.stack([np.array(easting), np.array(northing)], axis=1).astype(float)
        result = []
        for point in points:
            flood_prob_filtered = prob_df_filtered[(prob_df_filtered.X >= point[0] - prob_df_filtered.radius) & (prob_df_filtered.X <= point[0] + prob_df_filtered.radius)]
            flood_prob_filtered = flood_prob_filtered[(flood_prob_filtered.Y >= point[1] - flood_prob_filtered.radius) & (flood_prob_filtered.Y <= point[1] + flood_prob_filtered.radius)]
            if len(flood_prob_filtered) == 0:
                result.append(prob_dict_inverse[0])
                continue
            circles = flood_prob_filtered[["X", "Y"]].values
            radius_square = flood_prob_filtered.radius.values**2
            new_point = np.repeat([point], len(circles), axis=0)
            mask = np.max(np.where(np.sum((new_point - circles)**2, axis=1) < radius_square, flood_prob_filtered.prob_level.values, 0))
            result.append(prob_dict_inverse[mask])
        return np.array(result)


    def get_sorted_flood_probability(self, postcodes):
        """Get an array of flood risk probabilities from a sequence of postcodes.

        Probability is ordered High>Medium>Low>Very low>Zero.
        Flood risk data is extracted from the `Tool` flood risk file. 

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered sequence of postcodes

        Returns
        -------
       
        pandas.DataFrame
            Dataframe of flood probabilities indexed by postcode and ordered from `High` to `Zero`,
            then by lexagraphic (dictionary) order on postcode. The index is named `Postcode`, the
            data column is named `Probability Band`. Invalid postcodes and duplicates
            are removed.
        """
        postcodes = pd.Series(postcodes).fillna(0).astype(str).apply(trim_postcode)
        prob_dict = {"High": 4, "Medium": 3, "Low": 2, "Very Low": 1, "Zero": 0}
        # lat_long = np.stack(self.get_lat_long(postcodes), axis=1)
        filtered_postcodes = self.postcodes.loc[self.postcodes['Postcode'].isin(postcodes)].drop_duplicates(subset="Postcode").set_index("Postcode")
        lat_long = np.stack(filtered_postcodes.to_numpy(dtype='float64'), axis=1)
        e_n = geo.get_easting_northing_from_lat_long(lat_long[0], lat_long[1])
        probs = self.get_easting_northing_flood_probability(e_n[0], e_n[1])
        result = pd.DataFrame({"Postcode": filtered_postcodes.index, "Probability Band": probs}).set_index("Postcode")
        result["prob_level"] = result["Probability Band"].map(prob_dict)
        result = result.sort_values(["prob_level", "Postcode"], ascending=[False, True])
        return result[["Probability Band"]]

    def get_flood_cost(self, postcodes):
        """Get an array of estimated cost of a flood event from a sequence of postcodes.
        Parameters
        ----------

        postcodes: sequence of strs
            Ordered collection of postcodes

        Returns
        -------
       
        numpy.ndarray of floats
            array of floats for the pound sterling cost for the input postcodes.
            Invalid postcodes return `numpy.nan`.
        """
        postcodes = pd.Series(postcodes).fillna(0).astype(str).apply(trim_postcode)
        cord = self.property_value.loc[self.property_value['Postcode'].isin(postcodes)] #check if the postcodes are in the database
        cord = cord.set_index('Postcode').reindex(postcodes) #order the array by input order
        total_value = cord['Total Value'] #change boolean into values 
        total_value = total_value.fillna(0).to_numpy(dtype='float64')
        return total_value

    def get_annual_flood_risk(self, postcodes, probability_bands):
        """Get an array of estimated annual flood risk in pounds sterling per year of a flood
        event from a sequence of postcodes and flood probabilities.

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered collection of postcodes
        probability_bands: sequence of strs
            Ordered collection of flood probabilities

        Returns
        -------
       
        numpy.ndarray
            array of floats for the annual flood risk in pounds sterling for the input postcodes.
            Invalid postcodes return `numpy.nan`.
        """
        postcodes = pd.Series(postcodes).fillna(0).astype(str).apply(trim_postcode)
        iptframe = pd.DataFrame({'postcode':postcodes, 'Probability Band':probability_bands})
        dic = {'High':0.1, 'Medium':0.02, 'Low':0.01, 'Very Low':0.001, 'Zero':0, 'No Risk':0}
        iptframe['probability'] = iptframe['Probability Band'].map(dic)
        #init input
        flood_cost = self.get_flood_cost(postcodes) * 0.05
        risk_level = iptframe['probability'].values.astype(float)
        risk_value = risk_level * flood_cost
        return risk_value

    def get_sorted_annual_flood_risk(self, postcodes):
        """Get a sorted pandas DataFrame of flood risks.

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered sequence of postcodes

        Returns
        -------
       
        pandas.DataFrame
            Dataframe of flood risks indexed by (normalized) postcode and ordered by risk,
            then by lexagraphic (dictionary) order on the postcode. The index is named
            `Postcode` and the data column `Flood Risk`.
            Invalid postcodes and duplicates are removed.
        """
        postcodes = pd.Series(postcodes).fillna(0).astype(str).apply(trim_postcode)
        filtered_postcodes = self.postcodes.loc[self.postcodes['Postcode'].isin(postcodes)].drop_duplicates(subset="Postcode").set_index("Postcode")
        lat_long = np.stack(filtered_postcodes.to_numpy(dtype='float64'), axis=1)
        e_n = geo.get_easting_northing_from_lat_long(lat_long[0], lat_long[1])
        probs = self.get_easting_northing_flood_probability(e_n[0], e_n[1])
        flood_cost =  self.property_value.loc[self.property_value['Postcode'].isin(filtered_postcodes.index)].set_index('Postcode').reindex(filtered_postcodes.index)
        cost_rate = {'High':0.1, 'Medium':0.02, 'Low':0.01, 'Very Low':0.001, 'Zero':0, 'No Risk':0}
        flood_cost["prob_level"] = pd.Series(probs, index=filtered_postcodes.index).map(cost_rate)
        flood_cost["Flood Risk"] = flood_cost["Total Value"] * 0.05 * flood_cost.prob_level
        flood_cost = flood_cost.fillna(0.0).sort_values(["Flood Risk", "Postcode"], ascending=[False, True])
        return flood_cost[["Flood Risk"]]
