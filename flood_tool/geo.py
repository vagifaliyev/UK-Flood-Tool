"""Module implementing various geodetic transformation functions."""

from numpy import array, sin, cos, tan, sqrt, pi, arctan2, floor
import numpy as np

__all__ = ['get_easting_northing_from_lat_long',
           'WGS84toOSGB36']

class Ellipsoid(object):
    """Class acting as container for properties describing a terrestrial ellipsoid."""
    def __init__(self, a, b, F_0):
        self.a = a
        self.b = b
        self.n = (a-b)/(a+b)
        self.e2 = (a**2-b**2)/a**2
        self.F_0 = F_0
        self.H=0

class Datum(Ellipsoid):
    """Class acting as container for properties describing a map datum."""

    def __init__(self, a, b, F_0, phi_0, lam_0, E_0, N_0, H):
        super().__init__(a, b, F_0)
        self.phi_0 = phi_0
        self.lam_0 = lam_0
        self.E_0 = E_0
        self.N_0 = N_0
        self.H = H

def rad(deg, min=0, sec=0):
    """Convert degrees into radians."""
    return (deg+min/60.+sec/3600.)*(pi/180.)

def deg(rad, dms=False):
    """Convert radians into degrees."""
    d = rad*(180./pi)
    if dms:
        m = 60.0*(d%1.)
        return floor(d),  floor(m), round(60*(m%1.),4)
    else:
        return d

# Datum for the Ordenance Survey GB 1936 Datum, as used in the OS
# National grid

osgb36 = Datum(a = 6377563.396,
               b = 6356256.910,
               F_0 = 0.9996012717,
               phi_0 = rad(49.0),
               lam_0 = rad(-2.),
               E_0 = 400000,
               N_0 = -100000,
               H=24.7)

# Ellipsoid used for the WGS 1984 datum (i.e. for GPS coordinates)

wgs84 = Ellipsoid(a = 6378137, 
                  b = 6356752.3142,
                  F_0 = 0.9996)

def lat_long_to_xyz(latitude, longitude, radians=False, datum=osgb36):
    """Convert locations in latitude and longitude format to 3D on specified datum.

    Input arrays must be of matching length.
    
    Parameters
    ----------

    latitude: numpy.ndarray of floats
        latitudes to convert
    latitude: numpy.ndarray of floats
        latitudes to convert
    radians: bool, optional
        True if input is in radians, otherwise degrees assumed.
    datum: geo.Ellipsoid, optional
        Geodetic ellipsoid to work on

    Returns
    -------

    numpy.ndarray
        Locations in 3D (body Cartesian) coordinates.
    """

    if not radians:
        latitude = rad(latitude)
        longitude = rad(longitude)

    nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(latitude)**2)
  
    return array(((nu+datum.H)*cos(latitude)*cos(longitude),
                  (nu+datum.H)*cos(latitude)*sin(longitude),
                  ((1-datum.e2)*nu+datum.H)*sin(latitude)))

def xyz_to_lat_long(x, y, z, radians=False, datum=osgb36):
    """Convert locations in 3D to latitude longitude format on specified datum.

    Input arrays must be of matching length.
    
    Parameters
    ----------

    x: numpy.ndarray of floats
        x coordinate in body Cartesian 3D
    y: numpy.ndarray of floats
        z coordinate in body Cartesian 3D
    z: numpy.ndarray of floats
        z coordinate in body Cartesian 3D
    radians: bool, optional
        True if output should be in radians, otherwise degrees assumed.
    datum: geo.Ellipsoid, optional
        Geodetic ellipsoid to work on

    Returns
    -------

    latitude: numpy.ndarray
        Locations latitudes.
    longitude: numpy.ndarray
        Locations longitudes.
    """
    p = sqrt(x**2+y**2)

    ### invert for longitude
    longitude = arctan2(y, x)

    ### first guess at longitude
    latitude = arctan2(z,p*(1-datum.e2))

    ### Apply a few iterations of Newton Rapheson
    for _ in range(6):
        nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(latitude)**2)
        dnu = -datum.a*datum.F_0*cos(latitude)*sin(latitude)/(1-datum.e2*sin(latitude)**2)**1.5

        f0 = (z + datum.e2*nu*sin(latitude))/p - tan(latitude)
        f1 = datum.e2*(nu**cos(latitude)+dnu*sin(latitude))/p - 1.0/cos(latitude)**2
        latitude -= f0/f1

    if not radians:
        latitude = deg(latitude)
        longitude = deg(longitude)

    return latitude, longitude

class HelmertTransform(object):
    """Class to perform a Helmert Transform mapping (x,y,z) tuples from one datum to another.""" 
    
    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3,1))
        
        self.M = array([[1+s, -rz, ry],
                        [rz, 1+s, -rx],
                        [-ry, rx, 1+s]])

    def __call__(self, X):
        """ Transform a point or point set using the Helmert Transform."""
        return self.T + self.M.dot(X.reshape((3,-1)))

WGS84toOSGB36transform = HelmertTransform(20.4894e-6,
                             -rad(0,0,0.1502),
                             -rad(0,0,0.2470),
                             -rad(0,0,0.8421),
                             array([-446.448, 125.157, -542.060]))


def WGS84toOSGB36(latitude, longitude, radians=False):
    """ Wrapper to transform (latitude, longitude) pairs
    from GPS to OS datum."""
#    latlon in wgs to xyz in wgs
#    xyz in wgs to xzy in osgb36
#    xyz in osgb36 to latlon in osgb36
#    latitude = latitude
#    longitude = longitude
    
    lat_xyz = lat_long_to_xyz(latitude, longitude, radians, datum=wgs84)
    wgs_gb = WGS84toOSGB36transform(lat_xyz)
    out = xyz_to_lat_long(wgs_gb[0],wgs_gb[1],wgs_gb[2], True, datum=osgb36)
    
    return out
#    raise NotImplementedError    


# print(WGS84toOSGB36(np.array([rad(52, 39, 28.71)]),
#                             np.array( [rad(1, 42, 57.79)])))
# print(np.array([[rad(52, 39, 27.2531)],
#                             [rad(1, 43, 4.5177)]]))

def get_easting_northing_from_lat_long(latitude, longitude, radians=False):
    """ Convert GPS (latitude, longitude) to OS (easting, northing).
    
    Parameters
    ----------
    latitude : sequence of floats
               Latitudes to convert.
    longitude : sequence of floats
                Lonitudes to convert.
    radians : bool, optional
              Set to `True` if input is in radians. Otherwise degrees are assumed
    
    Returns
    -------

    easting : ndarray of floats
              OS Eastings of input
    northing : ndarray of floats
              OS Northings of input

    References
    ----------

    A guide to coordinate systems in Great Britain 
    (https://webarchive.nationalarchives.gov.uk/20081023180830/http://www.ordnancesurvey.co.uk/oswebsite/gps/information/coordinatesystemsinfo/guidecontents/index.html)
    """ 
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    lat, lon = WGS84toOSGB36(latitude, longitude, radians)

    a = osgb36.a
    b = osgb36.b
    F_0 = osgb36.F_0
    phi_0 = osgb36.phi_0
    lam_0 = osgb36.lam_0
    N_o = osgb36.N_0
    E_o = osgb36.E_0

    n = osgb36.n
    e2 = osgb36.e2

    v = a*F_0*(1-e2*sin(lat)**2)**(-0.5)
    rho = a*F_0*(1-e2)*(1-e2*sin(lat)**2)**(-1.5)
    #Calculate mu
    mu2 = (v/rho)-1
    M = b*F_0*((1+n+(5/4)*n**2+(5/4)*n**3)*(lat-phi_0)-(3*n+3*n**2+(21/8)*n**3)*sin(lat - phi_0)*cos(lat + phi_0)+((15/8)*n**2+(15/8)*n**3)*sin(2*(lat - phi_0))*cos(2*(lat + phi_0))-(35/24)*n**3*sin(3*(lat - phi_0))*cos(3*(lat + phi_0)))
    I = M + N_o
    II = (v/2)*sin(lat)*cos(lat)
    III = (v/24)*sin(lat)*(cos(lat)**3)*(5 -((tan(lat)**2)+9*mu2**2))
    IIIA = (v/720)*sin(lat)*cos(lat)**5*(61- (58*tan(lat)**2)+tan(lat)**4)
    IV = v*cos(lat)
    V = (v/6)*cos(lat)**3*((v/rho)-tan(lat)**2)
    VI = (v/120)*(cos(lat)**5)*(5-(18*tan(lat)**2)+tan(lat)**4+14*mu2-58*(tan(lat)**2)*mu2)
    E = E_o+IV*(lon-lam_0)+V*(lon-lam_0)**3+VI*(lon-lam_0)**5
    N = I + II*(lon-lam_0)**2 + III*(lon-lam_0)**4+IIIA*(lon-lam_0)**6
    return E, N

