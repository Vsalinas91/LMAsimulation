######################################################
# Various functions for LMA simulations
#
# The black_box function is the main driver function
#
# Contact:
# vanna.chmielewski@noaa.gov
#
#
# This model and its results have been submitted to the Journal of Geophysical Research.
# Please cite:
# V. C. Chmielewski and E. C. Bruning (2016), Lightning Mapping Array flash detection
# performance with variable receiver thresholds, J. Geophys. Res. Atmos., 121, 8600-8614,
# doi:10.1002/2016JD025159
#
# If any results from this model are presented.
######################################################

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import leastsq
from coordinateSystems import GeographicSystem
try:
    import cartopy.crs as ccrs
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import cartopy.geodesic as geodesic
    from shapely.geometry import Point
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import warnings
    warnings.filterwarnings("ignore")
    e=None
except ImportError as e:
    print('And older version of python is being used and cartopy cannot be imported. Defualting to Basemap for plotting.')
    from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def travel_time(X, X_ctr, c, t0=0.0, get_r=False):
    """ Units are meters, seconds.
        X is a (N,3) vector of source locations, and X_ctr is (N,3) receiver
        locations.

        t0 may be used to specify a fixed amount of additonal time to be
        added to the travel time.

        If get_r is True, return the travel time *and* calculated range.
        Otherwise
        just return the travel time.
    """
    ranges = (np.sum((X[:, np.newaxis] - X_ctr)**2, axis=2)**0.5).T
    time = t0+ranges/c
    if get_r == True:
        return time, ranges
    if get_r == False:
        return time


def received_power(power_emit, r, wavelength=4.762, recv_gain=1.0):
    """ Calculate the free space loss. Defaults are for no receive gain and
        a 63 MHz wave.
        Units: watts, meters
    """
    four_pi_r = 4.0 * np.pi * r
    free_space_loss = (wavelength*wavelength) / (four_pi_r * four_pi_r)
    return recv_gain * power_emit * free_space_loss


def precalc_station_terms(stations_ECEF):
    """ Given a (N_stn, 3) array of station locations,
        return dxvec and drsq which are (N, N, 3) and (N, N),
        respectively, where the first dimension is the solution
        for a different master station
    """
    dxvec = stations_ECEF-stations_ECEF[:, np.newaxis]
    r1    = (np.sum(stations_ECEF**2, axis=1))
    drsq  = r1-r1[:, np.newaxis]
    return dxvec, drsq


def linear_first_guess(t_i, dxvec, drsq, c=3.0e8):
    """ Given a vector of (N_stn) arrival times,
        calcluate a first-guess solution for the source location.
        Return f=(x, y, z, t), the source retrieval locations.
    """
    g = 0.5*(drsq - (c**2)*(t_i**2))
    K = np.vstack((dxvec.T, -c*t_i)).T
    f, residuals, rank, singular = lstsq(K, g)
    return f


def predict(p, stations_ECEF2, c=3.0e8):
    """ Predict arrival times for a particular fit of x,y,z,t source
        locations. p are the parameters to retrieve
    """
    return (p[3] + ((np.sum((p[:3] - stations_ECEF2) *
            (p[:3] - stations_ECEF2), axis=1))**0.5)) / c


def residuals(p, t_i, stations_ECEF2):
    return t_i - predict(p, stations_ECEF2)


def dfunc(p, t_i, stations_ECEF2, c=3.0e8):
    return -np.vstack(((p[:3] - stations_ECEF2).T / (c *
              (np.sum((p[:3] - stations_ECEF2) *
              (p[:3] - stations_ECEF2), axis=1))**0.5),
             np.array([1./c]*np.shape(stations_ECEF2)[0])))


def gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq, center_ECEF,
                       stations_ECEF, dt_rms, min_stations=5,
                       max_z_guess=25.0e3):
    """ t_all is a N_stations x N_points masked array of arrival times at
        each station.
        t_min is an N-point array of the index of the first unmasked station
        to receive a signal

        center_ECEF for the altitude check

        This streamlines the generator function, which emits a stream of
        nonlinear least-squares solutions.
    """
    m = t_mins[i]
    stations_ECEF2=stations_ECEF[selection]
    # Make a linear first guess
    p0 = linear_first_guess(np.array(t_all[:,i][selection]-t_all[m,i]),
                            dxvec[m][selection],
                            drsq[m][selection])
    t_i =t_all[:,i][selection]-t_all[m,i]
    # Checking altitude in lat/lon/alt from local coordinates
    latlon = np.array(GeographicSystem().fromECEF(p0[0], p0[1],p0[2]))
    if (latlon[2]<0) | (latlon[2]>25000):
        latlon[2] = 7000
        new = GeographicSystem().toECEF(latlon[0], latlon[1], latlon[2])
        p0[:3]=np.array(new)
    plsq = np.array([np.nan]*5)
    plsq[:4], cov, infodict, mesg,ier = leastsq(residuals, p0,
                            args=(t_i, stations_ECEF2),
                            Dfun=dfunc,col_deriv=1,full_output=True)
    plsq[4] = np.sum(infodict['fvec']*infodict['fvec'])/(
                     dt_rms*dt_rms*(float(np.shape(stations_ECEF2)[0]-4)))
    return plsq

def gen_retrieval(t_all, t_mins, dxvec, drsq, center_ECEF, stations_ECEF,
                  dt_rms, min_stations=5, max_z_guess=25.0e3):
    """ t_all is a N_stations x N_points masked array of arrival times at
        each station.
        t_min is an N-point array of the index of the first unmasked station
        to receive a signal

        center_ECEF for the altitude check

        This is a generator function, which emits a stream of nonlinear
        least-squares solutions.
    """
    for i in range(t_all.shape[1]):
        selection=~np.ma.getmaskarray(t_all[:,i])
        if np.all(selection == True):
            selection = np.array([True]*len(t_all[:,i]))
            yield gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq,
                                     center_ECEF, stations_ECEF, dt_rms,
                                     min_stations, max_z_guess=25.0e3)
        elif np.sum(selection)>=min_stations:
            yield gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq,
                                     center_ECEF, stations_ECEF, dt_rms,
                                     min_stations, max_z_guess=25.0e3)
        else:
            yield np.array([np.nan]*5)

def gen_retrieval_full(t_all, t_mins, dxvec, drsq, center_ECEF, stations_ECEF,
                       dt_rms, c0, min_stations=5, max_z_guess=25.0e3):
    """ t_all is a N_stations x N_points masked array of arrival times at
        each station.
        t_min is an N-point array of the index of the first unmasked station
        to receive a signal

        center_ECEF for the altitude check

        This is a generator function, which emits a stream of nonlinear
        least-squares solutions.

        Timing comes out of least-squares function as t*c from the initial
        station
    """
    for i in range(t_all.shape[1]):
        selection=~np.ma.getmaskarray(t_all[:,i])
        plsq = np.array([np.nan]*7)
        if np.all(selection == True):
            selection = np.array([True]*len(t_all[:,i]))
            plsq[:5] = gen_retrieval_math(i, selection, t_all, t_mins, dxvec,
                         drsq, center_ECEF, stations_ECEF, dt_rms,
                         min_stations, max_z_guess=25.0e3)
            plsq[5] = plsq[3]/c0 + t_all[t_mins[i],i]
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq
        elif np.sum(selection)>=min_stations:
            plsq[:5] = gen_retrieval_math(i, selection, t_all, t_mins, dxvec,
                                     drsq, center_ECEF, stations_ECEF, dt_rms,
                                     min_stations, max_z_guess=25.0e3)
            plsq[5] = plsq[3]/c0 + t_all[t_mins[i],i]
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq
        else:
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq


def array_from_generator2(generator, rows):
    """Creates a numpy array from a specified number
    of values from the generator provided."""
    data = []
    for row in range(rows):
        try:
            data.append(next(generator))
        except StopIteration:
            break
    return np.array(data)

def black_box(x,y,z,n,
              stations_local,ordered_threshs,stations_ecef,center_ecef,
              tanps,
              c0,dt_rms,tanp,projl,chi2_filter,min_stations=5,just_rms=False):
    """ This funtion incorporates most of the Monte Carlo functions and calls
        into one big block of code.

        x,y,z are the source location in the local tangent plane (m)
        n is the number of iterations

        stations_local is the the (N-stations, 3) array of station locations
        in the local tangent plane

        ordered_threshs is the N-station array of thresholds in the same order
        as the station arrays (in dBm).

        stations_ecef is (N,3) array in ECEF coordinates

        center_ecef is just the center location of the network, just easier
        to pass into the fuction separately to save some calculation

        c0 is th speed of light

        dt_rms is the standard deviation of the timing error (Gaussian, in s)

        tanp is the tangent plane object

        projl is the map projection object

        chi2_filter is the maximum allowed reduced chi2 filter for the
        calculation (use at most 5)

        min_stations is the minimum number of stations required to receive
        a source. This must be at least 5, can be higher to filter out more
        poor solutions

        If just_rms is True then only the rms error will be returned in a
        (x,y,z) array.
        If false then mean error (r,z,theta), standard deviation (r,z,theta)
        both in units of (m,m,degrees)
        and the number of bad/missed solutions will be returned.
    """
    points = np.array([np.zeros(n)+x, np.zeros(n)+y, np.zeros(n)+z]).T
    powers = np.empty(n)

    # # For the old 1/p distribution:
    # powers = np.random.power(2, size=len(points[:,0]))**-2

    # # For high powered sources (all stations contributing):
    # powers[:] = 10000

    # For the theoretical distribution:
    for i in range(len(powers)):
        powers[i] = np.max(1./np.random.uniform(0,1000,2000))

    # Calculate distance and power retrieved at each station and mask
    # the stations which have higher thresholds than the retrieved power

    points_f_ecef = (tanp.fromLocal(points.T)).T
    dt, ran  = travel_time(points, stations_local, c0, get_r=True)
    pwr = received_power(powers, ran)
    masking = 10.*np.log10(pwr/1e-3) < ordered_threshs[:,np.newaxis]
    masking2 = np.empty_like(masking)
    for i in range(len(stations_ecef[:,0])):
        masking2[i] = tanps[i].toLocal(points_f_ecef.T)[2]<0

    masking = masking | masking2
    pwr = np.ma.masked_where(masking, pwr)
    dt  = np.ma.masked_where(masking, dt)
    ran = np.ma.masked_where(masking, ran)

    # Add error to the retreived times
    dt_e = dt + np.random.normal(scale=dt_rms, size=np.shape(dt))
    dt_mins = np.argmin(dt_e, axis=0)
    # Precalculate some terms in ecef (fastest calculation)
    points_f_ecef = (tanp.fromLocal(points.T)).T
    full_dxvec, full_drsq = precalc_station_terms(stations_ecef)
    # Run the retrieved locations calculation
    # gen_retrieval returns a tuple of four positions, x,y,z,t.
    dtype=[('x', float), ('y', float), ('z', float), ('t', float),
           ('chi2', float)]
    # Prime the generator function - pauses at the first yield statement.
    point_gen = gen_retrieval(dt_e, dt_mins, full_dxvec, full_drsq,
                              center_ecef, stations_ecef, dt_rms,
                              min_stations)
    # Suck up the values produced by the generator, produce named array.
    # retrieved_locations = np.fromiter(point_gen, dtype=dtype)
    # retrieved_locations = np.array([(a,b,c,e) for (a,b,c,d,e) in
    #                                  retrieved_locations])
    retrieved_locations = array_from_generator2(point_gen,rows=n)
    retrieved_locations = retrieved_locations[:,[0,1,2,-1]]
    chi2                = retrieved_locations[:,3]
    retrieved_locations = retrieved_locations[:,:3]
    retrieved_locations = np.ma.masked_invalid(retrieved_locations)
    if just_rms == False:
        # Converts to projection
        # soluts = tanp.toLocal(retrieved_locations.T)
        # good   = soluts[2] > 0
        proj_soluts = projl.fromECEF(retrieved_locations[:,0],
                                     retrieved_locations[:,1],
                                     retrieved_locations[:,2])
        good = proj_soluts[2] > 0
        proj_soluts = (proj_soluts[0][good],proj_soluts[1][good],
                       proj_soluts[2][good])
        proj_points = projl.fromECEF(points_f_ecef[good,0],
                                     points_f_ecef[good,1],
                                     points_f_ecef[good,2])

        proj_soluts = np.ma.masked_invalid(proj_soluts)
        # Converts to cylindrical coordinates since most errors
        # are in r and z, not theta
        proj_points_cyl = np.array([(proj_points[0]**2+proj_points[1]**2)**0.5,
                              np.degrees(np.arctan(proj_points[0]/proj_points[1])),
                              proj_points[2]])
        proj_soluts_cyl = np.ma.masked_array([(proj_soluts[1]**2+proj_soluts[0]**2)**0.5,
                              np.degrees(np.arctan(proj_soluts[0]/proj_soluts[1])),
                              proj_soluts[2]])
        difs = proj_soluts_cyl - proj_points_cyl
        difs[1][difs[1]>150]=difs[1][difs[1]>150]-180
        difs[1][difs[1]<-150]=difs[1][difs[1]<-150]+180
        return np.mean(difs.T[chi2[good]<chi2_filter].T, axis=1
             ), np.std(difs.T[chi2[good]<chi2_filter].T, axis=1
             ), np.ma.count_masked(difs[0])+np.sum(chi2[good]>=chi2_filter
             )+np.sum(~good)
    else:
        #Convert back to local tangent plane
        soluts = tanp.toLocal(retrieved_locations.T)
        proj_soluts = projl.fromECEF(retrieved_locations[:,0],
                                     retrieved_locations[:,1],
                                     retrieved_locations[:,2])
        good = proj_soluts[2] > 0
        # good   = soluts[2] > 0
        difs   = soluts[:,good] - points[good].T
        return np.mean((difs.T[chi2[good]<chi2_filter].T)**2, axis=1)**0.5


def black_box_full(x,y,z,n,
              stations_local,ordered_threshs,stations_ecef,center_ecef,
              c0,dt_rms,tanp,projl,chi2_filter,min_stations=5):
    """ This funtion incorporates most of the Monte Carlo functions and calls
        into one big block of code with other non-standard output. Otherwise
        performs the same basic fuctions as black_box.

        x,y,z are the source location in the local tangent plane (m)
        n is the number of iterations

        stations_local is the (N-stations, 3) array of station locations
        in the local tangent plane

        ordered_threshs is the N-station array of thresholds in the same order
        as the station arrays (in dBm).

        stations_ecef is (N,3) array in ECEF coordinates

        center_ecef is just the center location of the network, just easier
        to pass into the fuction separately to save some calculation

        c0 is th speed of light

        dt_rms is the standard deviation of the timing error (Gaussian, in s)

        tanp is the tangent plane object

        projl is the map projection object

        chi2_filter is the maximum allowed reduced chi2 filter for the
        calculation (use at most 5)

        min_stations is the minimum number of stations required to receive
        a source. This must be at least 5, can be higher to filter out more
        poor solutions

        If just_rms is True then only the rms error will be returned in a
        (x,y,z) array (in m).
        If false then mean error (r,z,theta), standard deviation (r,z,theta)
        both in units of (m,m,degrees) and the number of bad/missed solutions
        will be returned.
    """
    # Creates an array of poins at x,y,z of n elements
    points = np.array([np.zeros(n)+x, np.zeros(n)+y, np.zeros(n)+z]).T
    powers = np.empty(n)
    for i in range(len(powers)):
        powers[i] = np.max(1./np.random.uniform(0,1000,2000))
    # Calculates distance and powers retrieved at each station to threshold
    dt, ran  = travel_time(points, stations_local, c0, get_r=True)
    pwr = received_power(powers, ran)
    masking = 10.*np.log10(pwr/1e-3) < ordered_threshs[:,np.newaxis]
    pwr = np.ma.masked_where(masking, pwr)
    dt  = np.ma.masked_where(masking, dt)
    ran = np.ma.masked_where(masking, ran)
    # Adds error to the retreived times
    dt_e = dt + np.random.normal(scale=dt_rms, size=np.shape(dt))
    dt_mins = np.argmin(dt_e, axis=0)
    # Precalculate some terms
    points_f_ecef = (tanp.fromLocal(points.T)).T
    full_dxvec, full_drsq = precalc_station_terms(stations_ecef)
    # Run the retrieved locations calculation
    # gen_retrieval returns a tuple of four positions, x,y,z,t.
    dtype=[('x', float), ('y', float), ('z', float),
           ('t', float), ('chi2', float), ('terror',float),
           ('stations', float)]
    # Prime the generator function - it pauses at the first yield statement.
    point_gen = gen_retrieval_full(dt_e, dt_mins, full_dxvec, full_drsq,
                                   center_ecef, stations_ecef, dt_rms,
                                   min_stations)
    # Suck up all the values produced by the generator, produce named array.
    # retrieved_locations = np.fromiter(point_gen, dtype=dtype)
    # retrieved_locations = np.array([(a,b,c,e,f,g) for (a,b,c,d,e,f,g) in
    #                                retrieved_locations])
    retrieved_locations = array_from_generator2(point_gen,rows=n)
    retrieved_locations = retrieved_locations[:,[0,1,2,4,5,6]]
    station_count       = retrieved_locations[:,5]
    terror              = retrieved_locations[:,4]
    chi2                = retrieved_locations[:,3]
    retrieved_locations = retrieved_locations[:,:3]
    retrieved_locations = np.ma.masked_invalid(retrieved_locations)
    # Converts to projection
    soluts = tanp.toLocal(retrieved_locations.T)
    good   = soluts[2] > 0
    station_count[~good] = np.nan
    # proj_points = projl.fromECEF(points_f_ecef[good,0],
    #                              points_f_ecef[good,1],
    #                              points_f_ecef[good,2])
    # proj_soluts = projl.fromECEF(retrieved_locations[good,0],
    #                              retrieved_locations[good,1],
    #                              retrieved_locations[good,2])
    # proj_soluts = np.ma.masked_invalid(proj_soluts)
    # ranges = (np.sum((retrieved_locations[good,:3][:,np.newaxis] -
    #                   stations_ecef)**2, axis=2)**0.5).T
    # cal_pwr = (np.mean(pwr*(4*np.pi*ranges/4.762)**2, axis=0))
    # return proj_points, proj_soluts, chi2[good], terror[good], station_count[good]
    # return powers[good], cal_pwr, pwr[0][good]
    # return station_count

def curvature_matrix(points,stations_local,ordered_threshs,c0,power,
                     timing_error,min_stations=5):
    """ Calculates the curvature matrix error solutions given stations and
        grid points of sources. Returns array of (x,y,z,ct) errors (in m)

        points is the n by 3 array of source points in x,y,z local tangent
        plane coordinates (m)

        stations_local is the (N-stations, 3) array of station locations
        in the local tangent plane.

        ordered_threshs is the N-station array of thresholds in the same order
        as the station arrays (in dBm).

        c0 is the speed of light

        power is the source power for all of the points in Watts

        timing_error is the Gaussian standard deviation of timing errors of
        the stations in seconds
    """
    # Find received powers to find which stations contibute for each source
    dist = np.sum((points - stations_local)**2,axis=1)**0.5
    test_power = received_power(power,dist)
    masking = 10.*np.log10(test_power/1e-3) < ordered_threshs
    # Precalculate some terms to simplify the calculations for each term
    ut1 = -dist[~masking]
    ut2 = stations_local[~masking] - points
    dist_term = dist[~masking]*dist[~masking]
    # Find each term of the curvature matrix
    if np.sum(~masking)>=min_stations:
        curvature_matrix = np.empty((4,4))
        curvature_matrix[0,0] = np.sum((ut2[:,1]**2 + ut2[:,2]**2)*
                                        dist_term**(-3./2.)*ut1+1)
        curvature_matrix[1,1] = np.sum((ut2[:,2]**2 + ut2[:,0]**2)*
                                        dist_term**(-3./2.)*ut1+1)
        curvature_matrix[2,2] = np.sum((ut2[:,1]**2 + ut2[:,0]**2)*
                                        dist_term**(-3./2.)*ut1+1)
        curvature_matrix[3,3] = np.sum(~masking)
        curvature_matrix[0,1] = np.sum(-ut1*ut2[:,0]*ut2[:,1]*
                                       dist_term**(-3./2.))
        curvature_matrix[0,2] = np.sum(-ut1*ut2[:,0]*ut2[:,2]*
                                       dist_term**(-3./2.))
        curvature_matrix[1,2] = np.sum(-ut1*ut2[:,1]*ut2[:,2]*
                                       dist_term**(-3./2.))
        curvature_matrix[3,0] = np.sum(-ut2[:,0]*dist_term**(-1./2.))
        curvature_matrix[3,2] = np.sum(-ut2[:,2]*dist_term**(-1./2.))
        curvature_matrix[3,1] = np.sum(-ut2[:,1]*dist_term**(-1./2.))
        curvature_matrix[1,0] = curvature_matrix[0,1]
        curvature_matrix[2,0] = curvature_matrix[0,2]
        curvature_matrix[2,1] = curvature_matrix[1,2]
        curvature_matrix[0,3] = curvature_matrix[3,0]
        curvature_matrix[2,3] = curvature_matrix[3,2]
        curvature_matrix[1,3] = curvature_matrix[3,1]
        # Find the error terms from the matrix solution
        errors = (np.linalg.inv(curvature_matrix/
                 (c0*c0*timing_error*timing_error)))**0.5
        return np.array([errors[0,0],errors[1,1],errors[2,2],errors[3,3]])
    else: return np.array([np.nan,np.nan,np.nan,np.nan])

#Below plotting routines are chosen based on whether Basemap is available or not.
#Typicall tied to which python version is being used to run the scripts/notebooks,
#python <= 2.x will have basemap, python <= 3.x will not.
if e is not None:
    #Running plotting routines on basemap for older python versions
    def mapped_plot(plot_this,from_this,to_this,with_this,dont_forget,
                    xmin,xmax,xint,ymin,ymax,yint,location):
        """Make plots in map projection with km color scales, requires input
           array, lower color scale limit, higher color scale limit, color scale,
           station location overlay, and array x-,y- min, max and interval, the
           location of the array center (in lat and lon). Output plot also
           contains 100, 200 km range rings
        """
        domain = (xmax-xint/2.)
        maps = Basemap(projection='laea',lat_0=location[0],lon_0=location[1],
                       width=domain*2,height=domain*2)
        s = plt.pcolormesh(np.arange(xmin-xint/2.,xmax+3*xint/2.,xint)+domain,
                           np.arange(ymin-yint/2.,ymax+3*yint/2.,yint)+domain,
                           plot_this, cmap = with_this)
        s.set_clim(vmin=from_this,vmax=to_this)
        plt.colorbar()
        plt.scatter(dont_forget[:,0]+domain, dont_forget[:,1]+domain, color='k')
        maps.drawstates()
        circle=plt.Circle((domain,domain),100000,color='k',fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circle)
        circle=plt.Circle((domain,domain),200000,color='k',fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circle)


    def nice_plot(data,xmin,xmax,xint,centerlat,centerlon,stations,color,cmin,
                  cmax,levels_t):
        """Make plots in map projection, requires input array, x-min max and
           interval (also used for y), center of data in lat, lon, station
           locations, color scale, min and max limits and levels array for color
           scale.
        """
        domain = xmax-xint/2.
        maps = Basemap(projection='laea',lat_0=centerlat,lon_0=centerlon,
                       width=domain*2,height=domain*2)
        s = plt.pcolormesh(np.arange(xmin-xint/2.,xmax+3*xint/2.,xint)+domain,
                           np.arange(xmin-xint/2.,xmax+3*xint/2.,xint)+domain,
                           data, cmap = color)
        s.set_clim(vmin=cmin,vmax=cmax)
        CS = plt.contour(np.arange(xmin,xmax+xint,xint)+domain,
                         np.arange(xmin,xmax+xint,xint)+domain,
                         data, colors='k',levels=levels_t)
        plt.clabel(CS, inline=1, fmt='%1.2f',fontsize=8)
        plt.scatter(stations[:,0]+domain, stations[:,1]+domain, color='k',s=2)
        maps.drawstates()
        fig = plt.gcf()
        circle=plt.Circle((domain,domain),100000,color='0.5',fill=False)
        fig.gca().add_artist(circle)
        circle=plt.Circle((domain,domain),200000,color='0.5',fill=False)
        fig.gca().add_artist(circle)

else:
    #Running plotting routines using cartopy for more recent Python versions 3.x
    def mapped_plot(fig,ax,field,cfield,plot_lon,plot_lat,buffer,geo,tanp,cmap,from_this,to_this,
                    lon,lat,label,levels=(0.05,0.1,0.5,1,5),plot_circles=True):
        """Make plots in map projection with km color scales, requires input
           array, lower color scale limit, higher color scale limit, color scale,
           station location overlay, and array x-,y- min, max and interval, the
           location of the array center (in lat and lon). Output plot also
           contains 100, 200 km range rings if desired (set plot_circles = True)

           Arguments:
                    -) fig is the current figure for plotting
                    -) axis is the desired axis to plot current figure (can be subplots for more than 1 axis)
                       note: as for nice_plot, the axis already is/should be projected onto the desired cartopy projection
                       in either the notebooks, or a custom script.
                    -) field is the field to be plotted with shape (e,m,n,z) where e is the error axis,
                       m and n are the spatial axes, and z is the vertical axis.
                    -) cfield is the field to be countoured as lines if levels are provided,
                    -) field_indx is the index of the error field desired to be plotted (e)
                    -) plot_lon and plot_lat are the spatial coordinates for plotting.
                    -) buffer is the map buffer which determines how much of the map is visualized the figure.\
                    -) tanp and geo are the coordinate transformation instances
                    -) lon,lat are the station coordinates
                    -) lables are the figure titles for the errors,
                    -) levels are the contour/field levels for plotting,
                    -) plot_circles plots range circles if set to True.
        """
        #Resetting coordinate names:
        lonp = plot_lon
        latp = plot_lat

        #Set Map Extents:
        ax.set_extent([lonp.min()+buffer,lonp.max()-buffer,
                       latp.min()+buffer,latp.max()-buffer])

        #Figure plotting:
        field   = field
        s       = ax.pcolormesh(lonp,latp,field,cmap=cmap,transform = ccrs.PlateCarree())
        divider = make_axes_locatable(ax)
        ax_cb   = divider.new_horizontal(size="5%", pad=.3, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        plt.colorbar(s,cax=ax_cb,label=label)
        s.set_clim(vmin=from_this,vmax=to_this)


        if levels is not None:
            CS = ax.contour(lonp,latp,
                         cfield, colors='k',levels=levels,
                         transform=ccrs.PlateCarree())
            plt.clabel(CS, inline=1, fontsize=10)
        ax.set_title(label)


        #Scatter plot all stations on map
        ax.scatter(lon,lat, color='k',s=15 ,transform = ccrs.PlateCarree())


        #Add land/map features including state lines
        ax.add_feature(cf.NaturalEarthFeature(
                       category='cultural',
                       name='admin_1_states_provinces_lines',
                       scale='50m',
                       facecolor='none',
                       ))
        ax.add_feature(cf.STATES)
        ax.coastlines('50m') #if near ocean/gulf

        if plot_circles == True:
            #Plot Circles:
            #This is super janky but works for anything that covers short distances (in relation to Earth's curvature);
            #solution based on: https://gis.stackexchange.com/questions/121256/creating-a-circle-with-radius-in-metres
            center_point = Point(0,0)
            #Buffers are in meters
            circle_100 = center_point.buffer(100e3)
            circle_200 = center_point.buffer(200e3)
            #Get circle boundary coordinates
            x_100,y_100 = np.array(circle_100.boundary.xy[0]),np.array(circle_100.boundary.xy[1])
            x_200,y_200 = np.array(circle_200.boundary.xy[0]),np.array(circle_200.boundary.xy[1])
            #Transform the boundary coordinates from local (meters) into lat lon
            c100_x,c100_y,_     = tanp.fromLocal(np.vstack((x_100,y_100,x_100*0)))
            lon_100,lat_100,_   = geo.fromECEF(c100_x,c100_y,_)

            c200_x,c200_y,_     = tanp.fromLocal(np.vstack((x_200,y_200,x_200*0)))
            lon_200,lat_200,_   = geo.fromECEF(c200_x,c200_y,_)
            #Plot Circles
            ax.plot(lon_100,lat_100,color='k',linewidth=1,transform = ccrs.PlateCarree())
            ax.plot(lon_200,lat_200,color='k',linewidth=1,transform = ccrs.PlateCarree())


    def nice_plot(data,lons,lats,centerlat,centerlon,stations,color,cmin,
                  cmax,levels_t,tanp,geo,axis,proj):
        """Make plots in map projection, requires input array, x-min max and
           interval (also used for y), center of data in lat, lon, station
           locations, color scale, min and max limits and levels array for color
           scale.

           Arguments:
                -) Data is the data field to be plotted
                -) lons,lats are the spatial coordinates of the data
                -) centerLat,centerLon (unused)
                -) stations are the lat lon station center locations
                -) color is the colormap to be used for plotting
                -) cmin,cmax are the color limits (vmin,vmax)
                -) levels_t are the levels of the contour data
                -) tanp,geo are the map projection instances used to Convert the
                   spatial coordinates of the circles drawn on the map
                -) axis is the axis of the figure in which the figure is to be plotted (can be
                   the only axis, or one of many subplots)
                -) proj is the data projection cartopy instance 
        """
        #Set projection to match original version
        # proj = ccrs.LambertAzimuthalEqualArea(central_latitude  = centerlat,
        #                                       central_longitude = centerlon)
        #Make Figure axis
        # ax = plt.axes(projection = proj)
        ax = axis

        #Set Map Extents:
        buffer = 0.08 #degrees
        ax.set_extent([lons[0]-buffer,lons[-1]+buffer,
                       lats[0]-buffer,lats[-1]+buffer])

        #Plot the simulation data
        s = ax.pcolormesh(lons, lats, data, cmap=color,transform = ccrs.PlateCarree())
        s.set_clim(vmin=cmin,vmax=cmax)
        #Contour specified intervals and label them inline
        CS = ax.contour(lons, lats, data, colors='k',levels=levels_t,transform = ccrs.PlateCarree())
        plt.clabel(CS, inline=1, fmt='%1.2f',fontsize=8)
        #Scatter plot all stations on map
        ax.scatter(stations[:,0], stations[:,1], color='k',s=2,transform = ccrs.PlateCarree())


        #Add land/map features including state lines
        ax.add_feature(cf.NaturalEarthFeature(
                       category='cultural',
                       name='admin_1_states_provinces_lines',
                       scale='50m',
                       facecolor='none',
                       ))
        ax.add_feature(cf.STATES)
        ax.coastlines('50m') #if near ocean/gulf

        #Plot Circles:
        #This is super janky but works for anything that covers short distances (in relation to Earth's curvature);
        #solution based on: https://gis.stackexchange.com/questions/121256/creating-a-circle-with-radius-in-metres
        center_point = Point(0,0)
        #Buffers are in meters
        circle_100 = center_point.buffer(100e3)
        circle_200 = center_point.buffer(200e3)
        #Get circle boundary coordinates
        x_100,y_100 = np.array(circle_100.boundary.xy[0]),np.array(circle_100.boundary.xy[1])
        x_200,y_200 = np.array(circle_200.boundary.xy[0]),np.array(circle_200.boundary.xy[1])
        #Transform the boundary coordinates from local (meters) into lat lon
        c100_x,c100_y,_     = tanp.fromLocal(np.vstack((x_100,y_100,x_100*0)))
        lon_100,lat_100,_   = geo.fromECEF(c100_x,c100_y,_)

        c200_x,c200_y,_     = tanp.fromLocal(np.vstack((x_200,y_200,x_200*0)))
        lon_200,lat_200,_   = geo.fromECEF(c200_x,c200_y,_)
        #Plot Circles
        ax.plot(lon_100,lat_100,color='gray',linewidth=1,transform = ccrs.PlateCarree())
        ax.plot(lon_200,lat_200,color='gray',linewidth=1,transform = ccrs.PlateCarree())
