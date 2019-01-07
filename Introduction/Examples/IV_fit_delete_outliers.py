''' This script generates a html file, where you can fit IV curves with the full diode equation 
including I_0, n, R_S and R_Sh (see Macabebe in Phys. Stat. Sol. (c))

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve IV_fit_delete_outliers.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/IV_fit_delete_outliers

in your browser.

'''

from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import (TextInput, CheckboxGroup,
                                  Select, Slider,Button)
from bokeh.models.layouts import Row, Column
from bokeh.io import curdoc
from bokeh.models.glyphs import Circle
import pandas as pd
import numpy as np
import scipy.optimize as sp
from scipy.constants import k,e
from scipy.interpolate import interp1d

#Insert file name and temperature in kelvin
file_name = 'IV_data_outlier.txt'
T = 295
#If manual_values = False, values for n, R_S, R_Sh and I_0 will be overwritten later
manual_values = True
#Manually chose some starting parameters
n = 1.2
R_S = 1000
R_Sh = 1e+7
I_0 = 1e-9

#Define helper functions
def line_func(x,a,b):
    return a*x+b
def get_line_paramters(y0,y1,x0,x1):
    if x0 >= x1:
        slope = (y0-y1)/(x0-x1)
    else:
        slope = (y1-y0)/(x1-x0)
    intersect = y0 -  slope*x0
    return [slope,intersect]

#Define functions necessary for fitting of I-V curves
def diode_current(Vol,I_0,n,R_Sh,R_S,T,I_guess='None'):
    #Define diode function as equation that can be solved as diode_func = 0
    def diode_func(I,Vol,I_0,n,R_Sh,R_S,T):
        return I_0*(np.exp(e*(Vol-I*R_S)/(n*k*T))-1)+(Vol-I*R_S)/R_Sh-I
    def diode_current_guess(Vol,I_0,n,R_Sh,T):
        I = []
        for V in Vol:
            if V < 0:
                I.append(V/R_Sh)
            elif V == 0:
                I.append(0.0)
            else:
                I.append(I_0*np.exp(e*V/(k*T*n)))
        return I
    # if no I_guess is passed explicitely, generate it from ideal diode euqation
    try:
        if I_guess == 'None':
            I_guess = diode_current_guess(Vol,I_0,n,R_Sh,T)
    except TypeError:
        I_guess = I_guess
    I = []
    for (j,V) in enumerate(Vol):
        if V < 0 or V > 0:
            # Numerical solver needs two starting points a and b around I_guess
            # diode_func(a) needs to have a different sign from diode_func(b)  
            for m in range(100):
                a = I_guess[j]+(m*I_guess[j]/10)
                b = I_guess[j]-(m*I_guess[j]/10)
                if (diode_func(a,V,I_0,n,R_Sh,R_S,T)*diode_func(b,V,I_0,n,R_Sh,R_S,T) < 0):
                    break
            try:
                I.append(float(sp.brentq(diode_func, a, b,args = (V,I_0,n,R_Sh,R_S,T),maxiter=10000)))
            except ValueError:
                I.append(np.nan)
        else:
            I.append(0.0)
    return I

# Define least squares function which shall be minimized later on
# params is array with fitting parameters
def diode_least_squares(params,Vol_data,I_data,T,method):
    I_0 = params[0]
    n = params[1]
    R_S = params[2]
    R_Sh = params[3]
    result = 0
    I_fit = diode_current(Vol_data,I_0,n,R_Sh,R_S,T,I_data)
    for (V,I,I_f) in zip(Vol_data,I_data,I_fit):
        if method == 'Log':
            if I_fit != 0:
                result += (np.log10(np.abs(I))-np.log10(np.abs(I_f)))**2
        elif method == 'Linear':
            result += (I-I_f)**2
    return result

#Define own function to perform brute force minimization
def brute_force_diode_least_squares(params,Vol_data,I_data,method,parameter_name,steps):
    I_0 = params[0]
    n = params[1]
    R_S = params[2]
    R_Sh = params[3]
    if parameter_name == 'R_S':
        R_S_array = np.linspace(R_S-R_S/2,R_S+R_S/2,steps)
        minimum = np.Inf
        for R_S_candidate in R_S_array:
            params_candidate = [I_0,n,R_S_candidate,R_Sh]
            res = diode_least_squares(params_candidate,Vol_data,I_data,T,method)
            if res < minimum:
                minimum = res
                R_S = R_S_candidate
    if parameter_name == 'I_0':
        I_0_array = np.linspace(I_0-I_0/2,I_0+I_0/2,steps)
        minimum = np.Inf
        for I_0_candidate in I_0_array:
            params_candidate = [I_0_candidate,n,R_S,R_Sh]
            res = diode_least_squares(params_candidate,Vol_data,I_data,T,method)
            if res < minimum:
                minimum = res
                I_0 = I_0_candidate
    result = [I_0,n,R_S,R_Sh]
    return(result)

#Delete outliers
def delete_outliers(IV_data,indices_of_outliers):
    # Drop row specified in indices_outliers
    IV_data.drop(labels=indices_of_outliers,inplace=True)

    return(IV_data)

# Define function that compares two dataframes and gives overlap as well as unique data points
def compare(IV_data,IV_data_sub):
    IV_data_overlap = pd.DataFrame(columns = IV_data.columns,index= range(len(IV_data['Voltage'])))
    IV_data_unique = pd.DataFrame(columns = IV_data.columns,index= range(len(IV_data['Voltage'])))
    for i,V in enumerate(IV_data['Voltage']):
        if V in list(IV_data_sub['Voltage']):
            for column in IV_data.columns:
                IV_data_overlap[column][i] = IV_data[column][i]
        else:
            for column in IV_data.columns:
                IV_data_unique[column][i] = IV_data[column][i]
    return([IV_data_overlap,IV_data_unique])

#Define functions for callbacks
#Update plot when manually changing parameters
def update_plot(attrname, old, new):
    #Get current slider values
    par_name = 'Saturation Current'
    I_0 = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Series Resistance'
    R_S = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Shunt Resistance'
    R_Sh = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Ideality Factor'
    n = sliders['{0} prefactor'.format(par_name)].value
    
    #Calculate new fit by eye curve
    IV_data['Current_fit_update'] = diode_current(IV_data['Voltage'],I_0,n,R_Sh,R_S,T,IV_data['Current'])
    IV_data['Abs_Current_fit_update'] = np.abs(IV_data['Current_fit_update'])
    
    #Push new data to plots
    source_fit_by_eye.data = dict(x=IV_data['Voltage'],y=IV_data['Current_fit_update'])
    source_fit_by_eye_log.data = dict(x=IV_data['Voltage'],y=IV_data['Abs_Current_fit_update'])

#Fitting
def fit_data():
    #Get current slider values
    par_name = 'Saturation Current'
    I_0 = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Series Resistance'
    R_S = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Shunt Resistance'
    R_Sh = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Ideality Factor'
    n = sliders['{0} prefactor'.format(par_name)].value
            
    #Get current fitting method
    method = methods_select.value
    method_fitting = fit_methods_select.value
    
    #Prepare Fitting data
    xdata = [x for x in IV_data['Voltage']]
    ydata = [y for y in IV_data['Current']]
   
    if method_fitting != 'By Eye':
        if method_fitting == 'Gradient Descent':
            #Guess for fit
            x0 = [I_0,n,R_S,R_Sh]
            #Bounds for fitting
            bounds = [(I_0/10,I_0*10),(1,5),(R_S/10,R_S*10),(R_Sh/10,R_Sh*10)]
            #Perform actual fitting
            res = sp.minimize(diode_least_squares,x0,bounds=bounds,args=(xdata,ydata,T,method))
            #Get fitting parameters
            print(res)
            params = res.x
            hess_inv = res.hess_inv
            I_0 = params[0]
            n = params[1]
            R_S = params[2]
            R_Sh = params[3]
            
        if 'Brute Force' in method_fitting:
            parameter_brute_force = method_fitting.split(' ')[-1]
            #Prepare range for brute force search
            params = [I_0,n,R_S,R_Sh]
            res = brute_force_diode_least_squares(params,xdata,ydata,method,parameter_brute_force,1000)
            params = res
            I_0 = params[0]
            n = params[1]
            R_S = params[2]
            R_Sh = params[3]
            print(res)
        
        #Generate final fit
        #Calculate new fit by eye curve
        IV_data['Current_fit_update'] = diode_current(IV_data['Voltage'],
                                                 I_0,n,R_Sh,R_S,T,IV_data['Current'])
        IV_data['Abs_Current_fit_update'] = np.abs(IV_data['Current_fit_update'])
        
        #Push new data to plots
        source_final_fit.data = dict(x=IV_data['Voltage'],y=IV_data['Current_fit_update'])
        source_final_fit_log.data = dict(x=IV_data['Voltage'],y=IV_data['Abs_Current_fit_update'])

#Saving data
def save_data():
    #Get current slider values
    par_name = 'Saturation Current'
    I_0 = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Series Resistance'
    R_S = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Shunt Resistance'
    R_Sh = sliders['{0} prefactor'.format(par_name)].value*10**sliders['{0} exponent'.format(par_name)].value
    par_name = 'Ideality Factor'
    n = sliders['{0} prefactor'.format(par_name)].value
    
    #Get current fitting method
    method = methods_select.value
    method_fitting = fit_methods_select.value
    
    #Perform Fit
    xdata = [x for x in IV_data['Voltage']]
    ydata = [y for y in IV_data['Current']]
    
    if method_fitting != 'By Eye':
        if method_fitting == 'Gradient Descent':
            #Guess for fit
            x0 = [I_0,n,R_S,R_Sh]
            #Bounds for fitting
            bounds = [(I_0/10,I_0*10),(1,5),(R_S/10,R_S*10),(R_Sh/10,R_Sh*10)]
            #Perform actual fitting
            res = sp.minimize(diode_least_squares,x0,bounds=bounds,args=(xdata,ydata,T,method))
            #Get fitting parameters
            print(res)
            params = res.x
            hess_inv = res.hess_inv
            I_0 = params[0]
            n = params[1]
            R_S = params[2]
            R_Sh = params[3]
            
        if 'Brute Force' in method_fitting:
            parameter_brute_force = method_fitting.split(' ')[-1]
            #Prepare range for brute force search
            params = [I_0,n,R_S,R_Sh]
            res = brute_force_diode_least_squares(params,xdata,ydata,method,parameter_brute_force,1000)
            params = res
            I_0 = params[0]
            n = params[1]
            R_S = params[2]
            R_Sh = params[3]
            print(res)
    
    data = {'I_0' : I_0, 'n' : n, 'R_S' : R_S, 'R_Sh' : R_Sh}

    params_final = pd.DataFrame(data = data, columns = ['I_0','n','R_S','R_Sh'],index = range(1))

    #Generate final fit
    #Calculate new fit by eye curve
    IV_data_raw['Current_fit_update'] = diode_current(IV_data_raw['Voltage'],I_0,n,R_Sh,R_S,
                                                      T,IV_data_raw['Current'])

    
    data = {'Voltage' : IV_data_raw['Voltage'], 'Current' : IV_data_raw['Current_fit_update']}
    IV_final_fit = pd.DataFrame(data = data,columns = ['Voltage','Current'],index = range(len(IV_data_raw['Current_fit_update'])))
    
    #Save errors of fitting procedure
    if method == 'log' or method == 'linear':
        if method == 'log':
            var_residuals = np.var((np.log(np.abs(IV_data['Current']))-
                                    np.log(np.abs(IV_data['Current_fit_update'])))**2)
        elif method == 'linear':
            var_residuals = np.var(np.abs(IV_data['Current']
                                          -np.abs(IV_data['Current_fit_update'])**2))
        covar = var_residuals*hess_inv
        error = np.sqrt(covar.matvec(np.ones(len(params))))
        errors = {'Delta_I_0' : error[0], 'Delta_n' : error[1], 
                  'Delta_R_S' : error[2], 'Delta_R_Sh' : error[3]}

        error_final = pd.DataFrame(data = errors, 
                                   columns = ['Delta_I_0','Delta_n','Delta_R_S','Delta_R_Sh'],
                                   index = range(1))
    
    #Save output files
    IV_final_fit.to_csv('IV_fit_for_{0}K.txt'.format(str(T)), header=['Voltage','Current'], index=None, 
                        sep=' ', mode='w')
    params_final.to_csv('IV_fit_parameters_for_{0}K.txt'.format(str(T)), header=['I_0','n','R_S','R_Sh'], index=None, sep=' ', mode='w')
    if method == 'log' or method == 'linear':
        error_final.to_csv('IV_error_fit_parameters_for_{0}K.txt'.format(str(T)), header=['Delta_I_0','Delta_n','Delta_R_S','Delta_R_Sh'], index=None, sep=' ', mode='w')

#Define function for selection and deleting of outliers
def update_plot_outlier_selection(attrname, old, new):
    # Get selected indices of outliers
    new_indices_of_outliers = new['1d']['indices']

    # Drop outliers from data displayed in final fit
    delete_outliers(IV_data,new_indices_of_outliers)
    IV_data.reset_index(inplace=True,drop=True)

    # Generate data file with selected outliers
    IV_data_outliers = compare(IV_data_raw,IV_data)[1]
    
    source_outliers.data = dict(x=IV_data_outliers['Voltage'], y=IV_data_outliers['Current'])
    source_outliers_log.data = dict(x=IV_data_outliers['Voltage'], y=IV_data_outliers['Abs_Current'])
    
    source_data.data = dict(x=IV_data['Voltage'], y=IV_data['Current'])
    source_data_log.data = dict(x=IV_data['Voltage'], y=IV_data['Abs_Current'])

#Save IV data without outliers
def save_data_without_outlier():
    IV_data_temp = IV_data.copy()
    IV_data_temp.reset_index(inplace=True,drop=True)
    data = {'Voltage' : IV_data_temp['Voltage'], 'Current' : IV_data_temp['Current']}
    IV_data_to_save = pd.DataFrame(data = data,columns = ['Voltage','Current'],
                                   index = range(len(IV_data_temp['Current'])))
    IV_data_to_save.reset_index(inplace=True,drop=True)
    IV_data_to_save.to_csv('IV_data_outlier_corrected.txt',header=['Voltage','Current'],index=None,
                          sep=' ',mode='w')
    
# Load IV data to be modelled (This needs to be adjusted for file format you have)
IV_data = pd.read_table(file_name,skiprows=1,names=['Voltage','Current'],delimiter=' ')

#Setup list for storing indices for outliers
indices_of_outliers = []

# Drop lines where voltage and current have opposite signs
# The numerical solver struggles with such non-physical data points!
indices_to_drop = []
for V,I,i in zip(IV_data['Voltage'],IV_data['Current'],range(len(IV_data['Current']))):
    if V*I < 0:
        indices_to_drop.append(i)
IV_data = IV_data.drop(indices_to_drop)
IV_data.reset_index(inplace = True,drop=True)
    
        
# Zero I-V data
# I(V = 0V) = 0 A
for V,I,i in zip(IV_data['Voltage'],IV_data['Current'],range(len(IV_data['Current']))):
    if V == 0:
        IV_data['Current'] = IV_data['Current'] - I
        break
        
# Prepare initial fit data
# Use rough approximations for the IV curve or manually inputed values

if manual_values == False:
    # Get n from slope of log(I) for small forward bias voltages, 
    # a.k.a. use the first few data points after V = 0V
    # Get I_0 from corresponding y intersect
    # Find where forward bias region starts
    for i in range(len(IV_data['Current'])):
        if IV_data['Voltage'][i]*IV_data['Voltage'][i+1] <= 0:
            index_0 = i
            break

    # Data could have been measured from reverse to forward bis or the other way around
    # Small forward biases are to be found at a indices range depending on the overal number of indices
    V_forward, I_forward = [], []
    for i in range(max(2,len(IV_data['Voltage'])//20)):
        if IV_data['Voltage'][index_0+1] > 0:
            V_forward.append(IV_data['Voltage'][index_0+i+1])
            I_forward.append(IV_data['Current'][index_0+i+1])
        elif IV_data['Voltage'][index_0+1] < 0:
            V_forward.append(IV_data['Voltage'][index_0-i-1])
            I_forward.append(IV_data['Current'][index_0-i-1])


    # Fit small forward bias region
    params, covar = sp.curve_fit(line_func,V_forward,np.log(I_forward),p0=[1,1])
    n = e/(k*T*params[0])
    I_0 = np.exp(params[1])


    # Get R_S from slope of I-V curve at large forward biases (5 data points)
    V_forward, I_forward = [], []
    for i in range(max(2,len(IV_data['Voltage'])//5)):
        if IV_data['Voltage'][0] > 0:
            V_forward.append(IV_data['Voltage'][i])
            I_forward.append(IV_data['Current'][i])
        elif IV_data['Voltage'][len(IV_data['Voltage'])-1] > 0:
            V_forward.append(IV_data['Voltage'][len(IV_data['Voltage'])-1-i])
            I_forward.append(IV_data['Current'][len(IV_data['Voltage'])-1-i])

    # Fit large forward bias region
    params, covar = sp.curve_fit(line_func,V_forward,I_forward,p0=[1,1])
    R_S = 1/params[0]

    # Get R_Sh from slope in reverse bias region
    V_reverse, I_reverse = [], []
    for i in range(index_0):
        if IV_data['Voltage'][0] < 0:
            V_reverse.append(IV_data['Voltage'][i])
            I_reverse.append(IV_data['Current'][i])
        elif IV_data['Voltage'][len(IV_data['Voltage'])-1] < 0:
            V_reverse.append(IV_data['Voltage'][len(IV_data['Voltage'])-1-i])
            I_reverse.append(IV_data['Current'][len(IV_data['Voltage'])-1-i])

    # Fit reverse bias region
    params, covar = sp.curve_fit(line_func,V_reverse,I_reverse,p0=[1,1])
    R_Sh = 1/params[0]


# Generate initial fit    
IV_data['Current_fit'] = diode_current(IV_data['Voltage'],I_0,n,R_Sh,R_S,T,I_guess=IV_data['Current'])

# Prepare absolute value versions of the currents for plotting later
IV_data['Abs_Current_fit'] = np.abs(IV_data['Current_fit'])
IV_data['Abs_Current'] = np.abs(IV_data['Current'])

# Keep raw data for comparing outlier corrected data to original data
IV_data_raw = IV_data.copy()

#Prepare plots
#Define here what interactive functions in plot you want
plot_config = dict(plot_height=330, plot_width=550,
                   tools="pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,hover,save")

#Set up plots
#Generate linear and log plots
plot_fit_by_eye = Figure(title='Linear Plot for T = {0} K'.format(str(T)),**plot_config,
                        x_axis_label='Applied Voltage (V)',y_axis_label='Current (A)')
plot_fit_by_eye_log = Figure(title='Logarithmic Plot for T = {0} K'.format(str(T)),**plot_config,
                             y_axis_type="log",x_axis_label='Applied Voltage (V)',
                            y_axis_label='Current (A)')
plot_final_fit = Figure(title='',**plot_config,
                       x_axis_label='Applied Voltage (V)',y_axis_label='Current (A)')
plot_final_fit_log = Figure(title='',**plot_config,
                            y_axis_type="log",x_axis_label='Applied Voltage (V)',
                           y_axis_label='Current (A)')

#Define data you wanna plot
source_data = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Current']))
source_data_log = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Abs_Current']))
source_final_fit = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Current_fit']))
source_final_fit_log = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Abs_Current_fit']))
source_fit_by_eye = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Current_fit']))
source_fit_by_eye_log = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Abs_Current_fit']))

#Define empty data source for outliers to be marked
source_outliers = ColumnDataSource(data=dict(x=[],y=[]))
source_outliers_log = ColumnDataSource(data=dict(x=[],y=[]))

#Prepare plots containing data for outlier selection
data = plot_fit_by_eye.scatter('x','y',source=source_data, legend='Measured Data',color = 'blue')
data_log = plot_fit_by_eye_log.scatter('x','y',source=source_data_log, legend='Measured Data',color = 'blue')
data_final_fit = plot_final_fit.scatter('x','y',source=source_data, legend='Measured Data',color = 'blue')
data_final_fit_log = plot_final_fit_log.scatter('x','y',source=source_data_log, legend='Measured Data',color = 'blue')

#Turn off that box selection highlights data points
nonselection_glyph = Circle(fill_color='blue', fill_alpha=1, line_color=None)
data.nonselection_glyph = nonselection_glyph
data_log.nonselection_glyph = nonselection_glyph
data_final_fit.nonselection_glyph = nonselection_glyph
data_final_fit_log.nonselection_glyph = nonselection_glyph

#Perform actual plotting
plot_fit_by_eye.line('x','y',source=source_fit_by_eye, line_width = 3,legend='Fit',color='black')
plot_fit_by_eye_log.line('x','y',source=source_fit_by_eye_log, line_width = 3,legend='Fit',color='black')
plot_final_fit.scatter('x','y',source=source_data, legend='Measured Data',color = 'blue')
plot_final_fit_log.scatter('x','y',source=source_data_log, legend='Measured Data',color = 'blue')
plot_final_fit.line('x','y',source=source_final_fit, line_width = 3,legend='Fit',color='black')
plot_final_fit_log.line('x','y',source=source_final_fit_log, line_width = 3,legend='Fit',color='black')

#Plot empty data source which corresponds to outliers
plot_fit_by_eye.scatter('x','y',source=source_outliers, legend='Outliers',color='purple')
plot_fit_by_eye_log.scatter('x','y',source=source_outliers_log, legend='Outliers',color='purple')

#Place legends
plot_fit_by_eye.legend.location = 'top_left'
plot_fit_by_eye_log.legend.location = 'top_left'
plot_final_fit.legend.location = 'top_left'
plot_final_fit_log.legend.location = 'top_left'

#Callbacks
parameter_names = ['Saturation Current','Series Resistance','Shunt Resistance','Ideality Factor']
parameters = [I_0,R_S,R_Sh,n]

#Use also slider for oder of magnitude
exponents = {}
prefactors = {}

#Go through parameters and get their order of magnitude
for par_name,par in zip(parameter_names,parameters):
    exponents[par_name] = np.floor(np.log10(par))
    prefactors[par_name] = par/10**(exponents[par_name])
   
# Define sliders
sliders = {}

for par_name in parameter_names:
    sliders['{0} prefactor'.format(par_name)] = Slider(start=1, 
                                                       end=10, 
                                                       value=prefactors[par_name], 
                                                       step=0.001, 
                                                       title='{0}: Prefactor'.format(par_name))
    if par_name == 'Saturation Current':
        sliders['{0} exponent'.format(par_name)] = Slider(start=-30, 
                                                           end=0, 
                                                           value=exponents[par_name], 
                                                           step=1, 
                                                           title='{0}: Exponent'.format(par_name))
    elif par_name == 'Shunt Resistance' or par_name == 'Series Resistance':
        sliders['{0} exponent'.format(par_name)] = Slider(start=-2, 
                                                           end=20, 
                                                           value=exponents[par_name], 
                                                           step=1, 
                                                           title='{0}: Exponent'.format(par_name))
    


#Buttons for saving and fitting
Fit_button = Button(label="Perform Fit")
Save_button = Button(label="Save Fit")

#Button for saving IV data without outliers
Save_data_wo_outlier_button = Button(label="Save data without outliers")

#Define Drop-down menu for fitting method
methods_select = Select(title='Fitting to:', value='Log', options=['Log','Linear'])
fit_methods_select = Select(title='Fitting method:', value='by eye', options=['By Eye','Gradient Descent',
                                                                              'Brute Force - I_0',
                                                                             'Brute Force - R_S'])


#Loop through callbacks and look for activity
for w in [sliders[key] for key in sliders]:
    w.on_change('value', update_plot)
Fit_button.on_click(fit_data)
Save_button.on_click(save_data)
Save_data_wo_outlier_button.on_click(save_data_without_outlier)
#Define Callback for selection of outliers
data.data_source.on_change('selected', update_plot_outlier_selection)
data_log.data_source.on_change('selected', update_plot_outlier_selection)

                               
#Define layout of resulting web side
plots = Column(children=[plot_fit_by_eye,plot_final_fit])
plots_log = Column(children=[plot_fit_by_eye_log,plot_final_fit_log])
inputs_sliders_prefactors = Column(children=[sliders['{0} prefactor'.format(par_name)] for par_name in parameter_names])
inputs_sliders_exponents = Column(children=[sliders['{0} exponent'.format(par_name)] for par_name in parameter_names if par_name != 'Ideality Factor'])                        
inputs_buttons = Column(children=[Fit_button,Save_button,methods_select,fit_methods_select,Save_data_wo_outlier_button])
inputs_sliders = Row(children=[inputs_sliders_prefactors,inputs_sliders_exponents])
inputs = Column(children=[inputs_sliders,inputs_buttons])
curdoc().add_root(Row(children=[plots,plots_log,inputs]))