''' This script generates a html file, where you can fit IV curves with the full diode equation 
including I_0, n, R_S and R_Sh (see Macabebe in Phys. Stat. Sol. (c))

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve IV_fit_by_eye.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/IV_fit_by_eye

in your browser.

'''

from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import (TextInput, CheckboxGroup,
                                  Select, Slider,Button)
from bokeh.models.layouts import Row, Column
from bokeh.io import curdoc
import pandas as pd
import numpy as np
import scipy.optimize as sp
from scipy.constants import k,e
from scipy.interpolate import interp1d

#Insert file name and temperature in kelvin
file_name = 'IV_data.txt'
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



# Load IV data to be modelled (This needs to be adjusted for file format you have)
IV_data = pd.read_table(file_name,skiprows=1,names=['Voltage','Current'],delimiter=' ')

# Drop lines where voltage and current have opposite signs
# The numerical solver struggles with such non-physical data points!
indices_to_drop = []
for V,I,i in zip(IV_data['Voltage'],IV_data['Current'],range(len(IV_data['Current']))):
    if V*I < 0:
        indices_to_drop.append(i)
IV_data = IV_data.drop(indices_to_drop)
IV_data.reset_index(inplace = True)
    
        
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

#Define data you wanna plot
source_data = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Current']))
source_data_log = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Abs_Current']))
source_fit_by_eye = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Current_fit']))
source_fit_by_eye_log = ColumnDataSource(data=dict(x=IV_data['Voltage'],y=IV_data['Abs_Current_fit']))

#Perform actual plotting
plot_fit_by_eye.scatter('x','y',source=source_data, legend='Measured Data',color='blue')
plot_fit_by_eye_log.scatter('x','y',source=source_data_log, legend='Measured Data',color='blue')
plot_fit_by_eye.line('x','y',source=source_fit_by_eye, line_width = 3,legend='Fit',color='black')
plot_fit_by_eye_log.line('x','y',source=source_fit_by_eye_log, line_width = 3,legend='Fit',color='black')

#Place legends
plot_fit_by_eye.legend.location = 'top_left'
plot_fit_by_eye_log.legend.location = 'top_left'

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
    



#Loop through callbacks and look for activity
for w in [sliders[key] for key in sliders]:
    w.on_change('value', update_plot)

                               
#Define layout of resulting web side
plots = Column(children=[plot_fit_by_eye])
plots_log = Column(children=[plot_fit_by_eye_log])
inputs_sliders_prefactors = Column(children=[sliders['{0} prefactor'.format(par_name)] for par_name in parameter_names])
inputs_sliders_exponents = Column(children=[sliders['{0} exponent'.format(par_name)] for par_name in parameter_names if par_name != 'Ideality Factor'])                        
inputs_sliders = Row(children=[inputs_sliders_prefactors,inputs_sliders_exponents])
inputs = Column(children=[inputs_sliders])
curdoc().add_root(Row(children=[plots,plots_log,inputs]))