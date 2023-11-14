#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:08:48 2023

@author: Marjolein Dekker, Onim Sarker, Rohan Tewari and  Youssef el Mourabit
"""

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy.optimize 
import datetime
#We divide by 100 because preference to work in decimals instead of percentages. 
dfData = pd.read_csv("sv.dat") /100
def fnPlotFigure(vData, cTitle):
    plt.figure(figsize=(14,6))
    plt.scatter(range(1,len(vData)+1),vData)
    plt.title(cTitle)
    plt.xlabel("Days")
    plt.ylabel("return")
    plt.show()
############################################ Question 1.A ##################################################
#Graph
fnPlotFigure(dfData, "Plot of financial time series returns")

#Descriptives
print(dfData.describe())

############################################ Question 1.B ##################################################

dfXt = np.log((dfData - np.mean(dfData)) **2)
fnPlotFigure(dfXt, "Transformed return data")

############################################ Question 1.C ##################################################
def fnSVFilter(params, dfData):
    """Filtering

    Args:
        params (list): Are the parameters that need to be estimated and consists of:
            params[0] : Sigma^2 of eta 
            params[1] : Omega
            params[2] : Tt (phi)
        dfData (_type_): _description_

    Returns:
        _type_: _description_
    """
    dfXt = np.array(dfData)
    n = len(dfXt)

    #Make vectors 
    a= np.zeros(n)
    p= np.zeros(n)
    v= np.zeros(n)
    f= np.zeros(n)
    k= np.zeros(n)
    cond_a= np.zeros(n)
    cond_p= np.zeros(n)

    sig_2_eps= np.pi ** 2 /2 #given
    sig_2_eta=  params[0] #To be estimated
    dOmega = params[1]
    dTt = params[2] 
    #Initialization 
    p[0]= sig_2_eta/(1-dTt**2)
    a[0]= dOmega/(1-dTt)
    dDt = -1.27
    # Do the Kalmann filter 
    #Zt = 1, dt = 0, Tt = phi which is in (0,1), ct = omega, Ht = sig2eps, Rt = 1, Qt = sig2eta
    for i in range(n-1):
        v[i]= dfXt[i] - a[i] - dDt 
        f[i]= p[i] + sig_2_eps
        k[i]= p[i]/f[i]
        cond_a[i]= a[i] + k[i]*v[i]
        cond_p[i]= p[i] - k[i]*p[i]
        a[i+1]= dTt * cond_a[i] + dOmega
        p[i+1]= cond_p[i] * dTt**2 + sig_2_eta

        if i==n-2:
            v[i+1]= dfXt[i+1] -a[i+1]
            f[i+1]= p[i+1] + sig_2_eps
            k[i+1]= p[i+1]/f[i+1]
            cond_a[i+1]= dTt * a[i+1] + k[i+1]*v[i+1]
            cond_p[i+1]= p[i+1] * dTt**2 - k[i+1]*p[i+1]
    return v, f, k, a, p, dTt 

def fnMinLogLikelihood(params, dfData):
    n = len(np.array(dfData))
    dV, dF, k, a, p, dTt = fnSVFilter(params, dfData) 
    dLogLikelihood = -1*(n/2)* np.log(2*np.pi) - 1/2* np.sum(np.log(dF[2:]) + (dV[2:]**2 / dF[2:]))
    return -1 * dLogLikelihood

# dfCopy = dfXt.copy()
# dfCopy['x'] = dfCopy.shift()
# dfCopy = dfCopy.dropna()
# dCov = dfCopy["// Pound/Dollar daily exchange rates \\sections 9.6 and 14.4"].cov(dfCopy['x'])
def fnEstimateParamters(dfXt, dPhi_ini): 
        
    dVar = dfXt.var()
    dMean = dfXt.mean()
    dOmega_ini = (1-dPhi_ini)*(dMean + 1.27)
    dSigmaEta2_ini = (1-dPhi_ini**2)*(dVar - np.pi**2/2)

    param_ini = [dSigmaEta2_ini, dOmega_ini, dPhi_ini]
    result = scipy.optimize.minimize(fnMinLogLikelihood, param_ini, args = dfXt, method = "SLSQP")
    return result
 
result= fnEstimateParamters(dfXt,  0.995)
print(result)

############################################ Question 1.4 ##################################################
def fnQd(result, dfXt): 
    """Smoothing and filtering 

    Args:
        result: Result of the optimization:

        dfXt (_type_): _description_

    Returns:
        Plots of the filtered and smoothed h_t and h_t tilde 
    """
    v, f, k, a, p, dTt = fnSVFilter(result.x, dfXt)

    n = len(dfXt)

    r= np.zeros(n)
    N_s= np.zeros(n)
    alpha_hat= np.zeros(n)
    V_smo= np.zeros(n)

    # Initialize
    r[n-1]=0
    N_s[n-1]=0

    # Do the Kalmann smoother
    for i in range(1,n):
        r[n-1-i]= (f[n-i])**-1 * v[n-i] + (dTt-k[n-i])*r[n-i]
        N_s[n-1-i]= (f[n-i])**-1  + (dTt-k[n-i])**2 *N_s[n-i]
        alpha_hat[n-i]= a[n-i] + p[n-i] *r[n-1-i]
        V_smo[n-i] = p[n-i] - p[n-i]**2 * N_s[n-1-i]
        if i==n-1:
            r0= (f[0])**-1 * v[0] + (dTt-k[0])*r[0]
            n0= f[0]**-1  + (dTt-k[0])**2 *N_s[0]
            alpha_hat[0]= a[0] + p[0] *r0
            V_smo[0]= p[0] - p[0]**2 * n0

    #Plotting figure: 

    plt.figure(figsize=(14,6))
    plt.title("Smoothed and Filtered h_t")
    plt.scatter(range(1,len(dfXt)+1), dfXt, color = "gray")
    plt.xlabel("time (t)")
    plt.plot(a, color = "blue")
    plt.plot(alpha_hat, color = "red")
    plt.show()

    # f, v, k, a, p, dTt (From estimate)

    #H_tilde part 
    dPsi_hat = result.x[1]/(1 - result.x[2])

    #plt.plot(dfData * 100)
    plt.figure(figsize=(14,6))
    plt.plot(a - dPsi_hat, color = "blue")
    plt.xlabel("time (t)")
    plt.title("Filtered and Smoothed estimates of h_t tilde")
    plt.plot(alpha_hat - dPsi_hat, color = "red")
    plt.show()
    
    return alpha_hat

alpha_hat = fnQd(result, dfXt)
fnQd(result, dfXt)
############################################ Question 1.E ##################################################
#Processing and analysis data
dfData_e =  pd.read_csv('realized_volatility.csv')
SnP500=dfData_e[dfData_e["Symbol"]== ".SPX"].reset_index(drop=True)
SnP500['date']= pd.to_datetime(SnP500['date'].copy())

def fnPlotFigure_e(x, y, cTitle, ylabel):
    plt.figure(figsize=(14,6))
    plt.scatter(x, y)
    plt.title(cTitle)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.show()
######## Question 1.E.A  Returns S&P500#######################
# Calculate log returns 
SnP500['log_ret'] = np.log(SnP500.close_price) - np.log(SnP500.close_price.shift(1))

# Plot logreturns prices 
fnPlotFigure_e(SnP500['date'][1:],SnP500["log_ret"][1:],"Log returns S&P500", "Log returns")

# Descriptive statistics
print("Start date:")
print(min(SnP500['date']))

print("End date:")
print(max(SnP500['date']))
print(SnP500["log_ret"][1:].describe())

######## Question 1.E.B  Returns S&P500 #######################
df_transformed_log_ret= np.log((SnP500['log_ret'][1:] - np.mean(SnP500['log_ret'][1:])) **2)

fnPlotFigure_e(SnP500['date'][1:],df_transformed_log_ret,"Transformed log returns S&P500", 
               "Transformed log returns")

######## Question 1.E.C  Returns S&P500 #######################
result_SnP500_ret= fnEstimateParamters(df_transformed_log_ret,  0.995)
print("Sigma eta, omega, phi")
print(result_SnP500_ret)

######## Question 1.E.D  Returns S&P500#######################

fnQd(result_SnP500_ret, df_transformed_log_ret)


############################################ Question 1.E Realized Variance ##################################################


#For this part we will use Section 6.2.3 for constructing the model with the 
# realized variance included. In the Assignment notes this is approach 2

#plot realized variance

#First we run the KF with the linearized model as before and store the prediction error and
# variances v*_{t} and F_{t} (following the notation form Section 6.2.3)

# We use the estimates we got from the first part of (E)
v_star, f, k, a, p, dTt = fnSVFilter(result_SnP500_ret.x, df_transformed_log_ret)
# Second we run the Kalman filter using log(RV_{t}) - 1.27 as observations; 
# We use the estimates for the paramters obtained from the first part of (E) 
# and we store the prediction errors as x_star
# We use the 5 minute realised variance 
adj_rv_5= np.log(SnP500["rv5"][1:]) -1.27
log_rv_5= np.log(SnP500["rv5"][1:]) 
x_star, f_rv, k_rv, a_rv, p_rv, dTt_Rv = fnSVFilter(result_SnP500_ret.x, adj_rv_5)

beta_hat= (np.sum(x_star* (f**-1) * x_star))**-1 *(np.sum(np.transpose(x_star)* (f**-1) * v_star))
var_beta_hat= (np.sum(x_star* (f**-1) * x_star))**-1

# We now have the estimates for the parameters defined in equation (4) of the Assignment
#  We first rewrite the variable for which we want to apply the Kalmann filter
# df_transformed_log_ret - beta_hat * log(rv_5)= h_{t} + u_{t}

#Plot realized variance

fnPlotFigure_e(SnP500['date'][1:],log_rv_5,"Log realized variance S&P500", "Realized variance")

#Descriptive statistics Realized variance

print(log_rv_5.describe())


#####
data_e= df_transformed_log_ret - beta_hat* log_rv_5
data_e.describe()
fnPlotFigure_e(SnP500['date'][1:],data_e,"Transformed log returns S&P500  - Log realized variance S&P500 ", "Diff")
result_RV_ret= fnEstimateParamters(data_e,  0.995)
fnQd(result_RV_ret, data_e)

############################################ Question 1.F ##################################################

# Original data
def dataOriginal():
    data = np.genfromtxt('sv.dat', dtype=float, delimiter=',')
    data = data[1:]
    data = (data - np.mean(data)) / 100
    
    return data

# SNP data
def dataSNP():
    dfData_e =  pd.read_csv('realized_volatility.csv')
    data = dfData_e.loc[dfData_e['Symbol'] == ".SPX"]
    data['log_ret'] = np.log(data.close_price) - np.log(data.close_price.shift(1))
    data = data['log_ret'].to_numpy()

    return data 

# Initialization 
N = 945
mu = 0
phi = 0.8
sigma_eta = 0.2
sigma = sigma_eta * 2 / (1 - phi ** 2)

alpha = np.zeros((N, len(dataOriginal())))              
ll = np.zeros((N, len(dataOriginal())))     
weight = np.zeros((N, len(dataOriginal())))     
a_tt = np.zeros((N, len(dataOriginal())))     
p_tt = np.zeros((N, len(dataOriginal())))     
a_tt_sum = np.zeros((N, 1))
p_tt_sum = np.zeros((N, 1))

def bootstrap(data, alpha, sigma, sigma_eta, phi, N, weight):
    # Initialize alpha tilde 0
    for i in range(0, N):
        alpha[0, i] = np.random.normal(0, sigma, 1)

    # Draw N values of alpha tilde from normal distribution
    for j in range(0, N):
        for k in range(1,N):
            alpha[k,j] = np.random.normal(phi * alpha[k-1,j], np.sqrt(sigma_eta), 1)

    # Corresponding weights are computed 
    for l in range(0, N): 
        ll[l, :] = np.exp(-0.5 * np.log(2 * np.pi * sigma ** 2) - ((1/(2*sigma**2))*np.exp(-alpha[i,:])*data[i]**2))
        weight[i, :] = ll[i, :] / np.sum(ll[i, :])
     
        a_tt[i, :] = np.sum(weight[i, :] * alpha[i, :])
        p_tt[i,:] = np.sum(weight[i,:] * (alpha[i,:]**2)-a_tt[i,:]**2)

    # Resampling
    for i in range(len(alpha)):
        resample = np.random.normal(phi * alpha[i], sigma_eta, N)
    
    # Figure of bootstrap filtering approach vs QML approach
    plt.figure(figsize=(16,10))
    plt.title("Esimates H_t Bootstrap vs QML")
    plt.plot(alpha_hat, color = "red")
    plt.plot(a_tt, color="green")
    plt.show()

    # Figure of resampling
    plt.figure(figsize=(16,10))
    plt.plot(resample, label="")
    plt.title("Resampling")
    plt.show()
 
# Bootstrap filter on original data 
bootstrap(dataOriginal(), alpha, sigma, sigma_eta, phi, N, weight)

# Bootstrap filter on index stocks
bootstrap(dataSNP(), alpha, sigma, sigma_eta, phi, N, weight)





