#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Last Updated:   04/24/2025
@Author      :   Sebin Oh
@Contact     :   sebin.oh@berkeley.edu
@Description :   

This script calculates the force vector for the given displacement vector using the modified Bouc-Wen-Baber-Noori model

"""


import numpy as np
import math

def force_BWBN(parameters, Disp):
    k0 = (2*math.pi/parameters[0])**2/9.8
    Fy = parameters[1]
    alpha = parameters[2]   
    beta = parameters[3]
    gamma = 1-beta
    n = parameters[4]
    deltaNu = parameters[5]
    deltaEta = parameters[6]
    pin_zeta0 = parameters[7]
    pin_p = parameters[8]
    pin_q = parameters[9]
    pin_psi = parameters[10]
    pin_delpsi = parameters[11]
    pin_lambda = parameters[12]

    # Basic parameters
    maxIter = 50000
    tolerance = 1e-12
    startPoint = 1e-5
    DispT = 0.0
    e_old = 0.0
    z_old = 0.0
    
    force_Total = []
    Disp = np.concatenate(([0], Disp),axis=None)
    
    for ii in range(1,len(Disp)):

        DispTdT = Disp[ii]
        deltaDisp = DispTdT - DispT

        # learning rate
        lr = 0.5

        # Perform Newton-Rhapson
        count = 0
        z_new = 1.0
        z_new_p = startPoint
        z_eval = startPoint
        
        while abs(z_new_p - z_new) > tolerance: #&& count < maxIter:
        
            # Step 1
            e_new = e_old + (1 - alpha)*deltaDisp*k0/Fy*z_eval

            nu_new = 1 + deltaNu*e_new
            eta_new = 1 + deltaEta*e_new

            a_1 = beta*np.sign(deltaDisp*z_eval) + gamma
            a_2 = (1-abs(z_eval)**n*a_1*nu_new)/eta_new
            
            # Pinching effect
            Zu = (1/((beta+gamma)*nu_new))**(1/n)
            try:
                temp = math.exp(-pin_p*e_new)
            except OverflowError:
                temp = 0
            zeta1 = pin_zeta0*(1-temp)
            zeta2 = (pin_psi+e_new*pin_delpsi)*(pin_lambda+zeta1)

            if zeta2 != 0:
                try:
                    temp = math.exp(-(z_eval*np.sign(deltaDisp)-pin_q*Zu)**2/zeta2**2)
                except OverflowError:
                    temp = 0
                h = 1 - zeta1*temp
            else:
                h = 1

            EvalFunc = z_eval - z_old - h*a_2*deltaDisp*k0/Fy

            # Step 2: evaluate the deriviative
            # Evaluate function derivatives with respect to z_eval for the Newton-Rhapson scheme
            e_new_ = (1 - alpha)*k0/Fy*deltaDisp

            nu_new_ = deltaNu*e_new_
            eta_new_ = deltaEta*e_new_

            Zu_ = -nu_new_*(beta+gamma)/n*((beta+gamma)*nu_new)**(-(n+1)/n)

            a_2_ = (-eta_new_*(1-abs(z_eval)**(n)*a_1*nu_new)-eta_new*(n*abs(z_eval)**(n-1)*a_1*nu_new*np.sign(z_eval)+abs(z_eval)**(n)*a_1*nu_new_))/eta_new**2

            # Pinching effect
            try:
                temp = math.exp(-pin_p*e_new)
            except OverflowError:
                temp = 0
            zeta1_ = pin_zeta0*pin_p*temp*e_new_
            zeta2_ = pin_psi*zeta1_ + pin_lambda*pin_delpsi*e_new_ + pin_delpsi*e_new_*zeta1 + pin_delpsi*e_new*zeta1_

            if zeta2 != 0:
                try:
                    temp = math.exp(-(z_eval*np.sign(deltaDisp)-pin_q*Zu)**2/zeta2**2)
                except OverflowError:
                    temp = 0
                a3 = -temp
                a4 = 2*zeta1*(z_eval*np.sign(deltaDisp)-pin_q*Zu)*(np.sign(deltaDisp)-pin_q*Zu_)/zeta2**2
                a5 = 2*zeta1*(z_eval*np.sign(deltaDisp)-pin_q*Zu)**2/zeta2**3
            else:
                a3 = 0
                a4 = 0
                a5 = 0
            h1_ = a3*(zeta1_ - a4 + zeta2_*a5)

            EvalFunc_ = 1 - (h1_*a_2+h*a_2_)*deltaDisp*k0/Fy
            
            # Step 3: Perform a new step
            z_new = z_eval - lr*EvalFunc/EvalFunc_

            # Step 4: Update the root
            z_new_p = z_eval
            z_eval = z_new

            count = count + 1

            # Modify the learning rate if the Newton-Rhapson scheme does not converge
            if count == maxIter:
                #print("WARNING: Could not find the root z, after maxIter")
                lr = lr*0.1
                count = 0
                z_new = 1.0
                z_new_p = startPoint
                z_eval = startPoint
        
        # Compute restoring force.
        force = alpha*k0*DispTdT + (1 - alpha)*Fy*z_eval

        DispT = DispTdT
        e_old = e_new
        z_old = z_eval

        force_Total = np.concatenate((force_Total, force),axis=None)

    return force_Total

