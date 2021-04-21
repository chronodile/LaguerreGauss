#Import modules
import numpy as np
import PIL
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import LagGaussBeamModeGenerator as lg
import scipy.special as spec
import scipy.integrate as integ
import math

def waist(w0,z,zr):
    return w0 * np.sqrt(1 + (z/zr)**2)

def lag_poly(x,p,l):
    return spec.assoc_laguerre(x,n=p,k=np.abs(l))

def gaussian(x,amp,mean,stdev,c):
    return amp*(1/(stdev*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-mean)/stdev)**2))) + c

def width_function(z,zr,waist,z0):
    width = waist * np.sqrt(1 + ((z-z0)/zr)**2)

    return width    
    
class function_fit:
    def __init__(self,directory,pls,image_name,centre_x=None,centre_y=None,crop_x=50,crop_y=50,super_number=0,boundaries=False):
        self.directory = directory
        self.pls = np.asarray(pls)   
        self.centre_x = centre_x
        self.centre_y = centre_y
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.get_centre_x = False
        self.get_centre_y = False
        self.image_name = image_name
        self.boundaries = boundaries
        
        self.super_number = super_number
        
        if centre_x == None:
            self.get_centre_x = True
            
        if centre_y == None:
            self.get_centre_y = True
        
        os.chdir(directory)
        
    def analyse_directory(self):
        #Open the directory and get a list of all the filenames
        pass
        os.chdir(self.directory)
        all_files = os.listdir()
        files = []
        
        for file in all_files:
            if file[-3:] == "bmp":
                files += [file]
        
        #Initialise an array to contain the background information
        background = np.zeros(shape=(1024, 1280))
        stdev_background = np.zeros(shape=(1024, 1280))
        
        #Initialise arrays to hold data
        x_width = np.asarray([])
        x_width_error = np.asarray([])
        y_width = np.asarray([])
        y_width_error = np.asarray([])
        zvalues = np.asarray([])
        
        x_mid = np.asarray([])
        x_mid_mod = np.asarray([])
        y_mid = np.asarray([])
        y_mid_mod = np.asarray([])
        
        if self.super_number == 0:
            x_holder = np.empty(((2*self.crop_x),int(len(files)/3) - 1))
            y_holder = np.empty(((2*self.crop_y),int(len(files)/3) - 1))
            
            x_red_chi = np.empty(int(len(files)/3) - 1)
            y_red_chi = np.empty(int(len(files)/3) - 1)

        else:
            x_holder = np.empty(((2*self.crop_x),int(len(files)/3) - 2))
            y_holder = np.empty(((2*self.crop_y),int(len(files)/3) - 2))
            
            x_red_chi = np.empty(int(len(files)/3) - 2)
            y_red_chi = np.empty(int(len(files)/3) - 2)
            
        if self.super_number == 3:
            x_amp_vals_1 = np.empty(int(len(files)/3) - 2)
            x_amp_vals_2 = np.empty(int(len(files)/3) - 2)
            x_amp_vals_3 = np.empty(int(len(files)/3) - 2)
            
            x_amp_error_1 = np.empty(int(len(files)/3) - 2)
            x_amp_error_2 = np.empty(int(len(files)/3) - 2)
            x_amp_error_3 = np.empty(int(len(files)/3) - 2)
            
            y_amp_vals_1 = np.empty(int(len(files)/3) - 2)
            y_amp_vals_2 = np.empty(int(len(files)/3) - 2)
            y_amp_vals_3 = np.empty(int(len(files)/3) - 2)
            
            y_amp_error_1 = np.empty(int(len(files)/3) - 2)
            y_amp_error_2 = np.empty(int(len(files)/3) - 2)
            y_amp_error_3 = np.empty(int(len(files)/3) - 2) 
            
        if self.super_number == 4:
            x_amp_vals_1 = np.empty(int(len(files)/3) - 2)
            x_amp_vals_2 = np.empty(int(len(files)/3) - 2)
            x_amp_vals_3 = np.empty(int(len(files)/3) - 2)
            x_amp_vals_4 = np.empty(int(len(files)/3) - 2)
            
            x_amp_error_1 = np.empty(int(len(files)/3) - 2)
            x_amp_error_2 = np.empty(int(len(files)/3) - 2)
            x_amp_error_3 = np.empty(int(len(files)/3) - 2)
            x_amp_error_4 = np.empty(int(len(files)/3) - 2)
            
            y_amp_vals_1 = np.empty(int(len(files)/3) - 2)
            y_amp_vals_2 = np.empty(int(len(files)/3) - 2)
            y_amp_vals_3 = np.empty(int(len(files)/3) - 2)
            y_amp_vals_4 = np.empty(int(len(files)/3) - 2)
            
            y_amp_error_1 = np.empty(int(len(files)/3) - 2)
            y_amp_error_2 = np.empty(int(len(files)/3) - 2)
            y_amp_error_3 = np.empty(int(len(files)/3) - 2) 
            y_amp_error_4 = np.empty(int(len(files)/3) - 2) 
            
        #Get the mean and standard deviations of the background
        for file in files:
            if self.super_number == 0:
                if file[:10] == "Background":
                    background_image = np.asarray(PIL.Image.open(file))[:,:,0]
                    
                    background += background_image
                    
                    file_split = file.split("-")
                    
                    background_exposure = float(file_split[1])
            else:
                if file[:14] == "BackgroundKino":
                    background_image = np.asarray(PIL.Image.open(file))[:,:,0]
                    
                    background += background_image
                    
                    file_split = file.split("-")
                    
                    background_exposure = float(file_split[1])
           
        #Mean background
        mean_background = background / 3
        
        #Standard deviation on the background
        for file in files:
            if file[:10] == "Background":
                background_image = np.asarray(PIL.Image.open(file))[:,:,0]
                
                stdev_background += (background_image - mean_background)**2
        
        stdev_background = stdev_background / np.sqrt(2)
        
        #Standard error on the background
        stderr_background = stdev_background / np.sqrt(3)
        
        #Normalise the mean and standard error to get the photons per millisecond
        mean_background_persec = mean_background / background_exposure
        stderr_background_persec = stderr_background / background_exposure
        
        #Get the total power of the image at the focus (z = 0.5m)
        for i in range(len(files)):
            file_split = files[i].split("-")
            print(file_split)
                
            
            if file_split[1] == "0.5":
                #Open the three images as arrays
                im1 = np.asarray(PIL.Image.open(files[i]))[:,:,0]
                im2 = np.asarray(PIL.Image.open(files[i + 1]))[:,:,0]
                im3 = np.asarray(PIL.Image.open(files[i + 2]))[:,:,0]
                
                #Extract z position and exposure time from file name
                file_split = files[i].split("-")
                z_value = float(file_split[1])
                exposure = float(file_split[2])
                
                #Get the mean value
                im = (im1/3) + (im2/3) + (im3/3)
                
                #Get the standard error
                stdev_im = ((im1 - im)**2 + (im2 - im)**2 + (im3 - im)**2) / np.sqrt(2)
                stderr_im = stdev_im / np.sqrt(3)
                
                #Normalise the mean and standard error to get the photons per millisecond
                im_persec = im / exposure
                stderr_im_persec = stderr_im / exposure
                
                #Substract the background count from the image and propagate the error
                im_persec -= mean_background_persec
                error_bg_subtract = np.sqrt((stderr_im_persec)**2 + (stderr_background_persec)**2)
                
                #Check whether image centre has been provided, and if not, set image centre as max pixel
                if self.get_centre_x == True:
                    self.centre_x = self.max_pixel_x(im_persec)
                    
                if self.get_centre_y == True:
                    self.centre_y = self.max_pixel_y(im_persec)
                
                #Crop the region of interest from the image
                cropped_image = self.crop(im_persec)
                cropped_error = self.crop(error_bg_subtract)
                
                #Get the total power and the error on the total power
                total_power_at_focus = np.sum(cropped_image)
                total_power_at_focus_error = np.sqrt(np.sum(cropped_error**2))
                
                break
        
        #Set up a counter which counts how many times the following loop has been used on a non-background file; start at -1 so can begin counting from 0
        counter = -1
        
        #Iterate over each data set
        for i in range(len(files)):
            if i % 3 == 0:
                if files[i][:10] != "Background":
                    #Add one to the counter to acknowledge success
                    counter += 1
                    
                    #Open the three images as arrays
                    im1 = np.asarray(PIL.Image.open(files[i]))[:,:,0]
                    im2 = np.asarray(PIL.Image.open(files[i + 1]))[:,:,0]
                    im3 = np.asarray(PIL.Image.open(files[i + 2]))[:,:,0]
                    
                    #Extract z position and exposure time from file name
                    file_split = files[i].split("-")
                    z_value = float(file_split[1])
                    exposure = float(file_split[2])
                    
                    #Get the mean value
                    im = (im1/3) + (im2/3) + (im3/3)
                                        
                    #Get the standard error
                    stdev_im = ((im1 - im)**2 + (im2 - im)**2 + (im3 - im)**2) / np.sqrt(2)
                    stderr_im = stdev_im / np.sqrt(3)
                    
                    #Normalise the mean and standard error to get the photons per millisecond
                    im_persec = im / exposure
                    stderr_im_persec = stderr_im / exposure
                    
                    #Substract the background count from the image and propagate the error
                    im_persec -= mean_background_persec
                    error_bg_subtract = np.sqrt((stderr_im_persec)**2 + (stderr_background_persec)**2)

                    #Check whether image centre has been provided, and if not, set image centre as max pixel
                    if self.get_centre_x == True:
                        self.centre_x = self.max_pixel_x(im_persec)
                        
                    if self.get_centre_y == True:
                        self.centre_y = self.max_pixel_y(im_persec)
                    
                    #Crop the region of interest from the image
                    cropped_image = self.crop(im_persec)
                    cropped_error = self.crop(error_bg_subtract)
                    
                    #Get the total power of this image
                    total_power = np.sum(cropped_image)
                    total_power_error = np.sqrt(np.sum(cropped_error**2))
                    
                    #Normalise the image's power to the power at the focus, and get the total error
                    normalised_image = cropped_image * (total_power_at_focus/total_power)
                    normalised_error = np.empty(normalised_image.shape)
                    
                    for u in range(normalised_image.shape[0]):
                        for v in range(normalised_image.shape[1]):
                            if cropped_image[u,v] == 0:
                                normalised_error[u,v] = 0
                            else:
                                normalised_error[u,v] = normalised_image[u,v] * np.sqrt((cropped_error[u,v]/cropped_image[u,v])**2 + (total_power_error / total_power)**2 + (total_power_at_focus_error / total_power_at_focus)**2)
                    
                    #Slice through the data in the x and y directions
                    slice_x = normalised_image[:,self.crop_x]
                    slice_y = normalised_image[self.crop_y,:]
                    slice_x_err = normalised_error[:,self.crop_x]
                    slice_y_err = normalised_error[self.crop_y,:]
                    
                    #Get arrays of the pixel numbers in the x and y directions
                    pixel_x = np.arange(0,cropped_image.shape[0],1) + (self.centre_x - self.crop_x)
                    pixel_y = np.arange(0,cropped_image.shape[1],1) + (self.centre_y - self.crop_y)
                    
                    print("Position: ",z_value)
                    print("Centre X: ",self.centre_x)
                    print("Centre Y: ",self.centre_y)
                    x_mid = np.append(x_mid,self.centre_x)
                    y_mid = np.append(y_mid,self.centre_y)
                    
                    #Get arrays of the x and y directions in terms of the size of the CCD pixel, 5.2*1e-6
                    ccd_x = (np.arange(0,cropped_image.shape[0],1) - self.crop_x) * 5.2*1e-6
                    ccd_y = (np.arange(0,cropped_image.shape[1],1) - self.crop_y) * 5.2*1e-6
                    
                    #Plot the data slices
                    #plt.plot(pixel_x,slice_x)
                    plt.errorbar(pixel_x,slice_x,yerr=slice_x_err,ecolor="gray",barsabove=False)
                    plt.xlabel("Pixel")
                    plt.ylabel("Intensity at Pixel")
                    plt.title("X Slice at Position " + str(z_value) + "m")
                    plt.show()
                    
                    #plt.plot(ccd_x,slice_x)
                    plt.errorbar(ccd_x,slice_x,yerr=slice_x_err,ecolor="gray",barsabove=False)
                    plt.xlabel("Position / m")
                    plt.ylabel("Intensity at Position")
                    plt.title("X Slice at Position " + str(z_value) + "m")
                    plt.show()
                    
                    #plt.plot(pixel_y,slice_y)
                    plt.errorbar(pixel_y,slice_y,yerr=slice_y_err,ecolor="gray",barsabove=False)
                    plt.xlabel("Pixel")
                    plt.ylabel("Intensity at Pixel")
                    plt.title("Y Slice at Position " + str(z_value) + "m")
                    plt.show()
                    
                    #plt.plot(ccd_y,slice_y)
                    plt.errorbar(ccd_y,slice_y,yerr=slice_y_err,ecolor="gray",barsabove=False)
                    plt.xlabel("Position / m")
                    plt.ylabel("Intensity at Position")
                    plt.title("Y Slice at Position " + str(z_value) + "m")
                    plt.show()
                    
                    """plt.plot(ccd_x,slice_x)
                    plt.plot(ccd_x,self.one_beam_model(ccd_x,a=np.amax(slice_x),w0=0.38*1e-3,z=z_value))
                    plt.show()"""
                    
                    print(self.pls[0,0],self.pls[0,1],1024*1e-9,z_value)
                    
                    #Fit model to x data                    
                    if self.super_number == 0:
                        #new_model = lambda x,a,w0:self.one_beam_model(x,a=a,w0=w0,z=(z_value-0.5))
                        #popt, pcov = sp.curve_fit(new_model,ccd_x,slice_x,p0=[np.amax(slice_x),0.4*1e-3],bounds=lims)
                        
                        xpopt,xpcov = sp.curve_fit(gaussian,ccd_x,slice_x,p0=[0.15,0,0.1*1e-3,0])
                        
                        #def gaussian(x,amp,mean,stdev,c):
                        
                        plt.plot(ccd_x,slice_x)
                        plt.plot(ccd_x,gaussian(ccd_x,*xpopt))
                        plt.show()
    
                        x_width = np.append(x_width,xpopt[2]*2)
                        x_model_errors = np.sqrt(np.diag(xpcov))
                        x_width_error = np.append(x_width_error,x_model_errors[2]*2)
                        
                        dof = len(slice_x) - len(xpopt)
                        
                        x_chi = (1/dof)*np.sum( ( (slice_x - gaussian(ccd_x,*xpopt))/(slice_x_err) )**2 )
                        
                        x_red_chi[counter] = x_chi
                        
                    elif self.super_number == 1:
                        model = lambda x,a,w0:self.one_beam_model(x=x,a=a,w0=w0,z=(z_value - 0.5))
                        xpopt,xpcov = sp.curve_fit(model,ccd_x,slice_x,p0=[1,0.0001])
                        
                        plt.plot(ccd_x,slice_x)
                        plt.plot(ccd_x,model(ccd_x,*xpopt))
                        plt.show()
                        
                        dof = len(slice_x) - len(xpopt)
                        
                        x_chi = (1/dof)*np.sum( ( (slice_x - model(ccd_x,*xpopt))/(slice_x_err) )**2 )
                        
                        x_red_chi[counter] = x_chi
                    
                    elif self.super_number == 2:
                        pass

                    elif self.super_number == 3:
                        if self.boundaries == True:
                            lims = ((0,0,0,0),(np.inf,np.inf,np.inf,np.inf))
                            model = lambda x,w0,a1,a2,a3:self.three_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,norm=False)
                            xpopt,xpcov = sp.curve_fit(model,ccd_x,slice_x,p0=[0.0001,1,1,1],bounds=lims)
                            
                        else:
                            model = lambda x,w0,a1,a2,a3:self.three_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,norm=False)
                            xpopt,xpcov = sp.curve_fit(model,ccd_x,slice_x,p0=[0.0001,1,1,1])
                            
                        plt.errorbar(ccd_x,slice_x,yerr=slice_x_err,ecolor="gray",barsabove=False)
                        plt.plot(ccd_x,model(ccd_x,*xpopt),linestyle="--",color="k")
                        plt.xlabel("Radial Distance from Propagation Axis / m")
                        plt.ylabel("Intensity at Position")
                        fname = "ModelFitX-Z" + str(z_value) + "-" + self.image_name + ".png"
                        plt.savefig(fname,dpi=300)
                        plt.show()
                        
                        dof = len(slice_x) - len(xpopt)
                        
                        x_chi = (1/dof)*np.sum( ( (slice_x - model(ccd_x,*xpopt))/(slice_x_err) )**2 )
                        
                        x_red_chi[counter] = x_chi
                        
                        #Get the normalised amplitudes of each of the components of the fitted model, and save them
                        amp1 = xpopt[1]
                        amp2 = xpopt[2]
                        amp3 = xpopt[3]
                        
                        totamp = amp1 + amp2 + amp3
                        
                        x_amp_vals_1[counter] = amp1/totamp
                        x_amp_vals_2[counter] = amp2/totamp
                        x_amp_vals_3[counter] = amp3/totamp
                        
                        #Get the errors on these amplitude values
                        popt_errors = np.sqrt(np.diag(xpcov))
                        
                        amp1_err = popt_errors[1]
                        amp2_err = popt_errors[2]
                        amp3_err = popt_errors[3]
                        
                        totamp_error = np.sqrt((amp1_err**2) + (amp2_err**2) + (amp3_err**2))
                        
                        x_amp_error_1[counter] = x_amp_vals_1[counter] * np.sqrt((amp1_err/amp1)**2 + (totamp_error/totamp)**2)
                        x_amp_error_2[counter] = x_amp_vals_2[counter] * np.sqrt((amp2_err/amp2)**2 + (totamp_error/totamp)**2)                        
                        x_amp_error_3[counter] = x_amp_vals_3[counter] * np.sqrt((amp3_err/amp3)**2 + (totamp_error/totamp)**2)
                    
                    elif self.super_number == 4:
                        if self.boundaries == True:
                            lims = ((0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf))
                            model = lambda x,w0,a1,a2,a3,a4:self.four_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,a4=a4,norm=False)
                            xpopt,xpcov = sp.curve_fit(model,ccd_x,slice_x,p0=[0.0001,1,1,1,1],bounds=lims)
                            
                        else:
                            model = lambda x,w0,a1,a2,a3,a4:self.four_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,a4=a4,norm=False)
                            xpopt,xpcov = sp.curve_fit(model,ccd_x,slice_x,p0=[0.0001,1,1,1,1])
                    
                        plt.errorbar(ccd_x,slice_x,yerr=slice_x_err,ecolor="gray",barsabove=False)
                        plt.plot(ccd_x,model(ccd_x,*xpopt),color="k",linestyle="--")
                        plt.xlabel("Radial Distance from Propagation Axis / m")
                        plt.ylabel("Intensity at Position")
                        fname = "ModelFitX-Z" + str(z_value) + "-" + self.image_name + ".png"
                        plt.savefig(fname,dpi=300)
                        plt.show()
                        
                        dof = len(slice_x) - len(xpopt)
                        
                        x_chi = (1/dof)*np.sum( ( (slice_x - model(ccd_x,*xpopt))/(slice_x_err) )**2 )
                        
                        x_red_chi[counter] = x_chi
                        
                        #Get the normalised amplitudes of each of the components of the fitted model, and save them
                        amp1 = xpopt[1]
                        amp2 = xpopt[2]
                        amp3 = xpopt[3]
                        amp4 = xpopt[4]
                        
                        totamp = amp1 + amp2 + amp3 + amp4
                        
                        x_amp_vals_1[counter] = amp1/totamp
                        x_amp_vals_2[counter] = amp2/totamp
                        x_amp_vals_3[counter] = amp3/totamp
                        x_amp_vals_4[counter] = amp4/totamp
                        
                        #Get the errors on these amplitude values
                        popt_errors = np.sqrt(np.diag(xpcov))
                        
                        amp1_err = popt_errors[1]
                        amp2_err = popt_errors[2]
                        amp3_err = popt_errors[3]
                        amp4_err = popt_errors[4]
                        
                        totamp_error = np.sqrt((amp1_err**2) + (amp2_err**2) + (amp3_err**2) + (amp4_err**2))
                        
                        x_amp_error_1[counter] = x_amp_vals_1[counter] * np.sqrt((amp1_err/amp1)**2 + (totamp_error/totamp)**2)
                        x_amp_error_2[counter] = x_amp_vals_2[counter] * np.sqrt((amp2_err/amp2)**2 + (totamp_error/totamp)**2)                        
                        x_amp_error_3[counter] = x_amp_vals_3[counter] * np.sqrt((amp3_err/amp3)**2 + (totamp_error/totamp)**2)
                        x_amp_error_4[counter] = x_amp_vals_4[counter] * np.sqrt((amp4_err/amp4)**2 + (totamp_error/totamp)**2)

                    
                    zvalues = np.append(zvalues,z_value)
                    
                    #Fit model to y data
                    if self.super_number == 0:
                        #new_model = lambda x,a,w0:self.one_beam_model(x,a=a,w0=w0,z=(z_value-0.5))
                        #popt, pcov = sp.curve_fit(new_model,ccd_x,slice_x,p0=[np.amax(slice_x),0.4*1e-3],bounds=lims)
                        
                        ypopt,ypcov = sp.curve_fit(gaussian,ccd_y,slice_y,p0=[0.15,0,0.1*1e-3,0])
                        
                        #def gaussian(x,amp,mean,stdev,c):
                        
                        plt.errorbar(ccd_x,slice_x,yerr=slice_x_err,ecolor="gray",barsabove=False,color="Blue")
                        plt.plot(ccd_y,gaussian(ccd_y,*ypopt),color="k",linestyle="--")
                        plt.show()
    
                        y_width = np.append(y_width,ypopt[2]*2)
                        y_model_errors = np.sqrt(np.diag(ypcov))
                        y_width_error = np.append(y_width_error,y_model_errors[2]*2)
                        
                        #Calculate the reduced chi-square values
                        dof = len(slice_y) - len(ypopt)
                        
                        y_chi = (1/dof)*np.sum( ( (slice_y - gaussian(ccd_y,*ypopt))/(slice_y_err) )**2 )
                        
                        y_red_chi[counter] = y_chi
                        
                    elif self.super_number == 1:
                        model = lambda x,a,w0:self.one_beam_model(x=x,a=a,w0=w0,z=(z_value - 0.5))
                        ypopt,ypcov = sp.curve_fit(model,ccd_y,slice_y,p0=[1,0.0001])
                        
                        plt.plot(ccd_y,slice_y)
                        plt.plot(ccd_y,model(ccd_y,*ypopt))
                        plt.show()
                        
                        dof = len(slice_y) - len(ypopt)
                        
                        y_chi = (1/dof)*np.sum( ( (slice_y - model(ccd_y,*ypopt))/(slice_y_err) )**2 )
                        
                        y_red_chi[counter] = y_chi
                        
                    elif self.super_number == 2:
                        pass
                    
                    elif self.super_number == 3:
                        if self.boundaries == True:
                            lims = ((0,0,0,0),(np.inf,np.inf,np.inf,np.inf))
                            model = lambda x,w0,a1,a2,a3:self.three_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,norm=False)
                            ypopt,ypcov = sp.curve_fit(model,ccd_y,slice_y,p0=[0.0001,1,1,1],bounds=lims)
                            
                        else:
                            model = lambda x,w0,a1,a2,a3:self.three_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,norm=False)
                            ypopt,ypcov = sp.curve_fit(model,ccd_y,slice_y,p0=[0.0001,1,1,1])
                    
                        plt.errorbar(ccd_y,slice_y,yerr=slice_y_err,ecolor="gray",barsabove=False)
                        plt.plot(ccd_y,model(ccd_y,*ypopt),color="k",linestyle="--")
                        plt.xlabel("Radial Distance from Propagation Axis / m")
                        plt.ylabel("Intensity at Position")
                        fname = "ModelFitY-Z" + str(z_value) + "-" + self.image_name + ".png"
                        plt.savefig(fname,dpi=300)
                        plt.show()
                        
                        dof = len(slice_y) - len(ypopt)
                        
                        y_chi = (1/dof)*np.sum( ( (slice_y - model(ccd_y,*ypopt))/(slice_y_err) )**2 )
                        
                        y_red_chi[counter] = y_chi
                        
                        #Get the normalised amplitudes of each of the components of the fitted model, and save them
                        amp1 = ypopt[1]
                        amp2 = ypopt[2]
                        amp3 = ypopt[3]
                        
                        totamp = amp1 + amp2 + amp3
                        
                        y_amp_vals_1[counter] = amp1/totamp
                        y_amp_vals_2[counter] = amp2/totamp
                        y_amp_vals_3[counter] = amp3/totamp
                        
                        #Get the errors on these amplitude values
                        popt_errors = np.sqrt(np.diag(ypcov))
                        
                        amp1_err = popt_errors[1]
                        amp2_err = popt_errors[2]
                        amp3_err = popt_errors[3]
                        
                        totamp_error = np.sqrt((amp1_err**2) + (amp2_err**2) + (amp3_err**2))
                        
                        y_amp_error_1[counter] = y_amp_vals_1[counter] * np.sqrt((amp1_err/amp1)**2 + (totamp_error/totamp)**2)
                        y_amp_error_2[counter] = y_amp_vals_2[counter] * np.sqrt((amp2_err/amp2)**2 + (totamp_error/totamp)**2)                        
                        y_amp_error_3[counter] = y_amp_vals_3[counter] * np.sqrt((amp3_err/amp3)**2 + (totamp_error/totamp)**2)
                        
                    elif self.super_number == 4:
                        if self.boundaries == True:
                            lims = ((0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf))
                            model = lambda x,w0,a1,a2,a3,a4:self.four_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,a4=a4,norm=False)
                            ypopt,ypcov = sp.curve_fit(model,ccd_y,slice_y,p0=[0.0001,1,1,1,1],bounds=lims)
                            
                        else:
                            model = lambda x,w0,a1,a2,a3,a4:self.four_beam_model(x=x,z=(z_value - 0.5),x0=0,w0=w0,a1=a1,a2=a2,a3=a3,a4=a4,norm=False)
                            ypopt,ypcov = sp.curve_fit(model,ccd_y,slice_y,p0=[0.0001,1,1,1,1])
                    
                        plt.errorbar(ccd_y,slice_y,yerr=slice_y_err,ecolor="gray",barsabove=False)
                        plt.plot(ccd_y,model(ccd_y,*ypopt),color="k",linestyle="--")
                        plt.xlabel("Radial Distance from Propagation Axis / m")
                        plt.ylabel("Intensity at Position")
                        fname = "ModelFitY-Z" + str(z_value) + "-" + self.image_name + ".png"
                        plt.savefig(fname,dpi=300)
                        plt.show()
                        
                        dof = len(slice_y) - len(ypopt)
                        
                        y_chi = (1/dof)*np.sum( ( (slice_y - model(ccd_y,*ypopt))/(slice_y_err) )**2 )
                        
                        y_red_chi[counter] = y_chi
                        
                        #Get the normalised amplitudes of each of the components of the fitted model, and save them
                        amp1 = ypopt[1]
                        amp2 = ypopt[2]
                        amp3 = ypopt[3]
                        amp4 = ypopt[4]
                        
                        totamp = amp1 + amp2 + amp3 + amp4
                        
                        y_amp_vals_1[counter] = amp1/totamp
                        y_amp_vals_2[counter] = amp2/totamp
                        y_amp_vals_3[counter] = amp3/totamp
                        y_amp_vals_4[counter] = amp4/totamp
                        
                        #Get the errors on these amplitude values
                        popt_errors = np.sqrt(np.diag(ypcov))
                        
                        amp1_err = popt_errors[1]
                        amp2_err = popt_errors[2]
                        amp3_err = popt_errors[3]
                        amp4_err = popt_errors[4]
                        
                        totamp_error = np.sqrt((amp1_err**2) + (amp2_err**2) + (amp3_err**2) + (amp4_err**2))
                        
                        y_amp_error_1[counter] = y_amp_vals_1[counter] * np.sqrt((amp1_err/amp1)**2 + (totamp_error/totamp)**2)
                        y_amp_error_2[counter] = y_amp_vals_2[counter] * np.sqrt((amp2_err/amp2)**2 + (totamp_error/totamp)**2)                        
                        y_amp_error_3[counter] = y_amp_vals_3[counter] * np.sqrt((amp3_err/amp3)**2 + (totamp_error/totamp)**2)
                        y_amp_error_4[counter] = y_amp_vals_4[counter] * np.sqrt((amp4_err/amp4)**2 + (totamp_error/totamp)**2)

        
                    print(x_holder.shape)
                    print(slice_x.shape)
                    
                    x_holder[:,counter] = slice_x
                    y_holder[:,counter] = slice_y

                    if self.super_number == 0:
                        x_mod_pos = self.centre_x + (xpopt[1]/(5.2*1e-6))
                        y_mod_pos = self.centre_y + (ypopt[1]/(5.2*1e-6))
                        
                        x_mid_mod = np.append(x_mid_mod,x_mod_pos)
                        y_mid_mod = np.append(y_mid_mod,y_mod_pos)
        
        if self.super_number == 0:
            print()
            print("Plot the position of the centre points")
            
            plt.plot(zvalues,x_mid)
            plt.plot(zvalues,x_mid_mod,linestyle="--")
            plt.show()
            
            plt.plot(zvalues,y_mid)
            plt.plot(zvalues,y_mid_mod,linestyle="--")
            plt.show()
            
        #Set the zero of the intensity heat map to be the minimum intensity
        x_holder = x_holder - np.amin(x_holder)
        y_holder = y_holder - np.amin(y_holder)
            
        #Normalise the intensity heat map
        x_holder = x_holder / np.amax(x_holder)
        y_holder = y_holder / np.amax(y_holder)
        
        print(np.amax(x_holder))
        print(np.amax(y_holder))
        
        print()
        print(x_holder.shape,y_holder.shape)
        print()
        
        
        print()
        print("Plot the Intensity Heat Map")
        
        x_slicelength = x_holder.shape[0] - 10
        x_radial_ticks = np.arange(5,x_slicelength + (x_slicelength/4),(x_slicelength/4))
        y_slicelength = y_holder.shape[0] - 10
        y_radial_ticks = np.arange(5,y_slicelength + (y_slicelength/4),(y_slicelength/4))
        
        print(x_slicelength,self.crop_x,x_radial_ticks)
        print((x_radial_ticks - self.crop_x))
        
        #plt.xticks(ticks=zvalues)
        #plt.yticks(ticks=ccd_x) 
        xmap = plt.imshow(x_holder,aspect="auto",interpolation="bicubic")
        plt.xticks(ticks=[0,10,20,30,40],labels=[-0.1,-0.05,0,0.05,0.1])
        plt.yticks(ticks=x_radial_ticks,labels=np.round((x_radial_ticks - (self.crop_x))*5.2*1e-3,2))
        plt.xlabel("Axial Distance from CCD to Focal Plane / m")
        plt.ylabel("Radial Distance from Propagation Axis / mm")
        cbar = plt.colorbar(xmap)
        cbar.set_label('Relative Intensity',rotation=90)
        fname = "IntensityMapX-" + self.image_name + ".png"
        plt.savefig(fname,dpi=300)
        plt.show()
        
        #plt.xticks(ticks=zvalues)
        #plt.yticks(ticks=ccd_y) 
        ymap = plt.imshow(y_holder,aspect="auto",interpolation="bicubic")
        plt.xticks(ticks=[0,10,20,30,40],labels=[-0.1,-0.05,0,0.05,0.1])
        plt.yticks(ticks=x_radial_ticks,labels=np.round((y_radial_ticks - (self.crop_y))*5.2*1e-3,2))
        plt.xlabel("Axial Distance from CCD to Focal Plane / m")
        plt.ylabel("Radial Distance from Propagation Axis / mm")
        cbar = plt.colorbar(ymap)
        cbar.set_label('Relative Intensity',rotation=90)
        fname = "IntensityPlotY-" + self.image_name + ".png"
        plt.savefig(fname,dpi=300)
        plt.show()
        
        #Plot the reduced chi-squared values
        plt.plot(zvalues-0.5,x_red_chi)
        plt.xlabel("Axial Distance from CCD to Focal Plane / m")
        plt.ylabel("Reduced Chi Square Value")
        fname = "ReducedChiSquaredX-" + self.image_name + ".png"
        plt.savefig(fname,dpi=300)
        plt.show()

        plt.plot(zvalues-0.5,y_red_chi)
        plt.xlabel("Axial Distance from CCD to Focal Plane / m")
        plt.ylabel("Reduced Chi Square Value")
        fname = "ReducedChiSquaredY-" + self.image_name + ".png"
        plt.savefig(fname,dpi=300)
        plt.show()
        
        print(x_red_chi)
        print(y_red_chi)
        print(np.mean(x_red_chi),np.mean(y_red_chi))
        
        x_chi_sum = 0
        y_chi_sum = 0
        x_count = 0
        y_count = 0
        
        for value in x_red_chi:
            if value != np.inf:
                x_chi_sum += value
                x_count += 1
                
        for value in y_red_chi:
            if value != np.inf:
                y_chi_sum += value
                y_count += 1
        
        print(x_chi_sum/x_count)
        print(y_chi_sum/y_count)

        
        if self.super_number == 0:
            print()
            
            print(len(x_width),len(x_width_error),len(y_width),len(y_width_error),len(zvalues))
            
            #Plot the widths in the x-direction
            plt.errorbar(zvalues,x_width,yerr=x_width_error,linestyle="",marker="x",ecolor="k",color="Blue")
            plt.plot(zvalues,width_function(zvalues,1.5*1e-4,0.05,0.5))
            plt.errorbar(zvalues,y_width,yerr=y_width_error,linestyle="",marker="x",ecolor="k",color="Green")
            plt.show()
            
            #Fit a function to the x and y widths
            x_popt,x_pcov = sp.curve_fit(width_function,zvalues,x_width,p0=[1.5*1e-4,0.05,0.5])
            y_popt,y_pcov = sp.curve_fit(width_function,zvalues,y_width,p0=[1.5*1e-4,0.05,0.5])
            
            #Plot the widths in the x-direction
            plt.errorbar(zvalues - 0.5,x_width/1e-3,yerr=x_width_error/1e-3,linestyle="",marker="x",ecolor="k",color="Blue",barsabove=True)
            plt.plot(zvalues - 0.5,width_function(zvalues,*x_popt)/1e-3,linestyle="--",color="royalblue")
            plt.errorbar(zvalues - 0.5,y_width/1e-3,yerr=y_width_error/1e-3,linestyle="",marker="x",ecolor="k",color="Green",barsabove=True)
            plt.plot(zvalues - 0.5,width_function(zvalues,*y_popt)/1e-3,linestyle="--",color="limegreen")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Width of the Beam / mm")
            plt.savefig("GaussianWaistProfile.png",dpi=300)
            plt.show()
        
            x_errors = np.sqrt(np.diag(x_pcov))
            y_errors = np.sqrt(np.diag(y_pcov))
        
            print("Waist X: ",x_popt[1]," +- ",x_errors[1])
            print("Waist Y: ",y_popt[1]," +- ",y_errors[1])
            print()
            print("rz X: ",x_popt[0]," +- ",x_errors[0])
            print("rz Y: ",y_popt[0]," +- ",y_errors[0])
            
            print()
            
            #Work out the reduced chi-squared value
            
            dof = len(x_width) - len(x_popt)
            
            x_chi = (1/dof)*np.sum( ( (x_width - width_function(zvalues,*x_popt))/(x_width_error) )**2 )
            y_chi = (1/dof)*np.sum( ( (y_width - width_function(zvalues,*y_popt))/(y_width_error) )**2 )
            
            print("Reduced chi-squareds: ",x_chi,y_chi)
            
            print()
            
            print("Now consider the points closer to the centre")
            
            lower_bound = np.where(zvalues == 0.45)[0][0]
            upper_bound = np.where(zvalues == 0.55)[0][0]
            
            zvalues = zvalues[lower_bound:upper_bound]
            x_width = x_width[lower_bound:upper_bound]
            y_width = y_width[lower_bound:upper_bound]
            x_width_error = x_width_error[lower_bound:upper_bound]
            y_width_error = y_width_error[lower_bound:upper_bound]
            
            #Fit a function to the x and y widths
            x_popt,x_pcov = sp.curve_fit(width_function,zvalues,x_width,p0=[1.5*1e-4,0.05,0.5])
            y_popt,y_pcov = sp.curve_fit(width_function,zvalues,y_width,p0=[1.5*1e-4,0.05,0.5])
            
            #Plot the widths in the x-direction
            plt.errorbar(zvalues - 0.5,x_width/1e-3,yerr=x_width_error/1e-3,linestyle="",marker="x",ecolor="k",color="Blue",barsabove=True)
            plt.plot(zvalues - 0.5,width_function(zvalues,*x_popt)/1e-3,linestyle="--",color="royalblue")
            plt.errorbar(zvalues - 0.5,y_width/1e-3,yerr=y_width_error/1e-3,linestyle="",marker="x",ecolor="k",color="Green",barsabove=True)
            plt.plot(zvalues - 0.5,width_function(zvalues,*y_popt)/1e-3,linestyle="--",color="limegreen")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Width of the Beam / mm")
            plt.savefig("GaussianWaistProfileSmall.png",dpi=300)
            plt.show()
            
            x_errors = np.sqrt(np.diag(x_pcov))
            y_errors = np.sqrt(np.diag(y_pcov))
            
            print("Waist X: ",x_popt[1]," +- ",x_errors[1])
            print("Waist Y: ",y_popt[1]," +- ",y_errors[1])
            print()
            print("rz X: ",x_popt[0]," +- ",x_errors[0])
            print("rz Y: ",y_popt[0]," +- ",y_errors[0])
            print()
    
            x_chi = (1/dof)*np.sum( ( (x_width - width_function(zvalues,*x_popt))/(x_width_error) )**2 )
            y_chi = (1/dof)*np.sum( ( (y_width - width_function(zvalues,*y_popt))/(y_width_error) )**2 )
            
            print("Reduced chi-squareds: ",x_chi,y_chi)
        
        #Plot the amplitude mixes of each model as a function of distance
        if self.super_number == 3:
            plt.errorbar(zvalues,x_amp_vals_1,linestyle="",marker="x",color="Blue",label=("p = " + str(self.pls[0,0])))
            plt.errorbar(zvalues,x_amp_vals_2,linestyle="",marker="x",color="Green",label=("p = " + str(self.pls[1,0])))
            plt.errorbar(zvalues,x_amp_vals_3,linestyle="",marker="x",color="Red",label=("p = " + str(self.pls[2,0])))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesX" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()
            
            plt.errorbar(zvalues,x_amp_vals_1,yerr=x_amp_error_1,linestyle="",marker="x",color="Blue",label=("p = " + str(self.pls[0,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,x_amp_vals_2,yerr=x_amp_error_2,linestyle="",marker="x",color="Green",label=("p = " + str(self.pls[1,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,x_amp_vals_3,yerr=x_amp_error_3,linestyle="",marker="x",color="Red",label=("p = " + str(self.pls[2,0])),ecolor="k",barsabove=True)
            plt.ylim((-3,4))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesWithErrorsX" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()
        
            plt.errorbar(zvalues,y_amp_vals_1,linestyle="",marker="x",color="Blue",label=("p = " + str(self.pls[0,0])))
            plt.errorbar(zvalues,y_amp_vals_2,linestyle="",marker="x",color="Green",label=("p = " + str(self.pls[1,0])))
            plt.errorbar(zvalues,y_amp_vals_3,linestyle="",marker="x",color="Red",label=("p = " + str(self.pls[2,0])))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesY" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()
            
            plt.errorbar(zvalues,y_amp_vals_1,yerr=y_amp_error_1,linestyle="",marker="x",color="Blue",label=("p = " + str(self.pls[0,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,y_amp_vals_2,yerr=y_amp_error_2,linestyle="",marker="x",color="Green",label=("p = " + str(self.pls[1,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,y_amp_vals_3,yerr=y_amp_error_3,linestyle="",marker="x",color="Red",label=("p = " + str(self.pls[2,0])),ecolor="k",barsabove=True)
            plt.ylim((-3,4))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesYWithErrors" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()
            
        #Plot the amplitude mixes of each model as a function of distance
        if self.super_number == 4:
            plt.errorbar(zvalues,x_amp_vals_1,linestyle="--",marker="x",color="cornflowerblue",label=("p = " + str(self.pls[0,0])))
            plt.errorbar(zvalues,x_amp_vals_2,linestyle="--",marker="x",color="limegreen",label=("p = " + str(self.pls[1,0])))
            plt.errorbar(zvalues,x_amp_vals_3,linestyle="--",marker="x",color="firebrick",label=("p = " + str(self.pls[2,0])))
            plt.errorbar(zvalues,x_amp_vals_4,linestyle="--",marker="x",color="goldenrod",label=("p = " + str(self.pls[3,0])))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesX" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()
            
            plt.errorbar(zvalues,x_amp_vals_1,yerr=x_amp_error_1,linestyle="--",marker="x",color="cornflowerblue",label=("p = " + str(self.pls[0,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,x_amp_vals_2,yerr=x_amp_error_2,linestyle="--",marker="x",color="limegreen",label=("p = " + str(self.pls[1,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,x_amp_vals_3,yerr=x_amp_error_3,linestyle="--",marker="x",color="firebrick",label=("p = " + str(self.pls[2,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,x_amp_vals_4,yerr=x_amp_error_4,linestyle="--",marker="x",color="goldenrod",label=("p = " + str(self.pls[3,0])),ecolor="k",barsabove=True)
            plt.ylim((-3,4))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesWithErrorsX" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()
        
            plt.errorbar(zvalues,y_amp_vals_1,linestyle="--",marker="x",color="cornflowerblue",label=("p = " + str(self.pls[0,0])))
            plt.errorbar(zvalues,y_amp_vals_2,linestyle="--",marker="x",color="limegreen",label=("p = " + str(self.pls[1,0])))
            plt.errorbar(zvalues,y_amp_vals_3,linestyle="--",marker="x",color="firebrick",label=("p = " + str(self.pls[2,0])))
            plt.errorbar(zvalues,y_amp_vals_4,linestyle="--",marker="x",color="goldenrod",label=("p = " + str(self.pls[3,0])))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesY" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()
            
            plt.errorbar(zvalues,y_amp_vals_1,yerr=y_amp_error_1,linestyle="--",marker="x",color="cornflowerblue",label=("p = " + str(self.pls[0,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,y_amp_vals_2,yerr=y_amp_error_2,linestyle="--",marker="x",color="limegreen",label=("p = " + str(self.pls[1,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,y_amp_vals_3,yerr=y_amp_error_3,linestyle="--",marker="x",color="firebrick",label=("p = " + str(self.pls[2,0])),ecolor="k",barsabove=True)
            plt.errorbar(zvalues,y_amp_vals_4,yerr=y_amp_error_4,linestyle="--",marker="x",color="goldenrod",label=("p = " + str(self.pls[3,0])),ecolor="k",barsabove=True)
            plt.ylim((-3,4))
            plt.legend(loc="best")
            plt.xlabel("Axial Distance from CCD to Focal Plane / m")
            plt.ylabel("Normalised Amplitude")
            fname = "ComponentAmplitudesYWithErrors" + self.image_name + ".png"
            plt.savefig(fname,dpi=300)
            plt.show()

    #Gets the x position of the maximum pixel
    def max_pixel_x(self,array):
        max_x = np.where(array == np.amax(array))[1][0]
    
        return max_x

    #Gets the y position of the maximum pixel    
    def max_pixel_y(self,array):
        max_y = np.where(array == np.amax(array))[0][0]
        
        return max_y
    
    #Crops the array provided 
    def crop(self,array):    
        crop = array[self.centre_y - self.crop_y:self.centre_y + self.crop_y, self.centre_x - self.crop_x:self.centre_x + self.crop_x]
        
        return crop
    
    def one_dim_electric_field(self,x,p=0,l=0,w0=1e-3,z=0,wlen=1024*1e-9,x0=0):
        
        zr = (np.pi * (w0**2)) / wlen
        
        w = waist(w0,z,zr)
        
        k = (2 * np.pi)/wlen
        
        if z != 0:
            rad = z*(1+(zr/z)**2)
            
        psi = np.arctan(z/zr)
        
        x = x - x0
        
        if l == 0:
            A = np.sqrt((2*math.factorial(p))/(2*np.pi*math.factorial(p + np.abs(l))))
        else: 
            A = np.sqrt((2*math.factorial(p))/(np.pi*math.factorial(p + np.abs(l))))
        
        if str(type(x)) == "<class 'numpy.ndarray'>":
            for i in range(len(x)):
                if x[i] < 0:
                    if z == 0:
                        phase = np.exp((1j*k*z) + (1j * l * (-1*np.pi/2))  - (1j*(np.abs(l) + 2*p + 1)*psi))
            
                    else: 
                        phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) + (1j * l * (-1*np.pi/2))  - (1j*(np.abs(l) + 2*p + 1)*psi))
                
                elif x[i] == 0:
                    if z == 0:
                        phase = np.exp((1j*k*z)  - (1j*(np.abs(l) + 2*p + 1)*psi))
            
                    else: 
                        phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) - (1j*(np.abs(l) + 2*p + 1)*psi))
                else:
                    if z == 0:
                        phase = np.exp((1j*k*z) + (1j * l * (np.pi/2))  - (1j*(np.abs(l) + 2*p + 1)*psi))
            
                    else: 
                        phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) + (1j * l * (np.pi/2))  - (1j*(np.abs(l) + 2*p + 1)*psi))
            
        else:
            if x < 0:
                if z == 0:
                    phase = np.exp((1j*k*z) + (1j * l * (-1*np.pi))  - (1j*(np.abs(l) + 2*p + 1)*psi))
        
                else: 
                    phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) + (1j * l * (-1*np.pi))  - (1j*(np.abs(l) + 2*p + 1)*psi))
            
            elif x == 0:
                if z == 0:
                    phase = np.exp((1j*k*z)  - (1j*(np.abs(l) + 2*p + 1)*psi))
        
                else: 
                    phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) - (1j*(np.abs(l) + 2*p + 1)*psi))
            else:
                if z == 0:
                    phase = np.exp((1j*k*z) + (1j * l * (np.pi))  - (1j*(np.abs(l) + 2*p + 1)*psi))
        
                else: 
                    phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) + (1j * l * (np.pi))  - (1j*(np.abs(l) + 2*p + 1)*psi))
                    
        complex_amplitude = A * (w0/w) * (((np.sqrt(2) * x)/w)**np.abs(l)) * (lag_poly(((2 * (x**2))/(w**2)),p,l)) * np.exp(-(x**2)/(w**2)) * phase
                
        electric_field = complex_amplitude * np.exp(-1j * k * z)
        
        return electric_field
        
    def one_beam_model(self,x,a,w0,z):
        tem = lg.LagGauss(p=self.pls[0,0],l=self.pls[0,1],w0=w0,wlen=1024*1e-9,z=z)
                
        intensity = tem.one_dim_intensity(x,amp=a)# + c
        
        return intensity
    
    def two_beam_model(self,x,z,w0,x0,a1=1,a2=1,norm=True):
        tem1 = lg.LagGauss(p=self.pls[0,0],l=self.pls[0,1])
        tem2 = lg.LagGauss(p=self.pls[1,0],l=self.pls[1,1])
        
        elec_field = (a1 * tem1.one_dim_electric_field(x,p=self.pls[0,0],l=self.pls[0,1],z=z,w0=w0,x0=x0)) + (a2 * tem2.one_dim_electric_field(x,p=self.pls[1,0],l=self.pls[1,1],z=z,w0=w0,x0=x0))
        
        intensity = elec_field * np.conj(elec_field)
        
        if norm == True:
            intensity = intensity / np.amax(intensity)
        
        return intensity
    
    def three_beam_model(self,x,z,w0,x0,a1=1,a2=1,a3=1,norm=True):
        tem1 = lg.LagGauss(p=self.pls[0,0],l=self.pls[0,1])
        tem2 = lg.LagGauss(p=self.pls[1,0],l=self.pls[1,1])
        tem3 = lg.LagGauss(p=self.pls[2,0],l=self.pls[2,1])
        
        elec_field = (a1 * tem1.one_dim_electric_field(x,p=self.pls[0,0],l=self.pls[0,1],z=z,w0=w0,x0=x0)) + (a2 * tem2.one_dim_electric_field(x,p=self.pls[1,0],l=self.pls[1,1],z=z,w0=w0,x0=x0)) + (a3 * tem3.one_dim_electric_field(x,p=self.pls[2,0],l=self.pls[2,1],z=z,w0=w0,x0=x0))
        
        intensity = np.real(elec_field * np.conj(elec_field))
        
        if norm == True:
            intensity = intensity / np.amax(intensity)
        
        return intensity
    
    def four_beam_model(self,x,z,w0,x0,a1=1,a2=1,a3=1,a4=1,norm=True):
        tem1 = lg.LagGauss(p=self.pls[0,0],l=self.pls[0,1])
        tem2 = lg.LagGauss(p=self.pls[1,0],l=self.pls[1,1])
        tem3 = lg.LagGauss(p=self.pls[2,0],l=self.pls[2,1])
        tem4 = lg.LagGauss(p=self.pls[3,0],l=self.pls[3,1])
        
        elec_field = (a1 * tem1.one_dim_electric_field(x,p=self.pls[0,0],l=self.pls[0,1],z=z,w0=w0,x0=x0)) + (a2 * tem2.one_dim_electric_field(x,p=self.pls[1,0],l=self.pls[1,1],z=z,w0=w0,x0=x0)) + (a3 * tem3.one_dim_electric_field(x,p=self.pls[2,0],l=self.pls[2,1],z=z,w0=w0,x0=x0)) + + (a4 * tem4.one_dim_electric_field(x,p=self.pls[3,0],l=self.pls[3,1],z=z,w0=w0,x0=x0))
        
        intensity = np.real(elec_field * np.conj(elec_field))
        
        if norm == True:
            intensity = intensity / np.amax(intensity)
        
        return intensity
    
    def counts_per_second(self,file,im):
        filename_split = file.split("-")
        
        im_counts_per_sec = im / float(filename_split[-2])
        
        return im_counts_per_sec
    
trial = function_fit(directory="C:/Users/shado/OneDrive/Desktop/L4 Dissertation/LG Beam Data/Superposition024",pls=[[0,0],[2,0],[4,0]],image_name="Super024",crop_x=200,crop_y=200,centre_x=709,centre_y=563,super_number=3,boundaries=True)
trial.analyse_directory()
 
"""print(os.path)       
os.chdir("C:/Users/shado/OneDrive/Desktop/L4 Dissertation/LG Beam Data/TEM00")
print(os.listdir())

file = os.listdir()[5].split("-")
print(file)

trial = function_fit(directory="C:/Users/shado/OneDrive/Desktop/L4 Dissertation/LG Beam Data/Superposition135",pls=np.asarray([[1,0],[3,0],[4,0]]),crop_x=70,crop_y=70)
xvals = np.arange(-10,10,0.01)
intensity = trial.three_beam_model(xvals,z=1000000,a1=0,a2=1,a3=2,w0=3,x0=0,norm=True)

plt.plot(xvals,intensity)
plt.show()

vals = np.asarray([[1,1,8],[2,3,2],[11,1,3],[12,1,3]])

print(vals)
print(np.where(vals == np.amax(vals)))

im = PIL.Image.open(os.listdir()[5])

im.show()    

im = np.asarray(im)[:,:,0]

print(im.shape)
print(np.where(im == np.amax(im))[0])

data = trial.image_crop(os.listdir()[5])

print(data.shape)
im = PIL.Image.fromarray(data)
im.show()"""