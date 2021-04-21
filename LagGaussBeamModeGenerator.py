#Code written by Mitch Walker
#2021
#Durham University Physics Dept

#Import modules
import numpy as np
import PIL
import matplotlib.pyplot as plt
import scipy.special as spec
import scipy.integrate as integ
import math
import os
from scipy import ndimage
import sys
#sys.path.append("C:\Users\shado\OneDrive\Desktop\L4 Dissertation\LUTInterferencePatterns\TEM01 Profile Data\Profile Fitting")
from scipy.optimize import curve_fit


#Calculate w(z) as a function of beam waist w(0) and Rayleigh range zR
def waist(w0,z,zr):
    return w0 * np.sqrt(1 + (z/zr)**2)

#Calculate a matrix of radius values
def radius(a=1):
    cartes = np.empty((1000,1000,2))
    
    for i in range(cartes.shape[0]):
        for j in range(cartes.shape[0]):
            cartes[i,j,0] = ((j) - 500)
            
            cartes[j,i,1] = ((j) - 500)
    
    radius = np.sqrt(cartes[:,:,0]**2 + cartes[:,:,1]**2) * a
    
    return radius

#Calculate a matrix of polar angle values
def polar():
    cartes = np.empty((1000,1000,2))
    
    for i in range(cartes.shape[0]):
        for j in range(cartes.shape[0]):
            cartes[i,j,0] = ((j) - 500)
            
            cartes[j,i,1] = ((j) - 500)
    
    x0 = 500
    y0 = 500
    
    phase = np.empty((1000,1000))
    
    for i in range(cartes.shape[0]):
        for j in range(cartes.shape[0]):
            
            if j == x0:
                if i > y0:
                    phase[i,j] = np.pi/2
                elif i < y0:
                    phase[i,j] = (3/2)*np.pi
                elif i == y0:
                    phase[i,j] = 0
            elif i == y0:
                if j < x0:
                    phase[i,j] = np.pi
                elif j > x0:
                    phase[i,j] = 0
                elif j == x0:
                    phase[i,j] = 0
            elif j>x0 and i>y0:
                phase[i,j] = np.arctan(np.abs(cartes[i,j,1]/cartes[i,j,0]))
            elif j<x0 and i>y0:
                phase[i,j] = (np.pi) + np.arctan(cartes[i,j,1]/cartes[i,j,0])
            elif j<x0 and i<y0:
                phase[i,j] = np.pi + np.arctan(cartes[i,j,1]/cartes[i,j,0])
            elif j>x0 and i<y0:
                phase[i,j] = ((2)*np.pi) + np.arctan(cartes[i,j,1]/cartes[i,j,0])
                
            if np.isnan(phase[i,j]) == True:
                print("Not a number ", i, " ", j)
                phase[i,j] = 0
                
    return phase

#Function to generate generalised Laguerre polynomials
def lag_poly(x,p,l):
    return spec.assoc_laguerre(x,n=p,k=np.abs(l))

    
class LagGauss:
    def __init__(self,p=0,l=0,w0=1e-3,z=0,zr=1,wlen=1024*1e-9):
        self.p = p
        self.l = l
        self.w0 = w0
        self.z = z
        self.zr = zr
        self.wlen = wlen
    
    #Generate a 1D intensity profile
    def one_dim_intensity(self,x,amp=1,x0=0):
        electric_field = self.one_dim_electric_field(x,self.p,self.l,self.w0,self.z,self.wlen,x0)
    
        intensity = np.real(electric_field * np.conj(electric_field))
        
        intensity = amp * (intensity / np.amax(intensity))
        
        return intensity
    
    #Generate the 2D intensity pattern of a Laguerre-Gauss beam 
    def two_dim_intensity(self,amp=255):
        electric_field = self.two_dim_electric_field(p=self.p,l=self.l,w0=self.w0,z=self.z,wlen=self.wlen)
        
        intensity = np.real(electric_field * np.conj(electric_field))
        
        intensity = amp * (intensity / np.amax(intensity))
        
        return intensity
    
    def axial_intensity(self,z,amp=1):
        electric_field = self.axial_electric_field(z,p=self.p,l=self.l,w0=self.w0,wlen=self.wlen)
        
        intensity = np.real(electric_field * np.conj(electric_field))
        
        intensity = amp * (intensity / np.amax(intensity))
        
        return intensity
    
    def slice_intensity(self,x,z,amp=255):
        electric_field = self.slice_electric_field(x,z)
        
        intensity = np.real(electric_field * np.conj(electric_field))
        
        intensity = amp * (intensity / np.amax(intensity))
                
        return intensity
    
    #Generate the 1D electric field
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
    
    #Generate the 2D electric field
    def two_dim_electric_field(self,p=0,l=0,w0=1e-3,z=0,wlen=1024*1e-9,rgrad=5.2*1e-6):
        x = radius(rgrad)
        phase = polar()
        
        zr = (np.pi * (w0**2)) / wlen
        
        w = waist(w0,z,zr)
        
        k = (2 * np.pi)/wlen
        
        if z != 0:
            rad = z*(1*(zr/z)**2)
            
        psi = np.arctan(z/zr)
       
        if l == 0:
            A = np.sqrt((2*math.factorial(p))/(2*np.pi*math.factorial(p + np.abs(l))))
        else: 
            A = np.sqrt((2*math.factorial(p))/(np.pi*math.factorial(p + np.abs(l))))
        
        for i in range(len(x)):
            if z == 0:
                phase = np.exp((1j*k*z) + (1j * l * (-1*np.pi))  - (1j*(np.abs(l) + 2*p + 1)*psi))
    
            else: 
                phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) + (1j * l * phase)  - (1j*(np.abs(l) + 2*p + 1)*psi))
            
        complex_amplitude = A * (w0/w) * (((np.sqrt(2) * x)/w)**np.abs(l)) * (lag_poly(((2 * (x**2))/(w**2)),p,l)) * np.exp(-(x**2)/(w**2)) * phase
        
        electric_field = complex_amplitude * np.exp(-1j * k * z)
                
        return electric_field
    
    #Generate the electric field on-axis over a given range of z
    def axial_electric_field(self,z,p=0,l=0,w0=1e-3,wlen=1024*1e-9):
        zr = (np.pi * (w0**2)) / wlen
        
        w = waist(w0,z,zr)
        
        k = (2 * np.pi)/wlen
        
        rad = np.empty(len(z))
        
        for i in range(len(z)):
            if z[i] != 0:
                rad[i] = z[i]*(1+(zr/z[i])**2)
            else:
                rad[i] = np.inf
            
        psi = np.arctan(z/zr)
        
        x = 0
        
        if l == 0:
            A = np.sqrt((2*math.factorial(p))/(2*np.pi*math.factorial(p + np.abs(l))))
        else: 
            A = np.sqrt((2*math.factorial(p))/(np.pi*math.factorial(p + np.abs(l))))
        
        phase = np.exp((1j*k*z) + (1j * l * (-1*np.pi))  - (1j*(np.abs(l) + 2*p + 1)*psi))
        
        complex_amplitude = A * (w0/w) * (((np.sqrt(2) * x)/w)**np.abs(l)) * (lag_poly(((2 * (x**2))/(w**2)),p,l)) * np.exp(-(x**2)/(w**2)) * phase
    
        electric_field = complex_amplitude * np.exp(-1j * k * z)
        
        return electric_field
    
    def slice_electric_field(self,x,z):
        electric_field = np.empty((len(x),len(z)))*(0+0j)
        
        for i in range(len(z)):
            zval = z[i]
            
            electric_slice = self.one_dim_electric_field(x,p=self.p,l=self.l,w0=self.w0,z=zval,wlen=self.wlen,x0=0)
            
            electric_field[:,i] = electric_slice
        
        return electric_field
    
    def two_dim_phase(self,p=0,l=0,w0=1e-3,z=0,wlen=1024*1e-9,rgrad=15*1e-6):
        x = radius(rgrad)
        phase = polar()
        
        zr = (np.pi * (w0**2)) / wlen
        
        k = (2 * np.pi)/wlen
        
        if z != 0:
            rad = z*(1*(zr/z)**2)
            
        psi = np.arctan(z/zr)
        
        for i in range(len(x)):
            if z == 0:
                phase = np.exp((1j*k*z) + (1j * l * (-1*np.pi))  - (1j*(np.abs(l) + 2*p + 1)*psi))
    
            else: 
                phase = np.exp((1j*k*z) + (1j*k*((x**2)/(2*rad))) + (1j * l * phase)  - (1j*(np.abs(l) + 2*p + 1)*psi))
    
    #Plot the 1D intensity as a function of the radius
    def plot_1D_intensity(self,x=np.arange(-5*1e-3,5*1e-3,1e-8),amp=1,color="Green"):
        intensity = self.one_dim_intensity(x,amp)
        
        plt.plot(x/1e-3,intensity,color=color)
        plt.ylabel("Normalised Intensity")
        plt.xlabel("Radius / mm")
        plt.show()
        
    def plot_axial_intensity(self,z=np.arange(-250*1e-3,250*1e-3,0.005*1e-3),amp=1,color="Green"):
        intensity = self.axial_intensity(z,amp=amp)
        
        plt.plot(z/1e-3,intensity,color=color)
        plt.ylabel("Normalised Intensity")
        plt.xlabel("Axial Distance from Waist / mm")
        plt.show()
    
    #Show the intensity pattern in 2D
    def show_2D_intensity_image(self,amp=255):
        im = PIL.Image.fromarray(self.two_dim_intensity(amp))
        im.show()
        
    #Save a plot of the intensity in 1D
    def save_1D_intensity_plot(self,fname,x=np.arange(-5*1e-3,5*1e-3,1e-8),colour="Green"):
        intensity = self.one_dim_intensity(x,self.p,self.l,self.w0,self.z,self.zr)
        intensity = intensity / np.amax(intensity)
        
        plt.plot(x,intensity,color=colour)
        plt.ylabel("Normalised Intensity")
        plt.xlabel("Radius")
        plt.savefig(fname,dpi=300)
        plt.show()
        
    def save_axial_intensity_plot(self,fname,z=np.arange(-250*1e-3,250*1e-3,0.005*1e-3),amp=1,color="Green"):
        intensity = self.axial_intensity(z,amp=amp)
        
        plt.plot(z/1e-3,intensity,color=color)
        plt.ylabel("Normalised Intensity")
        plt.xlabel("Axial Distance from Waist / mm")
        plt.savefig(fname,dpi=300)
        plt.show()
    
    #Save an image of the 2d Intensity pattern
    def save_2D_intensity_image(self,fname,amp=255):
        im = PIL.Image.fromarray(self.lgbeam(amp))
        im.save(fname)
        
    #Find the 1/e^2 radius in the radial direction
    def radial_waist():
        pass
    
    #Find the 1/e^2 radius in the axial direction
    def axial_waist():
        pass

class Superposition():
    def __init__(self,pls,w0=1e-3,rgrad=5.2*1e-6):
        self.pls = np.asarray(pls)
        self.rgrad = rgrad
        self.w0 = w0
        
    def one_dim_electric_super(self,pls,x):
        elec = np.complex64(np.zeros(len(x)))
        
        tem = LagGauss(p=0,l=0)
        
        for i in range(pls.shape[0]):
            elec += tem.one_dim_electric_field(x,p=pls[i,0],l=pls[i,1],w0=self.w0)
            
        return elec
    
    def one_dim_intensity_super(self,x,amp):
        elec = self.one_dim_electric_super(self.pls,x)
        
        intensity = np.real(elec * np.conj(elec))
        
        intensity = amp * (intensity / np.amax(intensity))
        
        return intensity
    
    def two_dim_electric_super(self,pls):
        elec = np.complex64(np.zeros((1000,1000)))
        
        tem = LagGauss(p=0,l=0)
        
        for i in range(pls.shape[0]):
            elec += tem.two_dim_electric_field(p=pls[i,0],l=pls[i,1],w0=self.w0,rgrad=self.rgrad)
            
        return elec
    
    def two_dim_intensity_super(self,amp):
        elec = self.two_dim_electric_super(self.pls)
        
        intensity = np.real(elec * np.conj(elec))
        
        intensity = amp * (intensity / np.amax(intensity))
        
        return intensity

    def axial_electric_super(self,pls,z):
        elec = np.complex64(np.zeros(len(z)))
        
        tem = LagGauss(p=0,l=0)
        
        for i in range(pls.shape[0]):
            elec += tem.axial_electric_field(z,p=pls[i,0],l=pls[i,1],w0=self.w0)
            
        return elec
    
    def axial_intensity_super(self,z,amp):
        elec = self.axial_electric_super(self.pls,z)
        
        intensity = np.real(elec * np.conj(elec))
        
        intensity = amp * (intensity / np.amax(intensity))
        
        return intensity
    
    def slice_electric_super(self,x,z,pls=None):
        elec = np.empty((len(x),len(z)))*(0+0j)
        
        if pls == None:
            pls = self.pls
    
        for i in range(pls.shape[0]):
            tem = LagGauss(p=pls[i,0],l=pls[i,1],w0=self.w0,wlen=1024*1e-9)
            
            elec += tem.slice_electric_field(x,z)
            
        return elec
    
    def slice_intensity_super(self,x,z,amp=255,pls=None):
        elec = self.slice_electric_super(x,z,pls=pls)
        
        intensity = np.real(elec * np.conj(elec))
        
        intensity = amp * (intensity / np.amax(intensity))
        
        return intensity
    
    
    def plot_1D_intensity(self,x,amp=1):
        intensity = self.onedimintensuper(x,amp)
        
        plt.plot(x/1e-3,intensity)
        plt.xlabel("Radius / mm")
        plt.ylabel("Normalised Intensity")
        plt.show()
        
        
    def save_1D_intensity_plot(self,fname,x,amp=1):
        intensity = self.one_dim_intensity_super(x,amp)
        
        plt.plot(x/1e-3,intensity)
        plt.xlabel("Radius / mm")
        plt.ylabel("Normalised Intensity")
        plt.savefig(fname,dpi=300)
        plt.show()
        
    def show_2D_intensity(self,amp=255):
        intensity = self.two_dim_intensity_super(amp)
        
        im = PIL.Image.fromarray(intensity)
        im.show()
        
    def save_2D_intensity_image(self,fname,amp=255):
        intensity = self.two_dim_intensity_super(amp)
        
        im = PIL.Image.fromarray(intensity)
        im = im.convert("P")
        im.save(fname)
        
    def plot_axial_intensity(self,z,amp=1):
        intensity = self.axial_intensity_super(z,amp)
        
        plt.plot(z/1e-3,intensity)
        plt.xlabel("Axial Distance from Waist / mm")
        plt.ylabel("Normalised Intensity")
        plt.show()
        
    def save_axial_intensity_plot(self,fname,z,amp=1):
        intensity = self.axial_intensity_super(z,amp)
        
        plt.plot(z/1e-3,intensity)
        plt.xlabel("Axial Distance from Waist / mm")
        plt.ylabel("Normalised Intensity")
        plt.savefig(fname,dpi=300)
        plt.show()

"""tem = LagGauss(p=0,l=0,w0=0.13*1e-3,z=0)
xvals = np.arange(-0.5,0.5,0.005)*1e-3
zvals = np.arange(-0.1,0.1,0.001)
superposition = Superposition(pls=[[1,0],[4,0],[6,0]],w0=0.13*1e-3)
array = superposition.two_dim_intensity_super(amp=1)#superposition.slice_intensity_super(xvals,zvals)
#im = PIL.Image.fromarray(array)
#im.show()

array = array / 255

x_radial_length = len(xvals) - 10
x_axial_length = len(zvals)
x_radial_ticks = np.arange(5,x_radial_length + (x_radial_length/4),(x_radial_length/4))
x_axial_ticks = np.arange(0,x_axial_length + (x_axial_length/4),(x_axial_length/4))

xmap = plt.imshow(array,aspect="auto",interpolation="bicubic")
plt.xticks(ticks=x_axial_ticks,labels=np.round((x_axial_ticks - (x_axial_length/2))*1e-3,2))
plt.yticks(ticks=x_radial_ticks,labels=np.round((x_radial_ticks - (100))*5.2*1e-3,2))
plt.xlabel("Axial Distance from CCD / m")
plt.ylabel("Radial Distance from Propagation Axis / mm")
cbar = plt.colorbar(xmap)
cbar.set_label('Relative Intensity',rotation=90)
fname = "TheoryIntensityRadial-" + "Super146" + ".png"
plt.savefig(fname,dpi=300)
plt.show()"""

"""xvals = np.arange(-0.1,0.1,0.001)

superposition = Superposition(pls=[[1,0],[4,0],[6,0]],w0=0.13*1e-3)
array = superposition.axial_intensity_super(xvals,1)

maxima = np.asarray([])

i0 = np.amax(array)

for i in range(len(array) - 2):
    j = i + 1
    
    if array[j-1] < array[j]:
        if array[j+1] < array[j]:
            if array[j] != i0:
                maxima = np.append(maxima,array[j])
            
ratio = np.amax(maxima)/i0

print("The ratio is ", ratio)

plt.plot(xvals,array)"""
            
    
    
    


"""tem = LagGauss(p=0,l=0,w0=0.13*1e-3,z=0)
xvals = np.arange(-0.5,0.5,0.005)*1e-3
intensity = tem.one_dim_intensity(xvals,amp=1,x0=0)
print(intensity)

plt.plot(xvals,intensity)"""

"""tem = LagGauss(p=0,l=0,w0=0.13*1e-3)
xvals = np.arange(-0.5,0.5,0.005)*1e-3
zvals = np.arange(-250,250,0.005)*1e-3
tem.plot_1D_intensity(xvals)
#tem.show_2D_intensity_image()
tem.plot_axial_intensity(zvals)

pvals = [0,1,2,3,4,5,6]
lvals = [0,1,2,3,4,5,6]

for i in range(len(pvals)):
    tem = LagGauss(p=pvals[i],l=0,w0=0.13*1e-3)
    
    xvals = np.arange(-0.5,0.5,0.005)*1e-3
    
    elec = tem.one_dim_electric_field(xvals,p=pvals[i],l=0,w0=0.13*1e-3,z=0,wlen=1024*1e-9,x0=0)

    plt.plot(xvals/1e-3,elec,linestyle="--",label=("p = " + str(pvals[i])))
    plt.legend(loc="best")
    
plt.show()

for i in range(len(pvals)):
    print("p = ",pvals[i])
    
    tem = LagGauss(p=pvals[i],l=0,w0=0.13*1e-3)
    
    xvals = np.arange(-0.5,0.5,0.005)*1e-3
    
    zvals = np.arange(-100,100,0.005)*1e-3
    
    radial_elec = tem.one_dim_electric_field(xvals,p=pvals[i],l=0,w0=0.13*1e-3,z=0,wlen=1024*1e-9,x0=0)
    
    radial_elec = radial_elec / np.amax(np.real(radial_elec))
    
    axial_elec = tem.axial_electric_field(zvals,p=pvals[i],l=0,w0=0.13*1e-3,wlen=1024*1e-9)
    
    axial_elec = axial_elec / np.amax(np.real(axial_elec))
    
    radial_intensity = radial_elec * np.conj(radial_elec)
    axial_intensity = axial_elec * np.conj(axial_elec)
    
    plt.plot(xvals/1e-3,np.real(radial_elec),linestyle="--",label="Real Amplitude")
    plt.plot(xvals/1e-3,np.imag(radial_elec),linestyle="--",label="Imaginary Amplitude")
    plt.plot(xvals/1e-3,radial_intensity,linestyle="-",label="Intensity",color="k")
    plt.xlabel("Radial Distance / mm")
    plt.ylabel("Normalised Amplitude")
    plt.legend(loc="best")
    fname = "RadialIntensityP-" + str(pvals[i])
    plt.savefig(fname,dpi=300)
    plt.show()
    
    plt.plot(zvals/1e-3,np.real(axial_elec),linestyle="--",label="Real Amplitude")
    plt.plot(zvals/1e-3,np.imag(axial_elec),linestyle="--",label="Imaginary Amplitude")
    plt.plot(zvals/1e-3,axial_intensity,linestyle="-",label="Intensity",color="k")
    plt.xlabel("Axial Distance / mm")
    plt.ylabel("Normalised Amplitude")
    plt.legend(loc="best")
    fname = "AxialIntensityP-" + str(pvals[i])
    plt.savefig(fname,dpi=300)
    plt.show()    """
    

    
"""plt.xlabel("Radial Distance / mm")
plt.ylabel("Electric Field")
plt.legend(loc="best")
plt.show()

gauss = LagGauss(p=0,l=0,w0=0.13*1e-3)
zvals = np.arange(-250*1e-3,250*1e-3,0.005*1e-3)
xvals = np.arange(-0.5,0.5,0.005)*1e-3
gaussz = gauss.axial_intensity(zvals)
gaussx = gauss.one_dim_intensity(xvals,amp=1)

gaussxw = np.amax(gaussx) * (1/np.exp(1)**2)
gausszw = np.amax(gaussz) * (1/np.exp(1)**2)

gaussz_prime = gaussz - gausszw
gaussx_prime = gaussx - gaussxw

delta_x = np.abs(np.where(gaussx == np.amax(gaussx))[0][0] - np.where(gaussx_prime == np.amin(np.abs(gaussx_prime)))[0][0])
delta_z = np.abs(np.where(gaussz == np.amax(gaussz))[0][0] - np.where(gaussz_prime == np.amin(np.abs(gaussz_prime)))[0][0])

gauss_vol = delta_x * delta_x * delta_z

p = [0,1,2,3,4,5,6]


for i in range(len(p)):
    for j in range(len(p)):  
        superpose = Superposition([(p[i],0),(0,p[j])],w0=0.13*1e-3,rgrad=1*1e-6)
            
        fname = "p" + str(p[i]) + "l" + str(p[j]) + "twodim.bmp"
        faxial = "p" + str(p[i]) + "l" + str(p[j]) + "axial.png"
        foned = "p" + str(p[i]) + "l" + str(p[j]) + "onedim.png"
        
        #superpose.save_2D_intensity_image(fname)
        
        zvals = np.arange(-250*1e-3,250*1e-3,0.005*1e-3)
        
        gauss = LagGauss(p=0,l=0,w0=0.13*1e-3)
        gaussz = gauss.axial_intensity(zvals)
        superz = superpose.axial_intensity_super(zvals,amp=1)
        
        xvals = np.arange(-0.5,0.5,0.005)*1e-3
        
        gaussx = gauss.one_dim_intensity(xvals,amp=1)
        superx = superpose.one_dim_intensity_super(xvals,amp=1)
        
        superx_prime = superx - (1/np.exp(1)**2)
        superz_prime = superz - (1/np.exp(1)**2)
        
        delta_x_super = np.abs(np.where(superx == np.amax(superx))[0][0] - np.where(np.abs(superx_prime) == np.amin(np.abs(superx_prime)))[0][0])
        delta_z_super = np.abs(np.where(superz == np.amax(superz))[0][0] - np.where(np.abs(superz_prime) == np.amin(np.abs(superz_prime)))[0][0])
        
        super_vol = delta_x_super * delta_x_super * delta_z_super
        volume_ratio = round((gauss_vol / super_vol),2)
        
        z_ratio = round((delta_z / delta_z_super),2)
        
        print("Volume Calculated")
        
        plt.plot(zvals/1e-3,superz,color="Green",label="Superposition")
        print("Superposition Plotted")
        plt.plot(zvals/1e-3,gaussz,color="Blue",linestyle="--",label="TEM00")
        print("Gaussian Plotted")
        plt.plot((zvals[0]/1e-3,zvals[-1]/1e-3),((1/np.exp(1)**2),(1/np.exp(1)**2)),linestyle="--",color="k",label="1/e^2")
        print("Waist Plotted")
        plt.scatter(x=(zvals[np.where(np.abs(superz_prime) == np.amin(np.abs(superz_prime)))[0][0]]/1e-3),y=(np.amin(np.abs(superz_prime)) + (1/np.exp(1)**2)),marker="x",color="Red")
        plt.xlabel("Axial Distance from Waist / mm")
        plt.ylabel("Normalised Intensity")
        plt.legend(loc="best")
        plt.title("Axial Confinement Reduced " + str(z_ratio) + " Times")
        plt.savefig(faxial,dpi=300)
        plt.show()
        
        plt.plot(xvals/1e-3,superx,color="Green",label="Superposition")
        print("Superposition Plotted")
        plt.plot(xvals/1e-3,gaussx,color="Blue",linestyle="--",label="TEM00")
        print("Gaussian Plotted")
        plt.plot((xvals[0]/1e-3,xvals[-1]/1e-3),((1/np.exp(1)**2),(1/np.exp(1)**2)),linestyle="--",color="k",label="1/e^2")
        print("Waist Plotted")
        plt.scatter(x=(xvals[np.where(np.abs(superx_prime) == np.amin(np.abs(superx_prime)))[0][0]]/1e-3),y=(np.amin(np.abs(superx_prime)) + (1/np.exp(1)**2)),marker="x",color="Red")
        plt.xlabel("Radial Distance from Axis / mm")
        plt.ylabel("Normalised Intensity")
        plt.legend(loc="best")
        plt.title("Volume Reduced " + str(volume_ratio) + " Times")
        plt.savefig(foned,dpi=300)
        plt.show()
        
        print(p[i],p[j])
        
        if p[i] < p[j]:
            superpose = Superposition([(p[i],0),(p[j],0)],w0=0.13*1e-3,rgrad=1*1e-6)
            
            fname = "p" + str(p[i]) + "-" + str(p[j]) + "twodim.bmp"
            faxial = "p" + str(p[i]) + "-" + str(p[j]) + "axial.png"
            foned = "p" + str(p[i]) + "-" + str(p[j]) + "onedim.png"
            
            #superpose.save_2D_intensity_image(fname)
            
            zvals = np.arange(-250*1e-3,250*1e-3,0.005*1e-3)
            
            gauss = LagGauss(p=0,l=0,w0=0.13*1e-3)
            gaussz = gauss.axial_intensity(zvals)
            superz = superpose.axial_intensity_super(zvals,amp=1)
            
            
            
            xvals = np.arange(-0.5,0.5,0.005)*1e-3
            
            gaussx = gauss.one_dim_intensity(xvals,amp=1)
            superx = superpose.one_dim_intensity_super(xvals,amp=1)
            
            superx_prime = superx - (1/np.exp(1)**2)
            superz_prime = superz - (1/np.exp(1)**2)
            
            delta_x_super = np.abs(np.where(superx == np.amax(superx))[0][0] - np.where(np.abs(superx_prime) == np.amin(np.abs(superx_prime)))[0][0])
            delta_z_super = np.abs(np.where(superz == np.amax(superz))[0][0] - np.where(np.abs(superz_prime) == np.amin(np.abs(superz_prime)))[0][0])
            
            super_vol = delta_x_super * delta_x_super * delta_z_super
            volume_ratio = round((gauss_vol / super_vol),2)
            
            z_ratio = round((delta_z / delta_z_super),2)
            
            plt.plot(xvals/1e-3,superx,color="Green",label="Superposition")
            plt.plot(xvals/1e-3,gaussx,color="Blue",linestyle="--",label="TEM00")
            plt.plot((xvals[0]/1e-3,xvals[-1]/1e-3),((1/np.exp(1)**2),(1/np.exp(1)**2)),linestyle="--",color="k",label="1/e^2")
            plt.scatter(x=(xvals[np.where(np.abs(superx_prime) == np.amin(np.abs(superx_prime)))[0][0]]/1e-3),y=(np.amin(np.abs(superx_prime)) + (1/np.exp(1)**2)),marker="x",color="Red")
            plt.xlabel("Radial Distance from Axis / mm")
            plt.ylabel("Normalised Intensity")
            plt.legend(loc="best")
            plt.title("Volume Reduced " + str(volume_ratio) + " Times")
            plt.savefig(foned,dpi=300)
            plt.show()
            
            plt.plot(zvals/1e-3,superz,color="Green",label="Superposition")
            plt.plot(zvals/1e-3,gaussz,color="Blue",linestyle="--",label="TEM00")
            plt.plot((zvals[0]/1e-3,zvals[-1]/1e-3),((1/np.exp(1)**2),(1/np.exp(1)**2)),linestyle="--",color="k",label="1/e^2")
            plt.scatter(x=(zvals[np.where(np.abs(superz_prime) == np.amin(np.abs(superz_prime)))[0][0]]/1e-3),y=(np.amin(np.abs(superz_prime)) + (1/np.exp(1)**2)),marker="x",color="Red")
            plt.xlabel("Axial Distance from Waist / mm")
            plt.ylabel("Normalised Intensity")
            plt.legend(loc="best")
            plt.title("Axial Confinement Reduced " + str(z_ratio) + " Times")
            plt.savefig(faxial,dpi=300)
            plt.show()
            
            print(p[i],p[j])
            
for i in range(len(p)):
    for j in range(len(p)):       
        for k in range(len(p)):
            if p[i] < p[j]:
                if p[j] < p[k]:
                    superpose = Superposition([(p[i],0),(p[j],0),(p[k],0)],w0=0.13*1e-3,rgrad=1*1e-6)
                    
                    fname = "p" + str(p[i]) + "-" + str(p[j]) + "-" + str(p[k]) + "twodim.bmp"
                    faxial = "p" + str(p[i]) + "-" + str(p[j]) + "-" + str(p[k]) + "axial.png"
                    foned = "p" + str(p[i]) + "-" + str(p[j]) + "-" + str(p[k]) + "onedim.png"
                    
                    #superpose.save_2D_intensity_image(fname)
                    
                    zvals = np.arange(-250*1e-3,250*1e-3,0.005*1e-3)
                    
                    gauss = LagGauss(p=0,l=0,w0=0.13*1e-3)
                    gaussz = gauss.axial_intensity(zvals)
                    superz = superpose.axial_intensity_super(zvals,amp=1)
                    
                    xvals = np.arange(-0.5,0.5,0.005)*1e-3
                    
                    gaussx = gauss.one_dim_intensity(xvals,amp=1)
                    superx = superpose.one_dim_intensity_super(xvals,amp=1)
                    
                    superx_prime = superx - (1/np.exp(1)**2)
                    superz_prime = superz - (1/np.exp(1)**2)
                    
                    delta_x_super = np.abs(np.where(superx == np.amax(superx))[0][0] - np.where(np.abs(superx_prime) == np.amin(np.abs(superx_prime)))[0][0])
                    delta_z_super = np.abs(np.where(superz == np.amax(superz))[0][0] - np.where(np.abs(superz_prime) == np.amin(np.abs(superz_prime)))[0][0])
                    
                    super_vol = delta_x_super * delta_x_super * delta_z_super
                    volume_ratio = round((gauss_vol / super_vol),2)
                    
                    z_ratio = round((delta_z / delta_z_super),2)
                    
                    plt.plot(xvals/1e-3,superx,color="Green",label="Superposition")
                    plt.plot(xvals/1e-3,gaussx,color="Blue",linestyle="--",label="TEM00")
                    plt.plot((xvals[0]/1e-3,xvals[-1]/1e-3),((1/np.exp(1)**2),(1/np.exp(1)**2)),linestyle="--",color="k",label="1/e^2")
                    plt.scatter(x=(xvals[np.where(np.abs(superx_prime) == np.amin(np.abs(superx_prime)))[0][0]]/1e-3),y=(np.amin(np.abs(superx_prime)) + (1/np.exp(1)**2)),marker="x",color="Red")
                    plt.xlabel("Radial Distance from Axis / mm")
                    plt.ylabel("Normalised Intensity")
                    plt.legend(loc="best")
                    plt.title("Volume Reduced " + str(volume_ratio) + " Times")
                    plt.savefig(foned,dpi=300)
                    plt.show()
                    
                    plt.plot(zvals/1e-3,superz,color="Green",label="Superposition")
                    plt.plot(zvals/1e-3,gaussz,color="Blue",linestyle="--",label="TEM00")
                    plt.plot((zvals[0]/1e-3,zvals[-1]/1e-3),((1/np.exp(1)**2),(1/np.exp(1)**2)),linestyle="--",color="k",label="1/e^2")
                    plt.scatter(x=(zvals[np.where(np.abs(superz_prime) == np.amin(np.abs(superz_prime)))[0][0]]/1e-3),y=(np.amin(np.abs(superz_prime)) + (1/np.exp(1)**2)),marker="x",color="Red")
                    plt.xlabel("Axial Distance from Waist / mm")
                    plt.ylabel("Normalised Intensity")
                    plt.legend(loc="best")
                    plt.title("Axial Confinement Reduced " + str(z_ratio) + " Times")
                    plt.savefig(faxial,dpi=300)
                    plt.show()
                    
                    print(p[i],p[j],p[k])"""