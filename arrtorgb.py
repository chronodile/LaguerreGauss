#Code written by Mitch Walker
#2021
#Durham University Physics Dept

"""
This file of code is for converting 1D and 2D arrays into RGB bmps for use with the SLM. It has the following functions:
    
onedimrgb - This function takes a 1D array of length n, and converts it into a 16-bit RGB image of size n by n pixels. For best use with the SLM, the array should have length 512.
twodimrgb - This function takes a 2D n by m array, and converts it into a 16-bit RGB image (with zero bits in the blue channel, which is ignored by the SLM)
"""

def onedimrgb(array,filename,imshow,red):
    """
    This function takes a 1D array of length n, and converts it into an 16-bit RGB image of size n by n pixels. For best use with the SLM, the array should have length 512
    
    -array is the array to be converted; it should be a 1D numpy array
    -filename is the name you want the file to have, including the filetype (this can be any format supported fully by PIL, but recommend .bmp for the SLM; e.g. "image.bmp"). This should be of type string
    -red, green, and blue are values of 0 or 1; declare which colour channels should be present in the image. It should be noted that the SLM disregards the blue channel
    -imshow is Boolean, True to show the image, False to not
    -red is Boolean, True to include red bits, False to not
    
    returns: none, instead saves image in the same folder as the code under the given filename
    """
    
    #Import the necessary modules to run the function
    import numpy as np
    import PIL
    
    #Check that the array is normalised to have a maximum value of 1
    maxval = np.max(array)
    
    if maxval != 1:
        array = array/maxval
    
    #Rescale the matrix so each entry is a 16-bit integer from 0 to 65535
    norm = np.uint16(array*65535)
    
    #Create an array with opposite size to the original matrix
    vertical = np.ones((len(array),1))
    
    #Multiply the array with the vertical array to create a 2D n by n array (this should be 512x512 for the SLM, to generate a bmp of the correct size)
    twodim = vertical*norm
    
    #Determine the red channel (fine value) array
    if red == True:
        red = np.uint8(norm % 256)
    elif red == False:
        red = np.uint8(np.zeros(norm.shape))
    
    #Determine the green channel (coarse value) array
    green = np.uint8((twodim - red)/256)
    
    #Define the blue channel (all zeros) array
    blue = np.uint8(np.zeros(twodim.shape))
    
    #Stack the colour arrays to form the RGB array
    rgb = np.dstack((red,green,blue))
    
    #Generate an RGB image and save it under the given filename
    image = PIL.Image.fromarray(rgb,"RGB")
    if imshow == True:
        image.show()
    image.save(filename)
    
    #Print statement to indicate the function ran correctly
    print("Successfully saved " + filename)
    
def twodimrgb(array,filename,imshow,red):
    """
    This function takes a 2D n by m array, and converts it into a 16-bit RGB image (with zero bits in the blue channel, which is ignored by the SLM)
    
    -array is a 2D array of arbitrary phase values
    -filename is the name you want the file to have, including the filetype (this can be any format supported fully by PIL, but recommend .bmp for the SLM; e.g. "image.bmp"). This should be of type string
    -imshow is Boolean, True to show the image, False to not
    -red is Boolean, True to include red bits, False to not
    
    returns: none, instead saves image in the same folder as the code under the given filename
    """
    
    #Import the necessary modules to run the function
    import numpy as np
    import PIL
    
    #Check that the array is normalised to have a maximum value of 1
    maxval = np.max(array)
    
    if maxval != 1:
        array = array/maxval
        
    #Rescale the matrix so each entry is a 16-bit integer from 0 to 65535
    norm = np.uint16(array*65535)
    
    #Determine the red channel (fine value) array
    if red == True:
        red = np.uint8(norm % 256)
    elif red == False:
        red = np.uint8(np.zeros(norm.shape))
    
    #Determine the green channel (coarse value) array
    green = np.uint8((norm - red)/256)
    
    #Define the blue channel (all zeros) array
    blue = np.uint8(np.zeros(norm.shape))
    
    #Stack the colour arrays to form the RGB array
    rgb = np.dstack((red,green,blue))
    
    #Generate an RGB image and save it under the given filename
    image = PIL.Image.fromarray(rgb,"RGB")
    if imshow == True:
        image.show()
    image.save(filename)
    
    #Print statement to indicate the function ran correctly
    print("Successfully saved " + filename)