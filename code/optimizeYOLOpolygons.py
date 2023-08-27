#these methods iterates through a dir of YOLO txt polygon annotations and replaces them after validating, optimizing, and removing invalid annotations (Bounding box annotations are returned as is).

import os
from PIL import Image 

from optimizationUtils import *

targetDir = 'D:/path/to/annotations/'  #path to annotation files
imageDir = 'D:/path/to/images/'
YOLOending=".txt"
imageEnding=".jpg"


# This is how many pixels of surface area an annotation is allowed to lose during polygon simplification
# For example, we want to optimize a polygon that encompases many pixels differently than a polygon over a low-res area
# The tool figures out what value to use by lerping from the low and high hand tweeked values
# simplifyAreaLossThreshold25 is for a 25x25 pixel annotation
# simplifyAreaLossThreshold200 is for a 200x200 pixel annotation
simplifyAreaLossThreshold25 :float = 2.5
simplifyAreaLossThreshold200 :float = 24

# coords outside this pixel distance will not be checked for stairstepping... and will be considered actually a straight or flat line
stairStepPixelThreshold: float = 4.9
# at what percentage of a pixel should two points be considered 'aligned' ?
alignedPixelThreshold: float = .05 

  
        
def fixYOLOdataFile(filePath, imagePath):
    try:
        # Read the image and gather resolution data
        try:
            img = Image.open(imagePath, "r") 
        except IOError:
            print ("Failed to open image: " + imagePath)
           
        else:
            w, h = img.size
              
            # Read the original file and modify the lines
            with open(filePath, 'r') as original_file:
                lines = original_file.readlines()
                modified_lines = [optimizePolygonLineAnnotation(line.rstrip(), w, h, stairStepPixelThreshold, alignedPixelThreshold, simplifyAreaLossThreshold25, simplifyAreaLossThreshold200) + "\n" for line in lines]

            # Replace the file with the modified version
            with open(filePath, 'w') as modified_file:
                modified_file.writelines(modified_lines)

        print("Successfully modified file.")
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
#############################################  main code

if os.path.exists(targetDir) == False:
    print ("The annotations path:" + targetDir + " was not found!")
    exit()
    
if os.path.exists(imageDir) == False:
    print ("The image path:" + imageDir + " was not found!")
    exit()
 
print ("Converting directory " + targetDir + "...\n") 

for filename in os.listdir(targetDir):
    filePath = os.path.join(targetDir, filename)

    if os.path.isfile(filePath) and filename.endswith(YOLOending):
        imagePath = os.path.join(imageDir, os.path.splitext(filename)[0] + imageEnding)  #os.path.splitext(filename)[0] = "filename.txt" is now "filename"
        if os.path.isfile(imagePath):
            print("Optimizing: " + filename)
            fixYOLOdataFile(filePath, imagePath)           
        else:
            print("WARNING: corresponding image not found: " + imagePath)
    
