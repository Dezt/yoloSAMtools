#these methods optimize YOLO txt polygon annotations and replaces them after validating and optimizing them, and removes invalid annotations (Bounding box annotations are returned as is).
#First, optimizePolygonLineAnnotation attempts to remove stairstepping effects from the mask to polygon conversion.
#Then, optimizePolygonLineAnnotation uses the Visvalingam-Wyatt "area preserving" line optimization algorithm (as opposed to distance preserving like Douglas-Peucker algorithm). See:
#https://martinfleischmann.net/line-simplification-algorithms/
#https://en.wikipedia.org/wiki/Visvalingam%E2%80%93Whyatt_algorithm
#https://github.com/fitnr/visvalingamwyatt

#NOTE disadvantages of Visvalingam-Wyatt algorithm:
#   - The algorithm does not differentiate between sharp spikes and shallow features, meaning that it will clean up sharp spikes that may be important.
#   - The algorithm simplifies the entire length of the curve evenly, meaning that curves with high and low detail areas will likely have their fine details eroded.

import visvalingamwyatt as vw  #requires pip install visvalingamwyatt 
from commonUtils import *


############################################# methods

def getAreaLossThreshold(annotationPixelArea:float, areaThreshold25:float, areaThreshold200:float) -> float:
    return remap(25*25, 200*200, areaThreshold25, areaThreshold200, annotationPixelArea)
    
    
def areCoordsWithinDistance(coordA:[float,float], coordB:[float,float], distance) -> bool:
    if abs(coordA[0] - coordB[0]) > distance:
        return False
    if abs(coordA[1] - coordB[1]) > distance:
        return False
    return True
    
#remove stair steps  _/¯
#INPUT: all coords for one polygon in pixel space
#OUTPUT: remaining coords for the polygon in pixel space but with stairstepped coords removed 
#Note that the algorithm biases DOWN and RIGHT...  for some reason the SAM algorithm for converting the pixel map to a polygon mask seems to bias up left, so this should slightly counter that.
def removeStairStepping(coords:[[float,float]], stairStepCheckDistance:float, alignmentThreshold:float) -> [[float,float]]:  
    numCoords = len(coords)
    
    #this removes the most obvious stairstepping
    i = 1
    while i < numCoords: 
        if areCoordsWithinDistance(coords[i-1], coords[i], stairStepCheckDistance): 
            if abs(coords[i-1][0] - coords[i][0]) < alignmentThreshold: #are these HORIZONTALLY aligned within the threshold? If so, consider them HORIZONTALLY aligned
                #we have a HORIZONTAL stairstep... BIAS RIGHT (delete leftmost vert)
                if coords[i-1][0] < coords[i][0]:
                    coords.pop(i-1)
                else:
                    coords.pop(i)
                numCoords -= 1
                #auto increments, since we deleted an element
                
            elif abs(coords[i-1][1] - coords[i][1]) < alignmentThreshold: #are these VERTICALLY aligned within the threshold? If so, consider them VERTICALLY aligned
                #we have a VERTICAL stairstep... BIAS DOWN (delete upper vert)
                if coords[i-1][1] < coords[i][1]:
                    coords.pop(i-1)
                else:
                    coords.pop(i)
                numCoords -= 1
                #auto increments, since we deleted an element
            else:
                #coords are not stairstepped
                i += 1
        else:
            i += 1
            
    #what remains now is a stairstepping that consists of slightly less obvious single pixel stairsteps 
    #that cross diagonally from one corner of a pixel to another
    #let's remove those here... 
    numCoords = len(coords)
    i = 1
    while i < numCoords: 
        if areCoordsWithinDistance(coords[i-1], coords[i], 1.1):  #these coords are within a pixel... get rid of the higher one.
            if coords[i-1][1] < coords[i][1]: #coord i is lower than i-1          
                coords.pop(i-1) #BIAS DOWN
            else:
                coords.pop(i) #BIAS DOWN
            numCoords -= 1
            #auto increments, since we deleted an element
        else:
            i += 1
            
    return coords
            

#this method handles a single line that defines one YOLO polygon annotation.
#It optimizes the annotation by removing repetitive or overly detailed vertices, and alleviates stairstepping
#it also validates the annotation and will return an empty string if the annotation is flat/invalid/zero area
def optimizePolygonLineAnnotation(line:str, resX:int, resY:int, stairStepCheckDistance:float = 4.9, alignmentThreshold:float = .05, areaLossThreshold25:float = 2.4, areaLossThreshold200:float = 23.9, verbose:bool = False) -> str:    
    elements = line.split()
    
    if not elements:
        return "";
    
    if len(elements) < 7: #(at least 3 coords + annotation type)
        if len(elements) == 5:
            return line; #it's a bounding box most likely - return it unchanged and move on.
        else:
            return ""; #delete this annotation completely, as there are too few coords??  
   
    #make sure there is both an x and y for each coord
    elements = validateYOLOPolygonArray(elements)
        
    if verbose == True:
        print("Found " + str(int(float(len(elements)))) + " verts. Optimizing... ", end="" )

    #prepare the coords for analysis
    #convert them into pixel based coords from normalized coords as we store them to allow uniform optimization and handling of polygons
    coords = []
    i = 1 #lets start at 1 because the first value denotes the annotation type and is not a coord
    while i < len(elements):  
        coords.append((float(elements[i])* float(resX), float(elements[i+1])* float(resY))) #use pixel based (not normalized) coords so that the algorithm can optimize without bias in a non-square image
        i += 2
    
    if not coords or len(coords) < 3:
        return ""; #delete this annotation completely, as there are too few coords??
     
    #preprocess by removing stairstepping
    coords = removeStairStepping(coords, stairStepCheckDistance, alignmentThreshold)

    if not coords or len(coords) < 3:
        return ""; #delete this annotation completely, as there are too few coords??
     
    #prepare to check for complete flatness
    bIsTotallyFlat = True 
    firstX:float = coords[0][0]
    firstY:float = coords[0][1]
    
    #keep track of the bounding box of the annotation to determine area later
    minX:float = firstX
    maxX:float = firstX
    minY:float = firstY
    maxY:float = firstY
     
    i = 0
    while i < len(coords):     
        
        #gather bounding box
        minX = min(minX, coords[i][0])
        maxX = max(maxX, coords[i][0])
        minY = min(minY, coords[i][1])
        maxY = max(maxY, coords[i][1])
    
        #keep track if this annotation is totally flat...
        if bIsTotallyFlat:
            if coords[i][0] != firstX and coords[i][1] != firstY:  # The AND here is very important, it accounts for possible flatness in either axis. A non-flat annotation will have at least 1 coord with both xy different from the first              
                bIsTotallyFlat = False

        i +=1      
    #somehow this annotation is completely flat on at least one axis.  We need to remove it as it is corrupted/invalid
    if bIsTotallyFlat == True:
        print ("WARNING: removing completely flat (invalid) annotation")
        return "" #returning a blank line removes this annotation from the file

    #print(str(minX) + " " + str(maxX) + " " + str(minY) + " " + str(maxY)) #print the bbox
    
    #what is the pixel area of the annotation?
    #area = width * height
    pixelArea:float = abs(maxX-minX)*abs(maxY-minY)
    #get the threshold we should use for the simplification
    #this is based on two hand tweaked values based on the pixel area of the annotation
    #we will use a value that is lerped from the two hand tweaked ones.
    lossThreshold = getAreaLossThreshold(pixelArea, areaLossThreshold25, areaLossThreshold200)

    #the MEAT.  Simplify using Visvalingam-Wyatt algorithm
    simplifier = vw.Simplifier(coords)   
    newcoords = []  
    newCoords = simplifier.simplify(threshold=lossThreshold)
    #newCoords = simplifier.simplify(ratio=0.5) # Simplify by percentage of points to keep   
    #newCoords = simplifier.simplify(number=1000) # Simplify by giving number of points to keep   
    #newCoords = simplifier.simplify(threshold=0.01) # Simplify by giving an area threshold (in the units of the data)

    if not newCoords.any() or len(newCoords) < 3:
        return ""; #delete this annotation completely, as there are too few coords??
 
    #reassemble the corrected annotation line and restore 0-1 normalized coords from pixel coords.
    outString : string = elements[0]
    for c in newCoords:
        outString += " " + str(c[0]/float(resX)) + " " + str(c[1]/float(resY)) 
        
    if verbose == True:
        print("now " + str(int(float(len(newCoords)))) + " verts.")
    
    return outString
    