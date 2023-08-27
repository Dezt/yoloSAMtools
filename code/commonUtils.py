#various methods that can be shared among the tools..


"""Linear interpolate on the scale given by a to b, using t as the point on that scale.
FROM: https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
Examples
--------
    50 == lerp(0, 100, 0.5)
    4.2 == lerp(1, 5, 0.8)
"""
def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

"""Inverse Linar Interpolation, get the fraction between a and b on which v resides.
Examples
--------
    0.5 == inv_lerp(0, 100, 50)
    0.8 == inv_lerp(1, 5, 4.2)
"""
def inv_lerp(a: float, b: float, v: float) -> float:
    return (v - a) / (b - a)
    
"""Remap values from one linear scale to another, a combination of lerp and inv_lerp.
i_min and i_max are the scale on which the original value resides,
o_min and o_max are the scale to which it should be mapped.
Examples
--------
    45 == remap(0, 100, 40, 50, 50)
    6.2 == remap(1, 5, 3, 7, 4.2)
"""
def remap(i_min: float, i_max: float, o_min: float, o_max: float, v: float) -> float:
    return lerp(o_min, o_max, inv_lerp(i_min, i_max, v))
   
"""   
accepts pixel UL + LR Bbox and converts it into a YOLO format with centerpoint + w/h
"""
def BBox2YOLOBBox(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

"""
converts YOLO Bbox into UL + LR Bbox
"""
def YOLOBBox2BBox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return [x1, y1, x2, y2]
    
"""   
#accepts an array of raw string elements from a YOLO polygon annotation including the annotation type element
#returns the same coords that have at a minimum, correct length
"""  
def validateYOLOPolygonArray(elements):
    #make sure there is both an x and y for each coord
    if len(elements)%2 != 1: #number of elements should be odd.  annotation type + even number of coords.  If it isn't, toss the last value, wtf.
        elements.pop()   
        
    return elements
"""   
#accepts a raw line from a YOLO polygon annotation including the annotation type element
#returns the same coords that have at a minimum a validated format and length
"""  
def validateYOLOPolygonArrayRawLine(rawLine):
    rawLine = rawLine.rstrip()

    if rawLine == "":
        return ""

    elements = rawLine.split()
 
    if not elements or len(elements) < 7: #(at least 3 coords + annotation type)
        return "" #delete this annotation completely, as there are too few coords??  
    
    #make sure there is both an x and y for each coord
    elements = validateYOLOPolygonArray(elements)

    #reassemble the validated line
    outString : string = "" #= elements[0]  #not needed, element 0 is already included
    first = True
    for e in elements:
        if first == True:
            outString += e
            first = False
        else:
            outString += " " + e

    return outString
    
    
"""   
#accepts an array of raw string elements from a YOLO polygon annotation in 0-1 NORMALIZED SPACE including the annotation type element,  and the image resolution
#returns bbox in PIXEL SPACE as 4 int values:  minX, minY, maxX, maxY
"""
def YOLOPolygon2BBox(elements, resX:int, resY:int):
      
    if not elements: 
        return [] #delete this annotation completely, wtf 
       
    if len(elements) == 5: #this annotation is VERY LIKELY a bbox already, convert it to pixel space and UL + LR bbox to use it as is.
        px = float(elements[1])* float(resX)
        py = float(elements[2])* float(resY)
        pw = float(elements[3])* float(resX)
        ph = float(elements[4])* float(resY)    
        return YOLOBBox2BBox(int(px), int(py), int(pw), int(ph)) 
        
    if len(elements) < 7: #(at least 3 coords + annotation type)
        return [] #delete this annotation completely, as there are too few coords??  
        
    elements = validateYOLOPolygonArray(elements)
 
    #prepare the coords for analysis
    #convert them into pixel based coords from normalized coords as we store them to allow uniform optimization and handling of polygons
    coords = []
    i = 1 #lets start at 1 because the first value denotes the annotation type and is not a coord
    while i < len(elements):  
        coords.append((float(elements[i])* float(resX), float(elements[i+1])* float(resY))) #use pixel based (not normalized) coords 
        i += 2
    
    if not coords or len(coords) < 3:
        return []; #delete this annotation completely, as there are too few coords??
      
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
        return [] #returning a blank line removes this annotation from the file

    #print(str(minX) + " " + str(maxX) + " " + str(minY) + " " + str(maxY)) #print the bbox
    return [int(minX), int(minY), int(maxX), int(maxY)]