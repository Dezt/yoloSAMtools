"""
This tool is designed to help an immature model make better annotations
    INPUT: a folder with yolo polygon annotations or box annotations (any particular file can have mixed box/polygon annotations as well)
    INPUT: a folder with associated images
    
    OUTPUT: a folder that will contain modified (hopefully improved) yolo annotations reworked by SAM
            *if convertToBBoxOnly == True: OUTPUT will be YOLO bounding boxes instead of SAM polygons

HOW IT WORKS:
We want to use our still 'immature' model to help us annotate polygons.
Our 'immature' model may be fairly good at detecting relevant objects but may have mediocre results in accurately segmenting their edges.
To help with this, we can convert those mediocre annotations into bounding boxes, and then use the very mature SAM model to reselect the edges for us.
The tool uses the process:
    #1 convert all existing YOLO polygons into bounding boxes (existing bbox annotations will also used)
    #2 use the bounding boxes with the very mature SAM model to give us new masks
    #3 convert those masks BACK AGAIN into YOLO polygon annotations, hopefully improving them
    #4 (if postProcessSAM == True) optimize those polygons so they are not too noisy and can still be hand corrected in roboflow or labelbox etc
See: https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb  section:  Usecase 2: Predict segmentations masks using bounding boxes

IF YOU GET ERROR - "DefaultCPUAllocator: not enough memory:"
Out-of-memory exceptions can occur when applying SAM inferrence to the bounding boxes if the image size is too large and with many annotations.
If you get an out-of-memory issue, you may be able to solve it in one of 2 ways:
    1)Increase your memory page size.
        If you are on Windows you can increase your virtual memory (page file) size in 'advanced system settings' > advanced tab > Performance [settings] > advanced tab > Virtual memory [Change...]
    2)Split up the annotations into different groups/files and run SAM separately on them

"""
    
    
#annotationsDir = 'D:/WORKING/DEV/WallProjectorML/WORKING/imageSegmentation/_transfer/mixedRoughAnnotations.yolov8/test/labels/'  #path to annotation files
#imageDir = 'D:/WORKING/DEV/WallProjectorML/WORKING/imageSegmentation/_transfer/mixedRoughAnnotations.yolov8/test/images/'  #can be the same as the annotationsDir.  It needs the associated image so it can know the resolution to do pixel based calculations   
annotationsDir =    '../demo/'  #path to annotation files
imageDir =          '../demo/'  #can be the same as the annotationsDir.  It needs the associated image so it can know the resolution to do pixel based calculations
outputDir =         '../'

YOLOending=".txt"
imageEnding=".jpg"

convertToBBoxOnly = False #True == don't use SAM, and simply use the tool to convert all annotations to YOLO bbox annotations
postProcessSAM = True #converted SAM maps return very dense polygons that stairstep across the pixels, it may be desirable to postprocess/optimize the polygons to reduce complexity so it can be practical to still hand edit them in labelbox/roboflow/etc




#############################################  internal setup

sam_checkpoint = "D:/path/to/sam_vit_h_4b8939.pth"
model_type = "vit_h"

import time

import os
from PIL import Image 
from commonUtils import *
from optimizationUtils import *

if convertToBBoxOnly == False: #this stuff is only needed if we are using SAM
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor

    import numpy as np
    import torch
    import torchvision
    from transformers import SamModel, SamProcessor

import matplotlib.pyplot as plt
import cv2
      


#############################################  methods

# INPUT is a string containing a YOLO formatted polygon annotation (a single raw line from an annotation file)
# OUTPUT is a string bbox annotation... depending on convertToBBoxOnly it will return a [xmax ymax xmin ymin] or a YOLO format [xcenter ycenter w h]
# NOTE: THIS WILL ASSUME ANNOTATIONS WITH LENGTH 5 ARE ALREADY BOUNDING BOXES, AND WILL HANDLE ACCORDINGLY:  This allows mixed content to be handled.
# also note that bounding boxes with an axis smaller than minBboxPixel will be removed.  SAM can throw an  "empty max() arg" error if asked to infer a mask from a bbox smaller than 12 or 13 pixels
def convertYOLOPolygonAnnotation2Bbox(line: str, resX:int, resY:int) -> str:   
    #convert a yolo annotation to a bounding box 
    elements = line.split()
    
    if not elements:
        return ""
    
    bbox = YOLOPolygon2BBox(elements, resX, resY) #ann, xmin,ymin,xmax,ymax   - note that this converts bbox coords to pixel space (it also handles if this polygon is actually a bbox annotation)
    if bbox == []:
        return "" #something is wrong with the array     

    #validate the array for bounds, it should be fully within the frame.
    bbox[0] = max(bbox[0], 1) #xmin
    bbox[1] = max(bbox[1], 1) #ymin
    bbox[2] = min(bbox[2], resX -1) #xmax
    bbox[3] = min(bbox[3], resY -1) #ymax
    
    if convertToBBoxOnly == True: #interrupt intended flow of this tool, to export an intermediate format of a bbox - but as YOLO format
        #reassemble the corrected annotation line and restore 0-1 normalized coords from pixel coords... basically to return a YOLO BB
        outbbox =  BBox2YOLOBBox(bbox[0], bbox[1], bbox[2], bbox[3], resX, resY)
        outString = elements[0] + " " + str(outbbox[0]) + " " + str(outbbox[1]) + " " + str(outbbox[2]) + " " + str(outbbox[3])    
        return outString
    else:
        #return as pixel based
        return elements[0] + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3])  


    
# INPUT: a string containing SAM friendly bbox annotations:  annotationType xmax ymax xmin ymin \n etc
# OUTPUT: a string containing the same bboxes converted to YOLO formatted Polygon annotations:  annotationType x1 y1 x2 y2 x3 y3 ...
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# translation from SAM masks to YOLO code is adapted from here: https://github.com/akashAD98/YOLOV8_SAM/blob/main/detect_multi_object_SAM.py
def processYOLOAnnotationsWithSAM(lines, resX:int, resY:int, raw_image):   
      
    print ("Setting up bboxes... ", end = "")
    
    bounding_boxes = []
    bounding_boxes.append([])
    annotationTypes = []
    for line in lines:
        #print (line)
        elements = line.split() #these should be validated bboxes already, since they come from the method above
        if not elements or len(elements) != 5:
            continue #ignore corrupted elements... not sure how they would get here, but let's keep things civil.
        annotationTypes.append(elements[0])
        #note the float conversion before the int conversion... https://stackoverflow.com/questions/1841565/valueerror-invalid-literal-for-int-with-base-10
        bounding_boxes[0].append([int(float(elements[1])), int(float(elements[2])), int(float(elements[3])), int(float(elements[4]))])  
        
    print ("DONE")  
    
    print ("Initializing SAM model...") 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #huggingface version of the model with transformers library
    #model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device) #NOTE! this downloads a 2.6GB model
    #processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    
    #prepare SAM to read our particular image
    predictor.set_image(raw_image)
    print ("Image set to SAM predictor.")  
    
    #convert boxes to a tensor... not sure what this means https://medium.com/@tadewoswebkreator/converting-bounding-box-to-segmentation-dataset-part-2-fb77cb0c806e
    input_boxes = torch.tensor(bounding_boxes, device=predictor.device)
    print ("Tensor created.")    
    
    #   memory exception can occur here...
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, raw_image.shape[:2])
    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
        )

    print ("Masks created.")   

    outStrings = []
    for i, mask in enumerate(masks):

        binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)

        # Find the contours of the mask
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0: 
            #this annotation was probably too small for SAM to infer on it.
            print ("WARNING! AN ANNOTATION WAS LOST! SAM failed to infer it (it was likely too small/covered too few pixels).")
            continue

        largest_contour = max(contours, key=cv2.contourArea)

        # Get the new bounding box
        #bbox = [int(x) for x in cv2.boundingRect(largest_contour)]

        # Get the segmentation mask for object
        segmentation = largest_contour.flatten().tolist()
        mask = segmentation

        # convert mask to numpy array of shape (N,2)
        mask = np.array(mask).reshape(-1, 2)

        # normalize the pixel coordinates
        mask_norm = mask / np.array([resX, resY])

        # compute the bounding box
        #xmin, ymin = mask_norm.min(axis=0)
        #xmax, ymax = mask_norm.max(axis=0)
        #bbox_norm = np.array([xmin, ymin, xmax, ymax])

        # concatenate bbox and mask to obtain YOLO format
        # yolo = np.concatenate([bbox_norm, mask_norm.reshape(-1)])
        yolo = mask_norm.reshape(-1)
            
        outLine = annotationTypes[i]
        for val in yolo:
            outLine += " " + "{:.9f} ".format(val)

        outLine = validateYOLOPolygonArrayRawLine(outLine) #ensure good formatting   

        outStrings.append(outLine + "\n")
   
    print ("Prediction complete.")   
    return outStrings
    
    
def processYOLOdataFile(filePath, imagePath, outPath):
    try:
        # Read the image and gather resolution data
        try:
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except IOError:
            print ("Failed to open image: " + imagePath)
           
        else:
            h, w, _ = image.shape                 
            # Read the original file and modify the lines
            with open(filePath, 'r') as original_file:
                lines = original_file.readlines()
                modified_lines = [convertYOLOPolygonAnnotation2Bbox(line.rstrip(), w, h) + "\n" for line in lines]

            if convertToBBoxOnly == False:  
                modified_lines = processYOLOAnnotationsWithSAM(modified_lines, w, h, image) #its much more efficient to do all the lines at once here since we must load the image into SAM
                if postProcessSAM == True:
                    print("Optimizing polygon annotations for:  " + os.path.basename(filePath))
                    modified_lines = [optimizePolygonLineAnnotation(line.rstrip(), w, h) + "\n" for line in modified_lines]
            
            # Spit out the modified version
            with open(outPath, 'w') as modified_file:
                modified_file.writelines(modified_lines)
                 

        print("Success.")
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

#############################################  main code

if os.path.exists(annotationsDir) == False:
    print ("The annotations path:" + annotationsDir + " was not found!")
    exit()
    
if os.path.exists(imageDir) == False:
    print ("The image path:" + imageDir + " was not found!")
    exit()
    
if os.path.exists(outputDir) == False:
    print ("The output directory:" + outputDir + " was not found!")
    exit()
 
print ("Converting directory " + annotationsDir + "...\n") 
start_time = time.perf_counter()
 
for filename in os.listdir(annotationsDir):
    filePath = os.path.join(annotationsDir, filename)

    if os.path.isfile(filePath) and filename.endswith(YOLOending):
        imagePath = os.path.join(imageDir, os.path.splitext(filename)[0] + imageEnding)  #os.path.splitext(filename)[0] = "filename.txt" is now "filename"
        if os.path.isfile(imagePath):
            print("\n\nProcessing: " + filename)
            processYOLOdataFile(filePath, imagePath, outputDir + filename)           
        else:
            print("\nWARNING: corresponding image not found: " + imagePath)
    
#mark how much time the script took to run.
end_time = time.perf_counter()    
elapsed_time = end_time - start_time
m, s = divmod(elapsed_time, 60)
h, m = divmod(m, 60)
timePadding = 2

print("\n\nDONE.\nElapsed time: hms " + str(int(h)).zfill(timePadding) +":" + str(int(m)).zfill(timePadding) +":" + str(int(s)).zfill(timePadding))