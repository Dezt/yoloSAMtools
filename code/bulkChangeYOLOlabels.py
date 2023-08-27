
#This will go through every yolo file in a directory, and replace every label class accoridngly

import os

targetDir = 'D:/path/to/annotations/'  #path to annotation files
YOLOending=".txt"


  
  
 # INPUT an int label or class
 # OUTPUT the replaced class
def changeLabelTo(inLabel:int) -> int:
    #THERE IS NO REASON FOR WHAT IS HERE, THIS IS THE PART TO EDIT TO YOUR PARTICULAR NEEDS
    if (inLabel == 10):
        return 1
    return 0
    
        
def checkLineLabels(rawLine) -> str:
    rawLine = rawLine.rstrip()

    if rawLine == "":
        return ""

    elements = rawLine.split()
    
    label:int = int(float(elements[0]))
    elements[0] = str(changeLabelTo(label))
    
    #reassemble the corrected annotation line   
    outString:str = ""
    first = True
    for e in elements:
        if first == True:
            outString += e
            first = False
        else:
            outString += " " + e
    
    return outString
        
        
#############################################  main code

if os.path.exists(targetDir) == False:
    print ("The annotations path:" + targetDir + " was not found!")
    exit()
    
 
print ("Processing directory " + targetDir + "...\n") 

for filename in os.listdir(targetDir):
    filePath = os.path.join(targetDir, filename)

    if os.path.isfile(filePath) and filename.endswith(YOLOending):
        # Read the original file and modify the lines
        with open(filePath, 'r') as original_file:
            lines = original_file.readlines()
            modified_lines = [checkLineLabels(line.rstrip()) + "\n" for line in lines]

        # Replace the file with the modified version
        with open(filePath, 'w') as modified_file:
            modified_file.writelines(modified_lines)  
            
print ("done.") 