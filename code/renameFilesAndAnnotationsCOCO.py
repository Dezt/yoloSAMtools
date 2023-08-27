# This will rename all files in a directory while also replacing the names (text) of those files in the given text document
# This is useful if you want to keep COCO annotations attached to files, but want to rename them.

import os

targetDir = 'D:/path/to/imageDir'
annotationsTxtFile = targetDir + "/_annotations.coco.json"

newImageName = "NewFileName_"
newImageStartingNum = 1
newImageZeroPadding = 3


if os.path.exists(targetDir) == False:
    print ("The path:" + targetDir + " was not found!")
    exit()
    
   

annotationsFile = open(annotationsTxtFile,"r")
if annotationsFile == IOError:
     print ("Could not open annotations file for r/w" + annotationsTxtFile)
     exit()
     
#contents of the annotations are stored
textContents =  annotationsFile.read()

 
for filename in os.listdir(targetDir):

    f = os.path.join(targetDir, filename)
    
    if os.path.isfile(f):
    
        if os.path.samefile(f,annotationsTxtFile): #don't rename the annotations file itself :P
            continue 
    
        fileEnding = os.path.splitext(filename)[1]  #blah.jpg  is now .jpg
        newFileName = newImageName + str(newImageStartingNum).zfill(newImageZeroPadding) + fileEnding
        newFilePath = targetDir + "/" + newFileName
        
        try:
            os.rename(f, newFilePath)
        except IOError:
            print ("RENAME FAILED:" + f)
           
        else:
            print ("Renamed -- " + filename + "    ----->    " + newFileName)
                    
            textContents = textContents.replace(filename, newFileName)   #replace the name inside the annotations file
             
            newImageStartingNum += 1
            
#save changes to annotations      
with open(annotationsTxtFile, 'w') as file:
    file.write(textContents)
    file.close()

print ("\n\nFile names in " + annotationsTxtFile + " have been replaced accordingly.")

