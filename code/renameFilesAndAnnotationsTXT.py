# This ***RENAMES/MOVES*** images and their corresponding YOLO annotation files accordingly

import os

targetImageDir =    'D:/path/to/images/'
targetTxtDir =      'D:/path/to/labels/'
outputDir =         'D:/path/to/somewhere/'

annotationFilesEnding = ".txt"
newImageName = "roughAnnotationsReprocess_"
newImageStartingNum = 1
newImageZeroPadding = 3


if os.path.exists(targetImageDir) == False:
    print ("The path:" + targetImageDir + " was not found!")
    exit()
    
if os.path.exists(targetTxtDir) == False:
    print ("The path:" + targetTxtDir + " was not found!")
    exit()
    
if os.path.exists(outputDir) == False:
    print ("The output path:" + targetDir + " was not found!")
    exit()

   

 
for filename in os.listdir(targetImageDir):

    f = os.path.join(targetImageDir, filename)   
    
    if os.path.isfile(f) and not filename.endswith(annotationFilesEnding):      
        fileEnding = os.path.splitext(filename)[1]  #blah.jpg  is now .jpg
        newFileName = newImageName + str(newImageStartingNum).zfill(newImageZeroPadding) + fileEnding
        newFilePath = outputDir + "/" + newFileName
      
        try:
            os.rename(f, newFilePath)
        except IOError:
            print ("RENAME FAILED:" + f)
            continue
        else:
            print ("Renamed -- " + filename + "    ----->    " + newFileName)                                
                      
            #handle the annnotation file separately, as it may not exist at all if the image is an annotated null
            ftxt = os.path.join(targetTxtDir, os.path.splitext(filename)[0] + annotationFilesEnding)
            
            if os.path.isfile(ftxt):
                newTxtFileName = newImageName + str(newImageStartingNum).zfill(newImageZeroPadding) + annotationFilesEnding
                newTxtFilePath = outputDir + "/" + newTxtFileName       
                try:
                    os.rename(ftxt, newTxtFilePath)
                except IOError:
                    print ("ANNOTATION FILE RENAME FAILED:" + ftxt)        

            newImageStartingNum += 1
                                


print ("\n\nFile names in " + outputDir + " have been replaced accordingly.")

