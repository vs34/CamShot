
from ultralytics import YOLO
import  os
import cv2
import time
from progress.bar import Bar

def annotation_coords(results):
    coords=[]
    
    for r in results:
        image_height, image_width = r.orig_shape
        boxes = r.boxes
        #print(boxes)
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            coords.append(str(str(int(box.cls[0]))+" "+str((x1+x2)/(2*image_width))+" "+str((y1+y2)/(2*image_height))+" "+str((x2-x1)/image_width)+" "+str((y2-y1)/image_height)))
            

    return coords


def write_annotation(videosfolderpath,folderpath,model,train_val_test="train"):

    starttime=time.time()

    os.mkdir(folderpath)

    type1=train_val_test


    labelsfoldername="labels1"
    
    imagespath=folderpath+"images"
    lablespath=folderpath+labelsfoldername
    
    os.mkdir(lablespath)
    os.mkdir(imagespath)

    os.mkdir(lablespath+"/"+type1)
    os.mkdir(imagespath+"/"+type1)

    videonames=os.listdir(videosfolderpath)

    for videoname in videonames:

        cap = cv2.VideoCapture(os.path.join(videosfolderpath,videoname))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Save frame as image
            imagename=f'{videoname}_frame_{frame_count:04d}.jpg'
            frame_filename = os.path.join(imagespath+"/"+type1, imagename)
            results = model(frame,device="cuda",classes=[0])
            infolist=annotation_coords(results)

            if(len(infolist)>0):
                with open(lablespath+"/"+type1+"/"+imagename[:-4]+".txt","w") as anofile:
                    for i in infolist:
                        anofile.write(i+"\n")
                
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

        # Release the video capture object
        cap.release()

    endtime=time.time()
    print(f"time taken: {endtime-starttime} seconds\nfor {len(videonames)}")

folderpath="datasets/newone/"

model = YOLO("yolov8x.pt")


write_annotation("C:/Users/dell/Downloads/testvids",folderpath,model)





