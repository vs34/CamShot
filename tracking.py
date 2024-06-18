import cv2
from ultralytics import YOLO

idcords = {}


def draw_text_with_background(img, text, pos, font, font_scale, font_thickness, text_color, bg_color):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    x, y = pos
    bg_x1, bg_y1 = x, y - text_h - 5
    bg_x2, bg_y2 = x + text_w, y + 5
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)
    

def coords(b):
    box = b.xyxy[0].cpu().numpy() 
    x1, y1, x2, y2 = map(int, box)

    ycoord = (y2 + y1) / 2
    xcoord = (x1 + x2) / 2
    return xcoord, ycoord

def show_cam_with_boxes(model, source, interval=5):

    location = {}
    frame_count = 0

    inside=0
    outside=0

    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, device="cuda")
            annotated_frame = results[0].plot()

            height,width, _ = annotated_frame.shape

            liney = int(height // 2) 
            cv2.line(annotated_frame, (0, liney), (width, liney), (0, 0, 255), 2)

            print(width,height,_)

            for b in results[0].boxes:
                id = int(b.id[0])
                x, y = coords(b)
                
                if id in location.keys():
                    prevx,prevy=location[id]
                    if (y<=liney and prevy>=liney):
                        inside+=1

                    if (y>=liney and prevy<=liney):
                        outside+=1
                
                location[id] = (x, y)
                
            text = f"people count: {len(results[0].boxes)} inside: {inside}, outside: {outside}"
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO("bestonetarushlvda.pt")

    vid = "C:/Users/dell/Downloads/cctv_stock.mp4"
    cam = "0"

    source = vid

    show_cam_with_boxes(model, source)
