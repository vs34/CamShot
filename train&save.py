from ultralytics import YOLO
import os

def main():
    model = YOLO('yolov9c.pt')
    model.train(data="coco8.yaml", epochs=100, imgsz=640)
def test():
    model = YOLO("yolov8n.pt")
    results = model("cctv_stock.mp4", show=True, imgsz=480, device='cuda')


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    #main()
    test()