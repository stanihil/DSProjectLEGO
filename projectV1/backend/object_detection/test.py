from ultralytics import YOLO
from PIL import Image

model = YOLO("/Users/abdulbalhas/Desktop/uni/sem7/DS Projekt/project/DSProjectLEGO/runs/detect/yolo26s_obj_det_1stRun/weights/best.pt")


results1 =model("/Users/abdulbalhas/Downloads/20261117_141928.jpg")
results2 =model("/Users/abdulbalhas/Downloads/20261117_141847.jpg")

a1 = results1[0].plot(
    line_width=2,
    font_size=4
)
a2 = results2[0].plot(
    line_width=2,
    font_size=4
)

Image.fromarray(a1[...,::-1]).save("bild1.jpg")
Image.fromarray(a2[...,::-1]).save("bild2.jpg")

