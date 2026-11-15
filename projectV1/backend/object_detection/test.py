from ultralytics import YOLO
from PIL import Image

model = YOLO("/Users/abdulbalhas/Desktop/uni/sem7/DS Projekt/project/DSProjectLEGO/runs/detect/models/yolo26s_obj_det_1stRun-6/weights/best.pt")


results1 =model("/Users/abdulbalhas/Downloads/20261115_183744.jpg", save=True)
results2 =model("/Users/abdulbalhas/Downloads/20261115_183751.jpg", save=True)

a1 = results1[0].plot(
    line_width=1,
    font_size=4
)
a2 = results2[0].plot(
    line_width=4,
    font_size=10
)

Image.fromarray(a1[...,::-1]).save("bild1.jpg")
Image.fromarray(a2[...,::-1]).save("bild2.jpg")

