from ultralytics import YOLO

model = YOLO("yolo26s.pt")

results = model.train(data = "/Users/abdulbalhas/Desktop/uni/sem7/DS Projekt/project/DSProjectLEGO/projectV1/dataset/object_det/data.yaml",
                      epochs=100,
                      name= "yolo26s_obj_det_1stRun",
                      device="mps")