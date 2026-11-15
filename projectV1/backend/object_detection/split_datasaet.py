from ultralytics.data.split import autosplit

autosplit(path="/Users/abdulbalhas/Desktop/uni/sem7/DS Projekt/project/DSProjectLEGO/projectV1/dataset/object_det/images",
          weights=(0.7, 0.2, 0.1),
          annotated_only=True)