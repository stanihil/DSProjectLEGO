# Usage


## Training data Labelling
manually labeled dataset using self made photos:
```\projectV1\dataset\generic```

already labeled dataset: 
https://www.kaggle.com/datasets/hbahruz/multiple-lego/data

not pre-labeled
https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,202309140842198941751-0

auto-labeling and bounding box creation script using YOLO
```\helper\auto_label_yolo.py```


## Object detection

detect object from image of pile and save individual bricks

## Classification

cassify bricks from individual images

## set recommendation
Execute the recommender:
```
/.venv/Scripts/python.exe /helper/set_recommender/set_recommender
`
```
The recommender only checks a limited number of the largest sets whose part
count does not exceed the available inventory. It then verifies exact
part/color quantities locally.
 

## UI
[streamlit.io](https://streamlit.io/)

