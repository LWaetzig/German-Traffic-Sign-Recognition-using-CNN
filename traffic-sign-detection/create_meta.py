import os
import pandas as pd


def create_meta():
    meta = pd.DataFrame(columns=["path", "class", "shape", "color", "description"])
    # {
    #     "shape": {
    #         "circle": 0,
    #         "triangle": 1,
    #         "hexagon": 2,
    #         "diamond": 3,
    #         "inverse triangle": 4,
    #     },
    #     "color": {
    #         "white": 0,
    #         "blue": 1,
    #         "yellow": 2,
    #         "red": 3,
    #     },
    # }

    path = [
        os.path.join("data" , "Meta", file)
        for file in os.listdir(os.path.join("data", "Meta"))
        if file.endswith(".png")
    ]
    sign_meta = {
        "0.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (20 km/h)"
        },
        "1.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (30 km/h)"
        },
        "2.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (50 km/h)"
        },
        "3.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (60 km/h)"
        },
        "4.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (70 km/h)"
        },
        "5.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (80 km/h)",
        },
        "6.png": {
            "shape": 0,
            "color": 0,
            "description" : "End of Speed Limit (80 km/h)",
        },
        "7.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (100 km/h)",
        },
        "8.png": {
            "shape": 0,
            "color": 3,
            "description" : "Speed Limit (120 km/h)",
        },
        "9.png": {
            "shape": 0,
            "color": 3,
            "description" : "No passing",
        },
        "10.png": {
            "shape": 0,
            "color": 3,
            "description" : "No passing for vehicles over 3.5 metric tons",
        },
        "11.png": {
            "shape": 1,
            "color": 3,
            "description" : "Right-of-way at the next intersection",
        },
        "12.png": {
            "shape": 3,
            "color": 2,
            "description" : "Priority road",
        },
        "13.png": {
            "shape": 4,
            "color": 3,
            "description" : "Yield",
        },
        "14.png": {
            "shape": 2,
            "color": 3,
            "description" : "Stop",
        },
        "15.png": {
            "shape": 0,
            "color": 3,
            "description" : "No vehicles",
        },
        "16.png": {
            "shape": 0,
            "color": 3,
            "description" : "Vehicles over 3.5 metric tons prohibited",
        },
        "17.png": {
            "shape": 0,
            "color": 3,
            "description" : "No entry",
        },
        "18.png": {
            "shape": 1,
            "color": 3,
            "description" : "General caution",
        },
        "19.png": {
            "shape": 1,
            "color": 3,
            "description" : "Dangerous curve to the left",
        },
        "20.png": {
            "shape": 1,
            "color": 3,
            "description" : "Dangerous curve to the right",
        },
        "21.png": {
            "shape": 1,
            "color": 3,
            "description" : "Double curve",
        },
        "22.png": {
            "shape": 1,
            "color": 3,
            "description" : "Bumpy road",
        },
        "23.png": {
            "shape": 1,
            "color": 3,
            "description" : "Slippery road",
        },
        "24.png": {
            "shape": 1,
            "color": 3,
            "description" : "Road narrows on the right",
        },
        "25.png": {
            "shape": 1,
            "color": 3,
            "description" : "Road work",
        },
        "26.png": {
            "shape": 1,
            "color": 3,
            "description" : "Traffic signals",
        },
        "27.png": {
            "shape": 1,
            "color": 3,
            "description" : "Pedestrians",
        },
        "28.png": {
            "shape": 1,
            "color": 3,
            "description" : "Children crossing",
        },
        "29.png": {
            "shape": 1,
            "color": 3,
            "description" : "Bicycles crossing",
        },
        "30.png": {
            "shape": 1,
            "color": 3,
            "description" : "Beware of ice/snow",
        },
        "31.png": {
            "shape": 1,
            "color": 3,
            "description" : "Wild animals crossing",
        },
        "32.png": {
            "shape": 0,
            "color": 0,
            "description" : "End of all speed and passing limits",
        },
        "33.png": {
            "shape": 0,
            "color": 1,
            "description" : "Turn right ahead",
        },
        "34.png": {
            "shape": 0,
            "color": 1,
            "description" : "Turn left ahead",
        },
        "35.png": {
            "shape": 0,
            "color": 1,
            "description" : "Ahead only",
        },
        "36.png": {
            "shape": 0,
            "color": 1,
            "description" : "Go straight or right",
        },
        "37.png": {
            "shape": 0,
            "color": 1,
            "description" : "Go straight or left",
        },
        "38.png": {
            "shape": 0,
            "color": 1,
            "description" : "Keep right",
        },
        "39.png": {
            "shape": 0,
            "color": 1,
            "description" : "Keep left",
        },
        "40.png": {
            "shape": 0,
            "color": 1,
            "description" : "Roundabout mandatory",
        },
        "41.png": {
            "shape": 0,
            "color": 0,
            "description" : "End of no passing",
        },
        "42.png": {
            "shape": 0,
            "color": 0,
            "description" : "End of no passing by vehicles over 3.5 metric tons",
        },
    }

    meta["path"] = path
    meta["class"] = [i.split("/")[2].split(".")[0] for i in meta["path"]]
    for i , row in meta.iterrows():
        meta.loc[i, "shape"] = sign_meta[row["path"].split("/")[-1]]["shape"]
        meta.loc[i, "color"] = sign_meta[row["path"].split("/")[-1]]["color"]
        meta.loc[i, "description"] = sign_meta[row["path"].split("/")[-1]]["description"]

    meta.to_csv(os.path.join("data", "meta.csv"))

