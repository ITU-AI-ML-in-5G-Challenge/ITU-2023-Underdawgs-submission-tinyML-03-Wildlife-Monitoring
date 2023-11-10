import pandas as pd

dx = 640/240 # Width
dy= 360/240 # Height

def get_dfs():
    train= pd.read_csv("train.csv")
    test= pd.read_csv("test.csv")

    train["xmin"]=train["xmin"]/dx
    train["ymin"]=train["ymin"]/dy
    train["xmax"]=train["xmax"]/dx
    train["ymax"]=train["ymax"]/dy
    train["xmin"]=train["xmin"]/240
    train["ymin"]=train["ymin"]/240
    train["xmax"]=train["xmax"]/240
    train["ymax"]=train["ymax"]/240

    test["xmin"]=test["xmin"]/dx
    test["ymin"]=test["ymin"]/dy
    test["xmax"]=test["xmax"]/dx
    test["ymax"]=test["ymax"]/dy
    test["xmin"]=test["xmin"]/240
    test["ymin"]=test["ymin"]/240
    test["xmax"]=test["xmax"]/240
    test["ymax"]=test["ymax"]/240
    return train, test


# def dummy():
    # train["xmin"]=train["xmin"]/train['width']
    # train["ymin"]=train["ymin"]/train['height']
    # train["xmax"]=train["xmax"]/train['width']
    # train["ymax"]=train["ymax"]/train['height']

    # test["xmin"]=test["xmin"]/test['width']
    # test["ymin"]=test["ymin"]/test['height']
    # test["xmax"]=test["xmax"]/test['width']
    # test["ymax"]=test["ymax"]/test['height']