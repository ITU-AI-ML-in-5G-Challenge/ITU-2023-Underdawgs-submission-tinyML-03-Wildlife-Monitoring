from keras.utils import to_categorical


def ohe_dict_func(unique_classes):

    ohe_dict = {}

    # Iterate through unique classes and one-hot encode each
    for i, class_name in enumerate(unique_classes):
        ohe = to_categorical(i, num_classes=len(unique_classes), dtype=float)
        ohe_dict[class_name] = list(ohe)

    return ohe_dict

# def ohe_dict_func(unique_classes):

#     ohe_dict = {}

#     # Iterate through unique classes and one-hot encode each
#     for i, class_name in enumerate(unique_classes):
#         # ohe = to_categorical(i, num_classes=len(unique_classes), dtype=float)
#         ohe_dict[class_name] = i

#     return ohe_dict