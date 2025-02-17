def select_features(model_type):
    # TODO: Here you have to select the features you want to use for your model.
    #  If you change your data set, you have to change this function accordingly.
    match model_type:
        case "CVM":
            features = [
                "xCenter",
                "yCenter",
                "xVelocity",
                "yVelocity"
            ]
        case "CAM":
            features = [
                "xCenter",
                "yCenter",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
            ]
        case "BCM":
            features = [
                "xCenter",  # x-coordinate of the vehicle's position
                "yCenter",  # y-coordinate of the vehicle's position
                "heading",  # Direction the vehicle is facing
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "lonVelocity",
                "latVelocity",
                "lonAcceleration",
                "latAcceleration"
            ]
        case "MLP":
            features = [
                "trackId",
                "xCenter",
                "yCenter",
                "heading",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "lonVelocity",
                "latVelocity",
                "lonAcceleration",
                "latAcceleration",
            ]
        case "LSTM":
            features = [
                "trackId",
                "xCenter",
                "yCenter",
                "heading",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "lonVelocity",
                "latVelocity",
                "lonAcceleration",
                "latAcceleration"
            ]
        case "GRU":
            features = [
                "trackId",
                "xCenter",
                "yCenter",
                "heading",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "lonVelocity",
                "latVelocity",
                "lonAcceleration",
                "latAcceleration"
            ]

    meta_features = ["recordingId", "trackId", "initialFrame", "finalFrame", "numFrames", "width", "length",
                     "class"]
    number_of_features = len(features)
    number_of_meta_features = len(meta_features)

    return features, number_of_features, meta_features, number_of_meta_features
