from cx_Freeze import Executable, setup

build_exe_options = {
    "packages": [
        "cv2",
        "mediapipe",
        "pygame",
        "numpy",
        "keras",
        "tensorflow",
        "requests",
        "pickle",
        "json",
    ],
    "include_files": [
        ("dnn_gesture_model.h5", "dnn_gesture_model.h5"),
        ("dnn_scaler.pkl", "dnn_scaler.pkl"),
        ("dnn_label_encoder.pkl", "dnn_label_encoder.pkl"),
        ("fonts/wqy-zenhei.ttc", "fonts/wqy-zenhei.ttc"),
    ],
    "zip_exclude_packages": ["*"],
    "excludes": ["torch"],
}

executables = [
    Executable(
        script="game.py",
        target_name="cognitive_training_game",
        base="gui",
    )
]

setup(
    name="cognitive_training_game",
    version="1.0.0",
    description="Cognitive Training Game",
    options={"build_exe": build_exe_options},
    executables=executables,
)
