import numpy as np
import open3d as o3d
import mediapipe as mp
from mediapipe.tasks.python import vision
from pathlib import Path

def mpImage(img):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)


def detectorInit(detection_confidence=.5):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions

    # Get the model path relative to the package
    model_path = Path(__file__).parent.parent / "face_landmarker_v2.task"

    options = FaceLandmarkerOptions(
        base_options=BaseOptions( model_asset_path= str(model_path) ),
            min_face_detection_confidence = detection_confidence,
            running_mode = vision.RunningMode.IMAGE,

            output_face_blendshapes = False,
            output_facial_transformation_matrixes = False,
        )

    return vision.FaceLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    from mediapipe.framework.formats import landmark_pb2
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

    # FACE MESH
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )

    # CONTOURS
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # EYES
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image
