import os
from predict import __predict
import io_utils as io
import rendering
import geometry_processing as gp

class Facemarker:

    def __init__(self,
                 mesh_path,
                 output_path="",
                 render_result=False,
                 ):


        self.__meshes = io.meshes_setup( io.import_mesh(mesh_path) ) 
        self.__projections = projections
        self.__render = render_result

        self.__results = None


        file = os.path.splitext(os.path.basename(mesh_path))[0]
        if not output_path:
            directory = output_path if output_path else os.path.dirname(mesh_path)
            out_path = os.path.join(directory, f"{file}_landmarks.json")

        else:
            out_path = output_path if output_path.endswith(".json") else os.path.join(output_path, f"{file}_landmarks.json")

        self.__output_path = out_path



    def predict(self, projections=100):
        rendering.ensure_display()

        self.__results.landmarks_3d, self.__results.closest_vertices_ids = __predict(self.__meshes, projections)


    def render_result(self):
        if not self.__results:
            print("Facemarks not yet calculated. First run self.predict().")
            return

        rendering.render_result(self.__meshes.actual_mesh, self.__facemarks.closest_vertices_ids)


    def save_facemarks_json():
        if not self.__facemarks:
            print("Facemarks not yet calculated. First run self.predict().")
            return

        io.save_facemarks_json(self.__mesh_path, self.__facemarks.landmarks_3d, self.__facemarks.closest_vertices_ids)
