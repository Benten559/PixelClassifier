from os.path import isdir, isfile, splitext, join
from os import listdir
from typing import List, Union
import cv2 as cv
import numpy as np
import pickle
import sklearn
import copy


class ImageContainer():
    """
    Provides a container interface for hyperspectral images not contained as a hypercube/envi load.
    The images may be of tif/png/jpeg/etc
    The intended use case of this object is to classify a single pixel of every wavelength iteratively
    """

    def __init__(self, img_dir: str, band_count: int, model_path:str, img_ext: str = "tif") -> None:
        self.band_count = band_count
        self.img_ext = f".{img_ext}" if img_ext[0] != "." else img_ext
        self.img_directory = self.validate_dir_source(img_dir)
        self.band_shape = (0, 0)
        self.band_stack = self.load_imagery()
        self.model = self.load_model(model_path)
        self.output_prediction = self.create_classification_mask()

    def validate_dir_source(self, img_dir: str) -> str:
        """
        Assures that path being used for the source imagery is valid and contains atleast the 
        """
        assert isdir(img_dir), f"{img_dir} is not a valid directory"
        return img_dir

    def load_imagery(self) -> Union[None, List[List[float]]]:
        """
        loads the wavelength data from the source img directory.
        Once image loading I/O is complete, a list of pixel-per-band stack is created.
        Each stack of bands corresponds to a pixel in column-major order (Top to bottom, left to right)  
        """

        try:
            band_images = [cv.imread(join(self.img_directory, img), cv.IMREAD_ANYDEPTH) for img in listdir(
                self.img_directory) if splitext(img)[1] == self.img_ext]
            
            self.single_band = copy.deepcopy(band_images[0])
            
            if self.band_count == len(band_images):
                
                self.band_shape = band_images[0].shape
                total_pixels = band_images[0].size
                
                flattened_band_images = [
                    img.ravel() for img in band_images if img.shape == self.band_shape]

                band_stack = [np.array([stack[pixel] for stack in flattened_band_images])
                              for pixel in range(total_pixels)]
                
                return band_stack

        except Exception as exc:
            raise RuntimeError(
                f"no images loaded from {self.img_directory}") from exc

    def load_model(self,model_path:str) :#-> Union[sklearn.svm.SVC,None]:
        """
        Initialize the pixel-classification model

        """
        if isfile(model_path):
            model = pickle.load(open(model_path, 'rb'))
            return model
        
    def create_classification_mask(self) -> Union[np.ndarray[int,float],None]:
        """
        Given a band_stack has been created successfully, each stack will be evaluated by the model.
        Each stack corresponds to a column-major pixel with the original image size. 
        This perfo
        Args:
            model_path (str): _description_rms predictions iteratively to then be reshaped into a 2d logical mask 
        """
        assert self.model != None, "Model path was not found"
        predicted_pixels = []
        for stack in self.band_stack:
            try:
                if np.isnan(stack[0]):
                    predicted_pixels.append(np.nan) 
                else:
                    if len(stack) == self.band_count:
                        classification = self.model.predict(stack.reshape(1,-1))[0]
                        predicted_pixels.append(classification)
            except Exception as e:
                print(f"the error {e}")
        # predicted_pixels = [np.nan if np.isnan(stack[0]) else self.model.predict(stack) for stack in self.band_stack]
        if len(predicted_pixels) > 0:
            return np.array(predicted_pixels).reshape(self.band_shape)
