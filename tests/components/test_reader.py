import os
import unittest
from typing import Dict, List

from kartezio.core.components import Components, register
from kartezio.data.dataset import DatasetReader
from kartezio.readers import *
from kartezio.types import TypeArray, TypeScalar
from roifile import ImagejRoi
import shutil
import zipfile







class ReaderTestCase(unittest.TestCase):
    def assertDataItemEqual(self, item1, item2):
        """Custom assertion for comparing DataItem objects."""
        self.assertEqual(item1.shape, item2.shape, "Shapes do not match.")
        self.assertEqual(item1.count, item2.count, "Counts do not match.")
        self.assertEqual(item1.visual, item2.visual, "Visual attributes do not match.")

        # Compare datalist arrays
        self.assertEqual(len(item1.datalist), len(item2.datalist), "Datalists lengths do not match.")
        for array1, array2 in zip(item1.datalist, item2.datalist):
            np.testing.assert_array_equal(array1, array2, "Datalist arrays do not match.")

    def setUp(self) -> None:

        self.reader = DatasetReader('.',False,False)
        # Directory to temporarily store .roi files
        temp_dir = "tmp_test"
        os.makedirs(temp_dir, exist_ok=True)

        # Create the Square ROI
        square_roi = ImagejRoi.frompoints(np.array([[0,0],[0,10],[10,10],[10,0]]))
        # Save the ROI to a file
        square_roi.tofile("tmp_test/1_Z0.roi")
        square_roi.tofile("tmp_test/1_Z1.roi")

        square_roi = ImagejRoi.frompoints(np.array([[11, 11], [11, 20], [20, 20], [20, 11]]))
        # Save the ROI to a file
        square_roi.tofile("tmp_test/2_Z2.roi")
        zip_filename = "tmp_test/rois.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for file in os.listdir("tmp_test"):
                if ".roi" in file:
                    zipf.write(os.path.join("tmp_test",file),file)

    def tearDown(self):
        shutil.rmtree("path/to/folder")
        print("'tmp_test' folder removed")


    def test_RoiPolygon(self):
        raw_image = np.zeros((20,20)).astype(np.uint8)
        raw_image[:11,:11] = 1
        input_reader = RoiPolygonReader(self.reader)
        labels = input_reader.read("1_Z0.roi", (20, 20))
        gt = DataItem([raw_image], (20, 20), 1)
        self.assertDataItemEqual(labels,gt)


    def test_RoiPolyHedron(self):
        raw_image = np.zeros((5,20,20)).astype(np.uint8)
        raw_image[0,:11,:11] = 1
        raw_image[1, :11, :11] = 1
        raw_image[2,11:,11:] = 2
        input_reader = RoiPolyhedronReader(self.reader)
        path = "/home/eliott.gaudillat/Documents/CRCT2024/SourceData/lamp_detection/dataset/train/train_y/train2.zip" # "tmp_test/rois.zip"
        path = "tmp_test/rois.zip"
        labels = input_reader.read(path, (5,20, 20))
        print(labels)
        gt = DataItem([raw_image], (20, 20), 1)
        self.assertDataItemEqual(labels,gt)



if __name__ == "__main__":
    unittest.main()
