import numpy as np
import h5py
from skimage import io, filters
import glob
from scipy import ndimage as ndi

if __name__ == "__main__":
    filepath = "/data/jim/alex/VAC/UCH.48h.REF.plateA.n1_AM/" # Filepath to plate images
    cellImages = []

    for img in glob.glob(filepath+"*Red -*"):
        imBW = io.imread(img)
        thresh = filters.threshold_otsu(imBW)
        mask = imBW <= thresh
        labeled = measure.label(mask, background=1)
        cells = ndi.find_objects(labeled)

        for i in np.arange(len(cells))[::-1]:
            # Only get large cells
            if cells[i][1].stop - cells[i][1].start < 20 or cells[i][0].stop - cells[i][0].start < 20:
                del cells[i]
            # Only get square(ish) cells
            elif (cells[i][0].stop - cells[i][0].start) <= 0.8*(cells[i][1].stop - cells[i][1].start) \
                    or (cells[i][1].stop - cells[i][1].start) <= 0.8*(cells[i][0].stop - cells[i][0].start):
                del cells[i]

        for i in np.arange(len(cells)):
            # Append cells to master list
            cellImages.append(imBW[cells[i]])

    h5f = h5py.File("./data/cellImages.h5", "w")
    h5f.create_dataset("cellImages", data=cellImages)
    h5f.close()
