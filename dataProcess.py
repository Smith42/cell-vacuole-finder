import numpy as np
import h5py
from skimage import io, filters, measure
import glob
from scipy import ndimage as ndi

def resizeArray(arr):
    """
    Interpolate array to fit (100,100).
    """

    outArr = np.zeros((100,100))

    # Resize the arr
    calmRatio = 34.0/np.amax(calmTmp.shape)
    stressRatio = 34.0/np.amax(stressTmp.shape)

    calm3d = scipy.ndimage.interpolation.zoom(calmTmp, (calmRatio))
    stress3d = scipy.ndimage.interpolation.zoom(stressTmp, (stressRatio))
    outArr[:arr.shape[0],:arr.shape[1]] = arr
    return normalise(outArr)

def normalise(inData):
    """
    Normalise array.
    """
    inDataAbs = np.fabs(inData)
    inDataMax = np.amax(inData)
    normalisedData = inDataAbs/inDataMax
    return normalisedData

if __name__ == "__main__":
    filepath = "/data/jim/alex/VAC/UCH.48h.REF.plateA.n1_AM/" # Filepath to plate images
    cellImages = np.zeros[1,100,100]

    for img in glob.glob(filepath+"*Red -*"):
        print(img)
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

        print("Cells found:", len(cells))
        if len(cells) != 0:
            for i in np.arange(len(cells)):
                # Append cells to master list
                cellImages = np.append(cellImages, resizeArray(imBW[cells[i]]), axis=0)

    h5f = h5py.File("./data/cellImages.h5", "w")
    h5f.create_dataset("cellImages", data=cellImages[1:])
    h5f.close()
