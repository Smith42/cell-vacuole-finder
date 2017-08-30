from __future__ import print_function
import numpy as np
import h5py
from skimage import io, filters, measure, segmentation
import glob
from scipy import ndimage as ndi
import time

def resizeArray(arr):
    """
    Interpolate array to fit (200,200).
    """

    outArr = np.zeros((200,200))

    # Resize the arr
    ratio = 200.0/np.amax(arr.shape)

    arr = ndi.interpolation.zoom(arr, (ratio))
    outArr[:arr.shape[0],:arr.shape[1]] = arr
    return normalise(outArr), ratio

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
    cellImages = np.zeros([1,200,200])
    cellRatios = np.zeros([1,1])
    t0 = time.time()
    ## Need to rewrite this... look at exampleImages

    redimgs = sorted(glob.glob(filepath+"*Red -*"))
    uvimgs = sorted(glob.glob(filepath+"*UV -*"))
    for i in len(redimgs):
        t1 = time.time()
        print("Markers from", uvimgs[i], "Cells from", redimgs[i])

        imUV = io.imread(uvimgs[i])
        thresh = filters.threshold_li(imUV)
        mask = imUV <= thresh
        labeled = measure.label(mask, background=1)
        markers = rank.median(labeled, disk(25))

        imBW = io.imread(redimgs[i])
        p0, p1 = np.percentile(imBW, (10, 70)) # These parameters can be changed to affect the sensitivity of measurement
        imBWrescaled = exposure.rescale_intensity(imBW, in_range=(p0, p1))
        thresh = filters.threshold_iso(imBWrescaled)
        mask = imBWrescaled <= thresh
        gradient = rank.gradient(mask==0, disk(2))

        labeled = segmentation.watershed(gradient, markers)
        labeled = segmentation.clear_border(labeled) # Get rid of border cells

        cells = filter(None, ndi.find_objects(labeled)) # Get rid of all that "None" cruft

        print("Cells found:", len(cells))
        if len(cells) != 0:
            for i in np.arange(len(cells)):
                # Append cells to master list
                cellIm, cellRat = resizeArray(imBW[cells[i]])
                cellImages = np.append(cellImages, cellIm[np.newaxis,...], axis=0)
                cellRatios = np.append(cellRatios, cellRat[np.newaxis,...], axis=0)
        t2 = time.time()
        print(t2-t1, "seconds")

    print("Running k-means cluster to filter out noise...")
    X_kmeans = k_means(np.reshape(cellImages,[-1,200*200]), 2, n_init=50) ## This looks like it works!!
    unique, counts = np.unique(X_kmeans[1], return_counts=True)
    print(dict(zip(unique, counts)))
    blueCells = np.argmax(counts) # This is where the cells are likely to be
    yellowCells = np.argmin(counts) # This is where the noise is likely to be
    blueMask = X_kmeans[1] == blueCells
    yellowMask = X_kmeans[1] == yellowCells
    cellImages = cellImages[1:] # Remove blank first cell
    cellImages = cellImages[blueMask] # Remove noise
    cellRatios = cellRatios[1:] # Remove blank first cell
    cellRatios = cellRatios[blueMask] # Remove noise

    print(t2-t0, "s total time")
    h5f = h5py.File("./data/cellImages.h5", "w")
    h5f.create_dataset("cellImages", data=cellImages)
    h5f.create_dataset("cellRatios", data=cellRatios)
    h5f.close()
