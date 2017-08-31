from __future__ import print_function
import numpy as np
import pickle
from skimage import io, filters, measure, segmentation, exposure
from skimage.filters import rank
from skimage.morphology import watershed, disk, reconstruction, remove_small_objects
from sklearn.cluster import KMeans
from sklearn.externals import joblib
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
    cellImages = np.zeros([1,200,200])
    t0 = time.time()

    redimgs = sorted(glob.glob(filepath+"*Red -*"))
    uvimgs = sorted(glob.glob(filepath+"*UV -*"))
    for i in np.arange(len(redimgs)):
        t1 = time.time()
        print("Markers from", uvimgs[i], "Cells from", redimgs[i])

        u = io.imread(uvimgs[i])
        thresh = filters.threshold_li(u)
        mask = u <= thresh
        labeled = measure.label(mask, background=1)
        markers = rank.median(labeled, disk(25))

        r = io.imread(redimgs[i])
        p0, p1 = np.percentile(r, (10, 70)) # These parameters can be changed to affect the sensitivity of measurement
        rRescaled = exposure.rescale_intensity(r, in_range=(p0, p1))
        thresh = filters.threshold_li(rRescaled)
        mask = rRescaled <= thresh
        gradient = rank.gradient(mask==0, disk(2))

        labeled = segmentation.watershed(gradient, markers)
        labeled = segmentation.clear_border(labeled) # Get rid of border cells

        cells = filter(None, ndi.find_objects(labeled)) # Get rid of all that "None" cruft

        print("Cells found:", len(cells))
        if len(cells) != 0:
            for j in np.arange(len(cells)):
                # Append cells to master list
                cellIm = resizeArray(r[cells[j]])
                cellImages = np.append(cellImages, cellIm[np.newaxis,...], axis=0)
        t2 = time.time()
        print(t2-t1, "seconds")

    print("Running k-means cluster to filter out noise...")
    model = KMeans(n_clusters=2, n_init=50).fit(np.reshape(cellImages,[-1,200*200]))
    joblib.dump(model, "./logs/model.pkl")
    predictions = model.predict(np.reshape(cellImages,[-1,200*200]))
    unique, counts = np.unique(predictions, return_counts=True)
    print(dict(zip(unique, counts)))
    blueCells = np.argmax(counts) # This is where the cells are likely to be

    log = open("./logs/blueCell.txt","w+")
    strOut = str(str(dict(zip(unique, counts)))+"\nNon-noise cells are likely: "+str(blueCells))
    log.write(strOut)
    log.close()
    print(t2-t0, "s total time")
