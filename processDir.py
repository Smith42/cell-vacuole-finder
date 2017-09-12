from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import numpy as np
import h5py
from skimage import io, filters, measure, segmentation, exposure
from skimage.filters import rank
from skimage.morphology import watershed, disk, reconstruction, remove_small_objects
from scipy.misc import toimage
import glob
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import time, datetime

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

def cutArr(A):
    """
    Remove rows and columns of zero from input arr A.
    """
    A = A[:,~np.all(A == 0, axis=0)]
    A = A[~np.all(A == 0, axis=1)]
    return A

def getVacuholes(cell):
    """
    Get vacuoles using skimage.morphology.construction()
    """
    seed = np.copy(cell)
    seed[1:-1, 1:-1] = cell.max()
    mask = cell

    filled = reconstruction(seed, mask, method='erosion')

    thresh = filters.threshold_mean(filled-cell)
    mask = (filled-cell) >= thresh
    mask = remove_small_objects(mask, 100)
    labeled = measure.label(mask)

    vacuoles = ndi.find_objects(labeled)

    return np.array(filter(None,vacuoles))

if __name__ == "__main__":
    filepath = "/data/jim/alex/VAC/UCH.48h.REF.plateA.n1_AM/" # Filepath to plate images

    t0 = time.time()

    # Lower and upper boundaries of inliers, according to genIQR.ipynb:
    loBound = 8700
    hiBound = 305476

    # If the naming convention for slides changes this will also need to be changed.
    redimgs = sorted(glob.glob(filepath+"*Red -*"))
    greenimgs = sorted(glob.glob(filepath+"*Green -*"))
    blueimgs = sorted(glob.glob(filepath+"*Blue -*"))
    uvimgs = sorted(glob.glob(filepath+"*UV -*"))

    for i in np.arange(len(redimgs)):
        t1 = time.time()
        cellImagesRAW = []
        vacuoleArr = []
        print("Markers from", uvimgs[i], "Cells from", redimgs[i])

        r = io.imread(redimgs[i])
        g = io.imread(greenimgs[i])
        b = io.imread(blueimgs[i])
        u = io.imread(uvimgs[i])

        thresh = filters.threshold_li(u)
        mask = u <= thresh
        labeled = measure.label(mask, background=1)
        markers = rank.median(labeled, disk(25))

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
                cellImagesRAW.append(r[cells[j]])
            for cell in cellImagesRAW:
                ind = getVacuholes(cell)
                vacuoleArr.append(ind)

        dt = str(datetime.datetime.now().replace(second=0, microsecond=0).isoformat("_"))

        fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(15,15))
        cells = np.array(cells)

        arr = np.dstack([r, g, b])
        img = toimage(arr)
        imBright = img.point(lambda p: p * 3) # Make each pixel brighter
        plt.imshow(imBright)

        plt.title("Vacuole finder output "+dt+"\n"+redimgs[i])
        ax.axis("off")

        cellData = []
        vacData = []

        for j in np.arange(np.shape(cells)[0]):
            avgx = int(np.mean((cells[j,1].start,cells[j,1].stop)))
            xst = cells[j,1].start
            avgy = int(np.mean((cells[j,0].start,cells[j,0].stop)))
            yst = cells[j,0].start
            cellSize = abs((cells[j,0].stop-cells[j,0].start)*(cells[j,1].stop-cells[j,1].start))
            noVacs = len(vacuoleArr[j])
            cellData.append([j,int(avgx),int(avgy),int(cellSize),noVacs])

            for k in np.arange(len(vacuoleArr[j])):
                avgvx = int(np.mean((vacuoleArr[j][k,1].start,vacuoleArr[j][k,1].stop)))
                avgvy = int(np.mean((vacuoleArr[j][k,0].start,vacuoleArr[j][k,0].stop)))
                vacSize = abs((vacuoleArr[j][k,0].stop-vacuoleArr[j][k,0].start)*(vacuoleArr[j][k,1].stop-vacuoleArr[j][k,1].start))
                vacData.append([j,int(xst+avgvx),int(yst+avgvy),int(vacSize)])
                plt.scatter(xst+avgvx,yst+avgvy,s=10,marker='.',c="white")

            plt.annotate(str(j),xy=(avgx,avgy),color="white")

        # Outlier detection through a IQR threshold on cell size
        cellData = np.array(cellData)
        vacData = np.array(vacData)
        outlierMask = (cellData[:,3] < loBound) | (cellData[:,3] > hiBound)
        outlierData = cellData[outlierMask]
        vOutlierMask = np.zeros(vacData.shape[0], dtype=bool)
        # Detect and label outliers, and find vacuoles in cells that are outliers
        for i in np.arange(np.shape(outlierData)[0]):
            plt.annotate(str(outlierData[i,0]),xy=(outlierData[i,1],outlierData[i,2]),color="blue")
            vOutlierMask = vOutlierMask + np.array(vacData)[:,0] == outlierData[i,0]

        np.savetxt("./logs/"+dt+"_slide_"+str(i)+"_cellData.csv", cellData[~outlierMask], fmt="%i", delimiter=",", header=redimgs[i]+"\ncellNo,xcoord,ycoord,size,noVacuoles")
        np.savetxt("./logs/"+dt+"_slide_"+str(i)+"_vacuoleData.csv", vacData[~vOutlierMask], fmt="%i", delimiter=",", header=redimgs[i]+"\ncellNo,xcoord,ycoord,size")
        plt.savefig("./figures/output/"+dt+"_slide_"+str(i)+"_outputExampleRGB.png")

        t2 = time.time()
        print(t2-t1, "seconds")

    print(t2-t0, "s total time")
