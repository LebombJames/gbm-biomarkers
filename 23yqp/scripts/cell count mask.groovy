import ij.process.ShortProcessor
import ij.process.ImageProcessor
import ij.ImagePlus
import ij.io.FileSaver

// Tile size in microns, will be used for tile creation and cell counting
print("RUNNING")
int tileSizeMicrons = (args && args.length > 1) ? args[1] : 100

createFullImageAnnotation(true)
runPlugin('qupath.lib.algorithms.TilerPlugin', """{"tileSizeMicrons":${tileSizeMicrons},"trimToROI":true,"makeAnnotations":true,"removeParentAnnotation":true}""")

selectAnnotations()

runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield":"Hematoxylin OD","requestedPixelSizeMicrons":0.0,"backgroundRadiusMicrons":8.0,"backgroundByReconstruction":true,"medianRadiusMicrons":0.0,"sigmaMicrons":1.5,"minAreaMicrons":10.0,"maxAreaMicrons":400.0,"threshold":0.1,"maxBackground":2.0,"watershedPostProcess":true,"cellExpansionMicrons":5.0,"includeNuclei":true,"smoothBoundaries":true,"makeMeasurements":true}')

def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// 2. Get the pixel calibration (physical size of 1 pixel)
def cal = server.getPixelCalibration()
double pixelWidth = cal.hasPixelSizeMicrons() ? cal.getPixelWidthMicrons() : 1.0

if (!cal.hasPixelSizeMicrons()) {
    print "WARNING: This image has no pixel calibration metadata. Assuming 1 pixel = 1 µm."
}

// 3. Convert your micrometer tile size into a pixel tile size
double tileSizePixels = tileSizeMicrons / pixelWidth

print "Tile size set to ${tileSizeMicrons} µm (${tileSizePixels} pixels)."

// Calculate the dimensions of our new output image (1 pixel per tile)
int width = server.getWidth()
int height = server.getHeight()
int cols = Math.ceil(width / tileSizePixels) as int
int rows = Math.ceil(height / tileSizePixels) as int

ShortProcessor processor = new ShortProcessor(cols, rows)

// First try to find all tiles. If there are none, assume the user made annotation tiles
def objects = hierarchy.getTileObjects()
if (objects.size() == 0) objects = hierarchy.getAnnotationObjects()

int countedCells = 0
int maxPixel = 0

for (ann in objects) {
    int childCount = ann.nChildObjects()
    if (childCount == null || childCount == 0 || ann.classification != null) continue
    //print(childCount)

    def roi = ann.getROI()
    //print(roi)
    if (roi == null) continue

    // Get the X/Y coordinates of the cell (QuPath coordinates are always in pixels)
    double cx = roi.getCentroidX()
    double cy = roi.getCentroidY()

    // Calculate which tile column and row the cell falls into
    int col = (cx / tileSizePixels) as int
    int row = (cy / tileSizePixels) as int

    // Add +1 to the corresponding pixel in our output image
    if (col >= 0 && col < cols && row >= 0 && row < rows) {
        processor.set(col, row, childCount)
        countedCells += childCount
    }
}

// 6. Convert to an ImagePlus object for saving
ImagePlus imp = new ImagePlus("Cell Counts Per ${tileSizeMicrons}um Tile", processor)

// 7. Define where to save the image (Saves into a folder called "export" in your project)
String outDir = buildFilePath(PROJECT_BASE_DIR, "export")
mkdirs(outDir)
String outPath = buildFilePath(outDir, "${server.getMetadata().getName()}_cellularity_${tileSizeMicrons}um.tif")

// Save as a 32-bit TIFF
FileSaver fs = new FileSaver(imp)
fs.saveAsTiff((args && args.length > 0) ? args[0] : outPath)

print "Processed ${countedCells} cells."
print "Successfully saved tile map to: " + outPath