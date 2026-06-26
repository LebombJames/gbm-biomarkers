import ij.process.FloatProcessor
import ij.ImagePlus
import ij.io.FileSaver

// Tile size in microns, will be used for tile creation and cell counting
int tileSizeMicrons = (args && args.length > 1) ? args[1] : 100

createFullImageAnnotation(true)
runPlugin('qupath.lib.algorithms.TilerPlugin', """{"tileSizeMicrons":${tileSizeMicrons},"trimToROI":true,"makeAnnotations":true,"removeParentAnnotation":true}""")

selectAnnotations()

addPixelClassifierMeasurements("necrosis", "necrosis")

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

// 4. Create a 32-bit Float image where pixel values will hold our cell counts
FloatProcessor fp = new FloatProcessor(cols, rows)

// 5. Iterate through all detections and map them to the grid
def objects = hierarchy.getTileObjects()
if (objects.size() == 0) objects = hierarchy.getAnnotationObjects()

for (ann in objects) {
    def children = ann.getChildObjects()

    def necrosisPct = ann.getMeasurements().get("necrosis: Necrosis %")
    print(necrosisPct)
    if (!necrosisPct) continue

    def roi = ann.getROI()
    if (roi == null) continue

    // Get the X/Y coordinates of the cell (QuPath coordinates are always in pixels)
    double cx = roi.getCentroidX()
    double cy = roi.getCentroidY()

    // Calculate which tile column and row the cell falls into
    int col = (cx / tileSizePixels) as int
    int row = (cy / tileSizePixels) as int

    // Add +1 to the corresponding pixel in our output image
    if (col >= 0 && col < cols && row >= 0 && row < rows) {
        fp.setf(col, row, necrosisPct as float)
    }


}

// 6. Convert to an ImagePlus object for saving
ImagePlus imp = new ImagePlus("Necrosis Per ${tileSizeMicrons}um Tile", fp)

// 7. Define where to save the image (Saves into a folder called "export" in your project)
String outDir = buildFilePath(PROJECT_BASE_DIR, "export")
mkdirs(outDir)
String outPath = buildFilePath(outDir, "${server.getMetadata().getName()}_necrosis_${tileSizeMicrons}um.tif")

// Save as a 32-bit TIFF
FileSaver fs = new FileSaver(imp)
fs.saveAsTiff((args && args.length > 0) ? args[0] : outPath)

print "Successfully saved tile map to: " + outPath