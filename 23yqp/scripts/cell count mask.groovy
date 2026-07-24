import ij.process.ShortProcessor
import ij.ImagePlus
import ij.io.FileSaver
import groovy.transform.CompileStatic

@CompileStatic
void mymain(String[] args) {
    print("RUNNING")
    
    var imageData = getCurrentImageData()
    if (imageData == null) {
        print "ERROR: No image open."
        return
    }
    
    var hierarchy = imageData.getHierarchy()
    var server = imageData.getServer()
    
    var cal = server.getPixelCalibration()
    double pixelWidth = cal.hasPixelSizeMicrons() ? cal.getPixelWidthMicrons() : 1.0d
    
    if (!cal.hasPixelSizeMicrons()) {
        print "WARNING: This image has no pixel calibration metadata. Assuming 1 pixel = 1 µm."
    }
    
    setImageType('BRIGHTFIELD_H_E')
    
    // Calculate MRI tile size (176 tiles along shortest side)
    int width = server.getWidth()
    int height = server.getHeight()
    
    int shortestSide = Math.min(width, height)
    double shortestSideMicrons = shortestSide * pixelWidth
    int mriTileSize = Math.round(shortestSideMicrons / 176.0) as int
    
    // Use args if provided, otherwise use default
    int tileSizeMicrons = (args && args.length > 1) ? args[1].toInteger() : 150
    double tileSizePixels = tileSizeMicrons / pixelWidth
    
    print "Tile size set to ${tileSizeMicrons} µm (${tileSizePixels.round(2)} pixels)."
    
    // Create the annotation and run the tiler.
    removeAnnotations()
    createFullImageAnnotation(true)
//    runPlugin('qupath.lib.algorithms.TilerPlugin', """{"tileSizeMicrons":${tileSizeMicrons},"trimToROI":true,"makeAnnotations":true,"removeParentAnnotation":true}""")
    
    //selectAnnotations()
    
    runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield":"Hematoxylin OD","requestedPixelSizeMicrons":0.0,"backgroundRadiusMicrons":8.0,"backgroundByReconstruction":true,"medianRadiusMicrons":0.0,"sigmaMicrons":1.5,"minAreaMicrons":10.0,"maxAreaMicrons":400.0,"threshold":0.1,"maxBackground":2.0,"watershedPostProcess":true,"cellExpansionMicrons":5.0,"includeNuclei":true,"smoothBoundaries":true,"makeMeasurements":true}')
    
    int cols = Math.ceil(width / tileSizePixels) as int
    int rows = Math.ceil(height / tileSizePixels) as int
    var processor = new ShortProcessor(cols, rows)
    
    var cells = hierarchy.getDetectionObjects()
    int countedCells = 0
    
    for (cell in cells) {
        var roi = cell.getROI()
        if (roi == null) continue
    
        // Get the exact X/Y coordinates of the cell
        double cx = roi.getCentroidX()
        double cy = roi.getCentroidY()
    
        // Calculate which grid coordinate the cell falls into
        int col = (cx / tileSizePixels) as int
        int row = (cy / tileSizePixels) as int
    
        // Add +1 to that pixel
        if (col >= 0 && col < cols && row >= 0 && row < rows) {
            int currentVal = processor.get(col, row)
            processor.set(col, row, currentVal + 1)
            countedCells++
        }
    }
    
    var imp = new ImagePlus("Cell Counts Per ${tileSizeMicrons}um Tile", processor)
    
    var outDir = createDirectoryInProject("export", "cell_count_${tileSizeMicrons}")
    var outPath = buildFilePath(outDir, "${server.getMetadata().getName()}.tif")
    
    var fs = new FileSaver(imp)
    if (fs.saveAsTiff((args && args.length > 0) ? args[0] : outPath)) {
        print "Processed ${countedCells} cells."
        print "Successfully saved tile map to: " + outPath
    } else {
        print "ERROR: Failed to save to " + outPath
    }
}
String[] safeArgs = []
if (binding.hasVariable('args')) {
    safeArgs = (binding.getVariable('args') ?: []) as String[]
}
mymain(safeArgs)