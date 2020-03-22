# Run with Python 3.7

import sys
from pathlib import Path
import sudoku_ar.app as app


PROJECT_ROOT = Path(__file__).parent.absolute()

if(len(sys.argv) == 1):
    app.run("https://test:test123@192.168.178.70:8080/video")
elif(len(sys.argv) == 2):
    if(str(sys.argv[1]) == "-ph"):
        # cv2.imread does not support pathlib paths
        app.run(str(PROJECT_ROOT / "sudoku_ar/resources/photo.jpg"))
    elif(str(sys.argv[1]) == "-train"):
        import sudoku_ar.classifier.number_classifier as classifier
        classifier.train()
    elif(str(sys.argv[1]) == "-gen"):
        import sudoku_ar.data.train_data_gen_script as generator
        generator.main()
