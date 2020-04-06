# Run with Python 3.7

import sys
import sudoku_ar.app as app

if len(sys.argv) == 1:
    print("Options are --train, --gen, --path")
elif len(sys.argv) == 2:
    if str(sys.argv[1]) == "--train":
        import sudoku_ar.classifier.number_classifier as classifier
        classifier.train()
    elif str(sys.argv[1]) == "--gen":
        import sudoku_ar.data.train_data_gen_script as generator
        generator.main()
elif len(sys.argv) == 3:
    if str(sys.argv[1]) == "--path":
        # cv2.imread does not support pathlib paths
        app.run(str(sys.argv[2]))
