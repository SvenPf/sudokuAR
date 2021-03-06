#!/usr/bin/env python3
# Run with Python 3 or higher

import sys
import os

APP_NAME = "sudoku_ar"

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
        os.system(sys.executable + " " + APP_NAME + " " + sys.argv[2])
