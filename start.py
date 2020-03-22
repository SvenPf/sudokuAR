# Run with Python 3.7

import sys
import os

import sudoku_ar.app as app

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/"

if(len(sys.argv) == 1):
    app.run("https://test:test123@192.168.178.70:8080/video")
elif(len(sys.argv) == 2):
    if(str(sys.argv[1]) == "-ph"):
        app.run(PROJECT_ROOT + "sudoku_ar/resources/photo.jpg")
    elif(str(sys.argv[1]) == "-train"):
        import sudoku_ar.classifier.number_classifier as classifier
        classifier.train()
    elif(str(sys.argv[1]) == "-gen"):
        import sudoku_ar.data.train_data_gen_script as generator
        generator.main()
