from modules.APK_to_image import *
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # Note: path params should be relatives path compared to the current location
    # of executor terminal
    # Benign
    print(f"=== Starting convert Bengin")
    proceed_apk2image('./data/Malware/APK_Benign', './data/apk_images/benign')
    print("======================================================================")

    # Riskware
    print(f"=== Starting convert Riskware")
    proceed_apk2image('./data/Malware/APK_Riskware', './data/apk_images/riskware')
    print("======================================================================")    