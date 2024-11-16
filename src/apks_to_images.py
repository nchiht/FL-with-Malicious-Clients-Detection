from modules.ApkImageConverter import *
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # Note: path params should be relatives path compared to the current location
    # of executor terminal

    # Riskware
    print(f"=== Starting convert Riskware")
    proceed_apk2image('F:\CICMaldroid\Riskware\Riskware', './data/apk_images/riskware')
    print("======================================================================")

    # # Benign
    # print(f"=== Starting convert Bengin")
    # proceed_apk2image('F:\CICMaldroid\Benign\Benign', './data/apk_images/benign')
    # print("======================================================================")

    # # Adware
    # print(f"=== Starting convert Adware")
    # proceed_apk2image('F:\CICMaldroid\Adware\Adware', './data/apk_images/adware')
    # print("======================================================================")

    # # SMS
    # print(f"=== Starting convert SMS")
    # proceed_apk2image('F:\CICMaldroid\SMS\SMS', './data/apk_images/sms')
    # print("======================================================================")

    # # Banking
    # print(f"=== Starting convert Banking")
    # proceed_apk2image('F:\CICMaldroid\Banking\Banking', './data/apk_images/banking')
    # print("======================================================================")