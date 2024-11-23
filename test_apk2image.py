from modules.ApkImageConverter import *
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    benign = os.path.abspath('./data/demo/APK_Benign')
    benign_images_path= os.path.abspath('./data/apk_images/benign')+'\\'

    print(benign_images_path)
    # Main loop for converting APK to image
    shas_list = apklist(benign)
    checkimg(benign_images_path,shas_list)

    start_time = time.time()
    count=1
    fail=0
    for sha in shas_list:
        print('Analysing ',count, 'of ', len(shas_list),': ', sha, end=" -> ")

        try:
            #Androguard
            a,d,dx = AnalyzeAPK(os.path.join(benign, sha))

            img = apk2image(a,d,dx)

            #Saving to local directory
            cv2.imwrite(benign_images_path + sha + '.jpg', img)
            print("success")

        except:
            print("fail")
            fail+=1
            pass
        count+=1

    print("Duration of convertion: ",time.time()-start_time, " - failed conversions: ",fail)