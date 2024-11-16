#Import libraries
from androguard.misc import AnalyzeAPK
from random import randrange
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import time
import sys
import cv2
import os


base_path = os.path.abspath('./src/utils/dictionaries')

#Importing dictionaries
with open(os.path.join(base_path,'API_calls_dict.json'), 'r') as file:
    API_calls_dict = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_apis_1.json'), 'r') as file:
    malicious_apis_1 = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_apis_2.json'), 'r') as file:
    malicious_apis_2 = json.load(file)
    file.close()

with open(os.path.join(base_path,'benign_api_1.json'), 'r') as file:
    benign_api_1 = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_apis_3.json'), 'r') as file:
    malicious_apis_3 = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_apis_4.json'), 'r') as file:
    malicious_apis_4 = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicous_activities.json'), 'r') as file:
    malicous_activities = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_services.json'), 'r') as file:
    malicious_services = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_recivers.json'), 'r') as file:
    malicious_recivers = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_intents.json'), 'r') as file:
    malicious_intents = json.load(file)
    file.close()

with open(os.path.join(base_path,'decleared_permissions.json'), 'r') as file:
    decleared_permissions = json.load(file)
    file.close()

with open(os.path.join(base_path,'malicious_permissions.json'), 'r') as file:
    malicious_permissions = json.load(file)
    file.close()

#Resizing all channels to a fixed shape by using Nearest neighbour interpolation
def OneD2TwoD_resize(image_pixels):

    h = int(len(image_pixels)**0.5)
    w = int(len(image_pixels)/h)

    unused_valus = len(image_pixels) - h*w
    inc = 0
    if(unused_valus != 0):
        unused_valus = len(image_pixels) - h*w
        empty_spots = w-unused_valus
        i = unused_valus
        j = unused_valus+empty_spots
        image_pixels += image_pixels[-j:-i]
        inc+=1

    image_pixels = np.array(image_pixels)
    twoD_image = image_pixels.reshape(h+inc,w)

    twoD_image = twoD_image.astype('float32')

    if(twoD_image.shape[0]>64):
        image_resized = cv2.resize(twoD_image, (64,64), interpolation = cv2.INTER_NEAREST)
    else:
        image_resized = cv2.resize(twoD_image, (64,64), interpolation = cv2.INTER_NEAREST)
    return image_resized

#Convert properties from classes.dex file to integers (pixel values) in range [0,255]
def dex2Image(dx,d):
    #Convert APIs to pixel values
    red_channel_stuff, API_calls_pixels = API_calls2pixels(dx)

    #Convert Opcodes to pixel values
    opcodes_pixls = list(opcodes2pixels(dx))

    blue_channel = OneD2TwoD_resize(API_calls_pixels + opcodes_pixls)

    #Convert Opcodes to pixel values
    opcodes_pixls = list(opcodes2pixels(dx))
    #blue_channel_stuff += opcodes_pixls


    #Extract malicious strings
    red_channel_stuff += strings2pixels(d)

    return blue_channel, red_channel_stuff

#Convert API calls to pixel values
def API_calls2pixels(dx):
    imprtnt_API_calls = []
    nrml_API_calls = []
    for api in dx.get_external_classes():
        api_tmp = api.get_vm_class()
        for i in api_tmp.get_methods():
            api_call = str(i).split('(')[0] # ignoring parameters and return type.
            if(api_call in API_calls_dict):
                imprtnt_API_calls.append(API_calls_dict[api_call])
            else:
                nrml_API_calls.append(sum(api_call.encode())%256)
    return imprtnt_API_calls, nrml_API_calls

#Convert opcodes to pixel values
def opcodes2pixels(dx):
    opcodes = []
    for method in dx.get_methods():
        if method.is_external():
            continue
        m = method.get_method()
        for ins in m.get_instructions():
          #if(ins.get_op_value not in opcodes):
          opcodes.append(ins.get_op_value())
    opcodes = list(set(opcodes))
    return opcodes

#Convert protected strings to pixel values
def strings2pixels(d):
    string_pixels = []
    all_strings = d[0].get_strings()

    for strng in all_strings:
        if(strng in API_calls_dict):
            string_pixels.append(API_calls_dict[strng])
        elif(strng in malicious_permissions):
            string_pixels.append(malicious_permissions[strng])
        elif(strng in malicous_activities):
            string_pixels.append(malicous_activities[strng])
        elif(strng in malicious_services):
            string_pixels.append(malicious_services[strng])
        elif(strng in malicious_recivers):
            string_pixels.append(malicious_recivers[strng])
        elif(strng in malicious_intents):
            string_pixels.append(malicious_intents[strng])
        else:
            pass
    return string_pixels

#Convert properties from AndroidMAnifest.xml file to pixel values
def manifest2Image(a):
    #Permissions
    permissions = a.get_permissions() + a.get_declared_permissions()
    imprtnt_permissions = []
    nrml_permissions = []
    for permission in permissions:
        perm = permission.split('.')[-1]
        # main key word (permission) is oftenly(always) comes at last.
        if(perm in malicious_permissions):
            imprtnt_permissions.append(malicious_permissions[perm])
        elif(perm in decleared_permissions):
            imprtnt_permissions.append(decleared_permissions[perm])
        else:
            nrml_permissions += list(perm.encode())
    nrml_permissions = sorted(nrml_permissions) # sort values
    imprtnt_permissions = sorted(imprtnt_permissions) # sort values

    #Activities
    activities = a.get_activities()
    imprtnt_activities = []
    nrml_activities = []
    for activity in activities:
        act = activity.split('.')[-1]
        if(act in malicous_activities):
            imprtnt_activities.append(malicous_activities[act])
        else:
            nrml_activities += list(act.encode())

    #Services
    services = a.get_services()
    imprtnt_services = []
    nrml_services = []
    for service in services:
        srvc = service.split('.')[-1]
        if(srvc in malicious_services):
            imprtnt_services.append(malicious_services[srvc])
        else:
            nrml_services += list(srvc.encode())

    #Recivers
    receivers = a.get_receivers()
    imprtnt_receivers = []
    nrml_receivers = []
    for receiver in receivers:
        recevr = receiver.split('.')[-1]
        if(recevr in malicious_recivers):
            imprtnt_receivers.append(recevr)
        else:
            nrml_receivers += list(recevr.encode())

    #Providers
    providers = a.get_providers()
    nrml_providers = []
    for provider in providers:
        nrml_providers += list(provider.encode())

    #Intents
    imprtnt_intents = []
    nrml_intents = []
    manifest_list = {'permissions':permissions,'activity' : activities, 'service': services, 'receiver':receivers, 'provider':providers}
    intents_itemtype = {'activity' : activities, 'service': services, 'receiver':receivers, 'provider':providers}
    for itemtype, listt in intents_itemtype.items():
        for item in listt:
            try:
                for intnts in a.get_intent_filters(itemtype, item).values():
                    for intnt in intnts:
                        if(intnt in malicious_intents):
                            imprtnt_intents.append(malicious_intents[intnt])
                        else:
                            nrml_intents += list(intnt.encode())
            except:
                pass
    red_channel_stuff = imprtnt_permissions + imprtnt_activities + imprtnt_services + imprtnt_intents + imprtnt_receivers
    green_channel = nrml_permissions + nrml_activities + nrml_services + nrml_receivers + nrml_providers + nrml_intents
    green_channel = OneD2TwoD_resize(green_channel)
    return green_channel, red_channel_stuff

#Collecting properties (pixel values) from files, place on channel, resize the channel and merge to make an image
def apk2image(a, d, dx):
    green_channel, red_channel_stuff1 = manifest2Image(a)

    blue_channel, red_channel_stuff2 = dex2Image(dx, d)

    red_channel = red_channel_stuff1 + red_channel_stuff2
    red_channel = OneD2TwoD_resize(red_channel)

    image = cv2.merge((blue_channel, green_channel, red_channel))
    image = image.astype(dtype='uint8')
    return image

def apklist(path):

    shas_list = []
    dirs = os.listdir(path)

    # This would print all the files and directories
    for file in dirs:
        shas_list.append(file.rstrip())

    return shas_list

def checkimg(path, shas):
    #Reading APK's SHA256 from downloaded APKs
    converted_imges = os.listdir(path)
    converted_shas = [sha[:-4] for sha in converted_imges]
    print(len(converted_imges), 'are converted.')

    shas_list = list(set(shas) - set(converted_shas))
    print(len(shas), 'APKs are remaining to be converted.')

def extract_channels(image):
    # Chia ảnh thành các kênh màu
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    return blue_channel, green_channel, red_channel

def show_channels(image):
    blue_channel, green_channel, red_channel = extract_channels(image)

    # Hiển thị ảnh blue
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(blue_channel, cmap='Blues')
    plt.title('Blue Channel')

    # Hiển thị ảnh green
    plt.subplot(132)
    plt.imshow(green_channel, cmap='Greens')
    plt.title('Green Channel')

    # Hiển thị ảnh red
    plt.subplot(133)
    plt.imshow(red_channel, cmap='Reds')
    plt.title('Red Channel')

    plt.show()

def proceed_apk2image(input_path, output_path):
    # benign = os.path.abspath('./data/demo/APK_Benign')
    # benign_images_path= os.path.abspath('./data/apk_images/benign') + '\\'
    apks_path = os.path.abspath(input_path)
    images_path = os.path.abspath(output_path) + '\\'

    # Main loop for converting APK to image
    shas_list = apklist(apks_path)
    checkimg(images_path,shas_list)

    start_time = time.time()
    count=1
    fail=0
    for sha in shas_list:
        print('Analysing ',count, 'of ', len(shas_list),': ', sha, end=" -> ")
        try:
            #Androguard
            a,d,dx = AnalyzeAPK(os.path.join(apks_path, sha))
            img = apk2image(a,d,dx)
            #Saving to local directory
            cv2.imwrite(images_path + sha.split('.')[0] + '.jpg', img)
            print("Success")

        except:
            print("Fail")
            fail+=1
            pass
        count+=1

    print("Duration of convertion: ",time.time()-start_time, "s - failed conversions: ",fail)