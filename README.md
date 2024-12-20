# FL-with-Malicious-Clients-Detection
## Dependencies
- Python 3.10

## Instruction

- Add APK converter

`git submodule add https://github.com/Luis-P-Duarte/Projeto-IA-22-23-Malware-Android.git modules/apk2image`

- Install requirements in apk2image module

`pip install -r modules\apk2image\requirements.txt`

- Install main requirements 

`pip install -r requirements.txt`

- Riskware: 541 fails

study case 1:

`python main.py -wr 5 -nm 10 -nd 5 -r 50 -n 50`

python main.py -d mnist -r 2 -n 4 -nm 1 -nd 0 --num_client_to_keep 2