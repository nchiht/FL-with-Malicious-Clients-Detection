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

python main.py -d cic -n 15 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 10 -a gaussian_attack -nm 3 -nd 2 -mag 1 -rat 0.5 -e 10
python main.py -d cic -n 15 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 10 -a gaussian_attack -nm 3 -nd 2 -mag 5 -rat 0.8 -t -e 10

python main.py -d cic -n 30 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 10 -a gaussian_attack -nm 8 -nd 4 -mag 1 -rat 0.5 -e 10 -->191219(step=1)

python main.py -d cic -n 30 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 10 -a gaussian_attack -nm 8 -nd 4 -mag 1 -rat 0.5 -e 10 -->220616(step=2) lor

python main.py -d cic -n 30 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 15 -a gaussian_attack -nm 8 -nd 4 -mag 5 -rat 0.8 -t -e 10 -->003229(step=2) lor 

python main.py -d cic -n 30 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 10 -a gaussian_attack -nm 8 -nd 4 -mag 1 -rat 0.2 -e 10 -->030202(step=2)

python main.py -d cic -n 30 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 10 -a gaussian_attack -nm 4 -nd 8 -mag 1 -rat 0.2 -e 10 -->123848(step=2) ==> lay

python main.py -d cic -n 30 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 10 -a gaussian_attack -nm 4 -nd 8 -mag 1 -rat 0.5 -e 10 -->185839(step=2) lor

python main.py -d cic -n 30 -r 50 -wr 5 --window_size 5 -str fedavg -a no_attack --no-defense --num_client_to_keep 20 -nm 0 -nd 0 -mag 0 -rat 0 -e 10 -->122933(step=2) ==> lay

python main.py -d cifar10 -n 50 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 30 -a gaussian_attack -nm 12 -nd 8 -mag 1 -rat 0.5 -e 10 --> 143949(step=2) ===> lay

python main.py -d cifar10 -n 50 -r 50 -wr 5 --window_size 5 -str fedavg --no-defense --num_client_to_keep 30 -a no_attack -nm 0 -nd 0 -mag 0 -rat 0 -e 10 -->162831(step=2) ===> lay

python main.py -d cic -n 50 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 30 -a gaussian_attack -nm 12 -nd 8 -mag 1 -rat 0.2 -e 10 --> 115438(step=2) lor

python main.py -d cic -n 50 -r 50 -wr 5 --window_size 5 -str our --num_client_to_keep 30 -a gaussian_attack -nm 12 -nd 8 -mag 1 -rat 0.5 -e 10 -->142010(step=2)