## Capture OBD data 

This repo includes two scripts to capture automotive data as CSV for upload and another to run inference or forward the results to a dashboard for integration.


https://github.com/user-attachments/assets/0f7b480e-c590-4770-b50e-cac6be4cd58b



https://github.com/user-attachments/assets/153315f4-0d1e-4c8c-ab77-994b7c9b4038



The scenario and car used for this example is a simulated air leak for the unhealthy condition simulated by disconnedting the intake sensor or vacume line:

 <img width="1148" height="577" alt="image" src="https://github.com/user-attachments/assets/3b0bf210-347c-40ee-82a4-defdcc22fb03" />


* simulating the airleak by disrupting the intake airflow measurements, disconnect the MAF sensor or a vaccume line.

<img width="720" height="1073" alt="image" src="https://github.com/user-attachments/assets/561d4931-2e57-433a-b40d-0c9adabd3052" />



* Capture OBD-II data from an ELM327 (USB or Bluetooth/RFCOMM) into a CSV that Edge Impulse can ingest.

<img width="1130" height="597" alt="image" src="https://github.com/user-attachments/assets/2fe14c9a-766b-49e7-a0f4-25a9f24d91c1" />



* Replay a CSV through an Edge Impulse .eim model and print predictions (top-k per window).

    <img width="1130" height="597" alt="image" src="https://github.com/user-attachments/assets/044460a7-dc69-4c66-bd6f-14928c4d8b18" />


* Usinging the Python-OBD library for integration



## Install Steps

```
# 0) System packages (Raspberry Pi OS / Debian/Ubuntu)
sudo apt update
sudo apt install -y python3-venv python3-pip bluez rfcomm

# (optional) If edge_impulse_linux import complains about audio/camera deps
# sudo apt install -y portaudio19-dev    # enables pip install pyaudio later

# 1) Create a virtualenv (avoids PEP 668 / system pip issues)
python3 -m venv ~/.venvs/ei
source ~/.venvs/ei/bin/activate
pip install --upgrade pip wheel

# 2) Install Python deps (minimal)
pip install edge_impulse_linux numpy pyserial

# If your Edge Impulse package asks for OpenCV or PyAudio on import, add:
# pip install "opencv-python>=4.5.1.48,<5"
# pip install pyaudio

# 3) (Recommended) Serial permissions so you don’t need sudo
sudo usermod -a -G dialout $USER
# reboot or log out/in to apply
```


## Bluetooth Classic (if you’ll use an ELM327 over BT/RFCOMM):

```
# Pair & trust in bluetoothctl (PIN is often 1234 or 0000 on ELM clones)
bluetoothctl
# power on
# agent on
# scan on
# pair XX:XX:XX:XX:XX:XX
# trust XX:XX:XX:XX:XX:XX
# connect XX:XX:XX:XX:XX:XX
# quit

# Bind RFCOMM channel 1 to a TTY
sudo rfcomm bind 0 XX:XX:XX:XX:XX:XX 1
# You now have /dev/rfcomm0 for the capture script

```


### Capture from USB

```
# Find the device (easiest to use the by-id path)
ls -l /dev/serial/by-id

# Capture at 2 Hz for 2 minutes (AUTO protocol)
python3 collect_obd_data.py \
  --port /dev/ttyUSB0 --baud 38400 --proto 0 \
  --hz 2 --label healthy --outfile obd_healthy.csv --duration-s 120
```

### Capture from Bluetooth

```
 Pair / trust in bluetoothctl (PIN often 1234 or 0000), then:
sudo rfcomm bind 0 XX:XX:XX:XX:XX:XX 1

python3 collect_obd_data.py \
  --port /dev/rfcomm0 --baud 115200 --proto 0 \
  --hz 2 --label healthy --outfile obd_bt.csv --duration-s 120
```


Replay the captured CSV to run inference

```
python3 play_csv_to_eim.py \
  --model ./your_model.eim \
  --csv ./obd_healthy.csv \
  --axes "RPM [RPM],PEDAL INPUT [%],MAF [g/s],NOx [ppm]" \
  --hz 2 --window-ms 1000 --step-ms 1000 \
  --print-topk 2 --out-jsonl predictions.jsonl
```


Cron job to run on a pi for BMW used in the webinar

```
crontab -e
```
paste the following
```
# m h  dom mon dow   command
@reboot /home/pi/start_obd_logger.sh >> /home/pi/obd_logs/boot.log 2>&1
```

ctrl+x save


