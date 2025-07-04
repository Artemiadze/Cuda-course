# Setup Instructions

1. Для начала выполните в терминале Ubuntu Linux  `sudo apt update && sudo apt upgrade -y && sudo apt autoremove` затем перейдите в [downloads](https://developer.nvidia.com/cuda-downloads)
2. Заполните следующие параметры, соответствующие устройству, на котором вы будете проходить этот курс: Операционная система
   - Архитектура
   - Дистрибутив
   - Версия
   - Тип установщика
3. вам нужно будет выполнить команду, очень похожую на приведенную ниже в разделе "запустить файл".

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
```

4. в конце концов, вы сможете запустить `nvcc --version` и получить информацию о компиляторе nvidia cuda (версия и т.д.).
   также запустите `nvidia-smi`, чтобы убедиться, что nvidia распознает вашу версию cuda и подключенный графический процессор

5. Если `nvcc` не работает, запустите `echo $SHELL`. Если там написано bin/bash, добавьте следующие строки в файл ~/.bashrc. Если там написано bin/zsh, добавьте в файл ~/.zshrc. 
```bash
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```
Выполните `source ~/.zshrc` или `source ~/.bashrc` после этого попробуйте `nvcc -V` снова

## Alternatively

- Запустите этот скрипт: `./cuda-installer.sh`

## For [WSL2](https://medium.com/@omkarpast/technical-documentation-for-clean-installation-of-ubuntu-cuda-cudnn-and-pytorch-on-wsl2-9b265a4b8821)
