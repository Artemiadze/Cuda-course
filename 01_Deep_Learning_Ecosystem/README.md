# Chapter 01: The Current Deep Learning Ecosystem


### **DISCLAIMER:** 

В этой части не рассматривается ничего технического в CUDA. Лучше показать вам экосистему вместо того, чтобы вслепую вдаваться в технические детали. Из моего опыта изучения этого материала, имея достаточное понимание экосистемы поможет вам распланировать всё должным образом, и это обеспечит вас мотивацией  к изучению на начальной стадии. 

По мере того, как мы будем углубляться в детали, я рекомендую вам исследовать и экспериментировать с тем, что вам покажется интересным (в этом разделе вы найдете интересные материалы). Если вы просто слушаете, как кто-то рассказывает о предмете в течение 20 часов, вы ограничиваете свои познания. Сложно ориентироваться в широте и глубине инфраструктуры глубокого обучения. Испытывать дискомфорт и ломать что-то - лучший способ учиться.


## Research
- PyTorch ([PyTorch - Fireship](https://www.youtube.com/watch?v=ORMx45xqWkA&t=0s&ab_channel=Fireship))
    - Если вы смотрите это, я предполагаю, что вы имеете хотя бы небольшой опыт работы с PyTorch. Если нет, я предлагаю видео по PyTorch от [Daniel Bourke](https://www.youtube.com/watch?v=Z_ikDlimN6A)
    - Pytorch поставляется с ночной и стабильной версиями ⇒ https://discuss.pytorch.org/t/pytorch-nightly-vs-stable/105633
        
        ночные выпуски, скорее всего, будут нестабильными, но в них будут представлены передовые обновления и оптимизация универсального фреймворка
        
    - Пользователи предпочитают PyTorch благодаря удобству использования
    - Вы найдёте претренированные модели в  torchvision (`pip install torchvision`) и `torch.hub`. Экосистема pytorch развивалась более децентрализовано но немного тяжелее чтобы ориентироваться в подходе к получению предварительно обученных моделей. Люди будут выпускать их модели на гитхаб-репозитории вместо их отправки в централизированную базу данных моделей. Huggingface чаще всего используется благодаря усилиям сообщества
    - Хорошая поддержка ONNX
- TensorFlow ([TensorFlow - Fireship](https://www.youtube.com/watch?v=i8NETqtGHms))
    - Отлично задокументирована и имеет обширное сообщество. Также является одним из наиболее используемый фреймворков глубинного обучения.
    - Comparatively the slowest DL framework
    - Создан компанией Google (designed for TPUs) и общего назначения ML (SVM, decision trees, etc).
    - Претренированный модели могут быть найдены здесь ⇒ https://www.tensorflow.org/resources/models-datasets
    - Хорошая поддержка для претренированных моделей.
    - Ограниченная поддержка ONNX (`tf2onnx`)
- Keras
    - Аналогично `torch.nn` для TensorFlow, но более высокого уровня.
    - Отдельная библиотека но глубоко интегрируется с TensorFlow, служит его основным высокоуровневым API
    - Полноценный фреймворк для создания и обучения модулей, а не только модулей нейронных сетей
- JAX ([JAX - Fireship](https://www.youtube.com/watch?v=_0D5lXDjNpw))
    - JIT-compiled Autograd Xccelerated Linear Algebra
    - Документация здесь ⇒ https://jax.readthedocs.io/en/latest/
    - Выглядит как numpy
    - Reddit сообщество  JAX ⇒ https://www.reddit.com/r/MachineLearning/comments/1b08qv6/d_is_it_worth_switching_to_jax_from/
    - JAX и Tensorflow развиваются компанией Google
    - Использует XLA (xccelerated linear algebra) компилятор
    - `tf2onnx` supported
- MLX
    - Развивается компанией Apple для Apple Silicon
    - Open-source framework
    - Фокусируется на высокопроизводительном машинном обучении на устройствах Apple
    - Спроектирован для тренировки и заключения
    - Оптимизирован для Apple's архитектуры Metal GPU
    - Поддержка динамических вычислительных графов
    - Подходит для исследований и разработки новых моделей ML
- PyTorch Lightning
    - https://www.reddit.com/r/deeplearning/comments/t31ppy/for_what_reason_do_you_or_dont_you_use_pytorch/
    - в основном это сокращение стандартного кода и распределенное масштабирование
    - `Trainer()` в отличие от тренировочного цикла


## Production
- Inference-only
    - vLLM
        - ‣
    - TensorRT
        - Интегрируется в pytorch для inference
        - Поддержка ONNX для загруженных моделей
        - высоко оптимизированные ядра cuda с учетом следующего
            - преимущества разреженности
            - квантование логического вывода
            - аппаратная архитектура
            - схемы доступа к памяти в видеопамяти и встроенной памяти на кристалле
        - сокращение от тензорного времени выполнения
        - разработан, сконструирован и поддерживается компанией Nvidia
        - создан специально для логического вывода LLM
        - использует некоторые из методов, которые мы рассматриваем в этом курсе, но абстрагируется от них для удобства использования
        - похоже, что TensorRT требует, чтобы Onnx изучил это
        - review https://nvidia.github.io/TensorRT-LLM/
        - переходите по ссылкам по порядку для изучения
            - https://nvidia.github.io/TensorRT-LLM/
            - https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
            - https://pytorch.org/TensorRT/getting_started/installation.html#installation
        - 
- Triton
    - Разработан и поддерживается компанией OpenAI ⇒ https://openai.com/index/triton/
    - CUDA-like, но на python и избавляет от путаницы при разработке ядра на обычном CUDA C/C++. Также соответствует рекордной производительности при умножении матриц
    - Get started ⇒ https://triton-lang.org/main/index.html
    - Напиши свой первое Triton-ядро ⇒ https://triton-lang.org/main/getting-started/tutorials/index.html
    - Triton Inference Server
        - https://developer.nvidia.com/triton-inference-server
    - https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf is the original Triton paper
    - triton-viz - это основной инструментарий Triton для профилирования и визуализации
    - Python позволяет точно контролировать то, что происходит на графическом процессоре, не беспокоясь о неожиданных хитросплетениях и сложностях C/C++.
        - Устраняет явное управление памятью `cudaMalloc`, `cudaMemcpy`, `cudaFree`
        - Нет необходимости в проверке ошибок / макросах `CUDA_CHECK_ERROR`
        - Уменьшена сложность при индексации на уровне сетки / блока / потока в параметрах запуска ядра
    
![](../05%20Writing%20your%20First%20Kernels/assets/triton1.png)
    
- torch.compile
    - Требует больше внимания чем TorchScript и, как правило, это более высокопроизводительней
    - Компилирует модель до статического представления, чтобы динамический графический компонент pytorch не беспокоился о том, что что-то изменится. Запускает модель как оптимизированный двоичный файл вместо стандартного готового pytorch
    - https://discuss.pytorch.org/t/the-difference-between-torch-jit-script-and-torch-compile/188167
- TorchScript
    - Может быть быстрым в сценариях, специально когда разворачивается на C++
    - Повышение производительности может зависеть от вашей архитектуры нейронной сети
    - https://discuss.pytorch.org/t/the-difference-between-torch-jit-script-and-torch-compile/188167
- ONNX Runtime
    - https://youtu.be/M4o4YRVba4o
    - “**ONNX Runtime training** можно ускорить время обучения модели на многоузловых графических процессорах NVIDIA для моделей transformer с помощью однострочного добавления к существующим сценариям обучения PyTorch”
    - Разработан и поддерживается компанией Microsoft
- Detectron2
    - Поддерживает обучения и тестирование модели
    - Проект по компьютерному зрению под началом Facebook (Meta)
    - Алгоритмы обнаружения и сегментации

## Low-Level
- CUDA
    - Compute unified device architecture (CUDA) можно рассматривать как язык программирования для графических процессоров nvidia.
    - CUDA libs ⇒ cuDNN, cuBLAS, cutlass (быстрая линейная алгебра и DL алгоритмы). cuFFT для быстрого свёртки (FFTs рассматриваются в курсе)
    - самостоятельная реализация ядер основана на аппаратная архитектура (Nvidia по-прежнему делает это скрытно, передавая специальные флаги компилятору)
- ROCm
    - CUDA эквивалентна AMD GPUs
- OpenCL
    - Открытый вычислительный язык (Open Computing Language) 
    - CPUs, GPUs, digital signal processors, other hardware
    - с тех пор как NVIDIA спроектирована CUDA, он будет превосходить OpenCL в задачах Nvidia. Если вы работаете с встраиваемыми системы (EE/CE), этому все еще стоит научиться.

## Inference for Edge Computing & Embedded Systems
    
- Периферийные вычисления относятся к локальным вычислениям с низкой задержкой и высокой эффективностью в реальных распределенных системах, таких как автопарки. Tesla FSD является ярким примером периферийных вычислений, поскольку в ней используется нейронная сеть, работающая локально в автомобиле. Он также должен отправлять данные обратно в Tesla, чтобы они могли улучшить свои модели.

- CoreML
- В первую очередь для развертывания предварительно обученных моделей на устройствах Apple
    - Оптимизирован для вывода на устройстве
    - Поддерживает обучение на устройстве
    - Поддерживает широкий спектр типов моделей (зрение, естественный язык, речь и т.д.)
    - Хорошо интегрируется с экосистемой Apple (iOS, macOS, watchOS, tvOS)
    - Обеспечивает конфиденциальность данных, сохраняя их на устройстве
    - Позволяет преобразовывать модели из других платформ
    - Разработан для того, чтобы разработчики приложений могли легко внедрять ML в свои приложения
- PyTorch Mobile
- TensorFlow Lite

## Easy to Use
- FastAI
    - Высокоуровневый API: Созданный на основе передовых технологий, Festival предоставляет более удобный интерфейс для решения распространенных задач глубокого обучения.
    - Быстрое прототипирование: Предназначен для быстрого внедрения современных моделей глубокого обучения.
    - Лучшие практики: Включает в себя много отличных практик и последние достижения в сфере глубинного обучения.
    - Менье кода: Typically requires less code to implement complex models compared to raw PyTorch.
    - Transfer learning: Excellent support for transfer learning out of the box.
- ONNX
    - Open Neural Network eXchange
    - `torch.onnx.export(model, dummy_input, "resnet18.onnx")`
    
    ```python
    import tensorflow as tf
    import tf2onnx
    import onnx
    
    # Load your TensorFlow model
    tf_model = tf.keras.models.load_model('path/to/your/model.h5')
    
    # Convert the model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
    
    # Save the ONNX model
    onnx.save(onnx_model, 'path/to/save/model.onnx')
    ```
    
    ![Untitled](assets/onnx.png)
    
- wandb
    - Сокращение от весов и погрешностей
    - Простая интеграция с проектами с помощью нескольких строк кода
    - Совместная работа в команде
    - Сравнение экспериментов с интуитивно понятным пользовательским интерфейсом
    
    ![Untitled](assets/wandb.png)
        
    
## Cloud Providers
- AWS
    - EC2 instances
    - Sagemaker (jupyter notebooks on a cluster, human data labelling/annotation, model training & deployment on AWS infrastructure)
- Google Cloud
    - Vertex AI
    - VM instances
- Microsoft Azure
    - Высокая скорость
- OpenAI
- VastAI
    - ссылка на изображение пользовательского интерфейса здесь
- Lambda Labs
    - Дешевые графические процессоры для центров обработки данных
## Compilers
- XLA
    - Специализированный компилятор для линейной алгебры, оптимизирующий вычисления с использованием тензорного потока
    - Обеспечивает низкоуровневую оптимизацию и генерацию кода для JAX
    - Выполняет оптимизацию всей программы, не ограничиваясь отдельными операциями, для оптимизации всего графика вычислений
    - Обеспечивает эффективное выполнение на различных аппаратных средствах (центральных и графических процессорах) за счет создания оптимизированного машинного кода
    - Реализованы расширенные возможности оптимизации, такие как operation fusion, которая объединяет множество операций в одном, более эффективном ядре
    - Позволяет JAX достигать высокой производительности без ручного написания специфичного для оборудования кода
- LLVM
- MLIR
- NVCC
    - Nvidia CUDA компилятор
    - Работает со всем, что есть в наборе инструментов CUDA
        
## Misc
- Huggingface