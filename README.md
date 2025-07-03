# CUDA Course

GitHub репозиторий для изучения работы с CUDA и GPU. Материалы репозитория были переведены на русский и опираются на [youtube курс](https://www.youtube.com/watch?v=86FAWCzIe_4&list=WL&index=2) от [freeCodeCamp.org](https://www.youtube.com/@freecodecamp)

> Note: This course is designed for Ubuntu Linux. Windows users can use Windows Subsystem for Linux or Docker containers to simulate the ubuntu Linux environment.

## Table of Contents

1. [The Deep Learning Ecosystem](01_Deep_Learning_Ecosystem/README.md)
2. [Setup/Installation](02_Setup/README.md)
3. [C/C++ Review](03_C_and_C++_Review/README.md)
4. [Gentle Intro to GPUs](04_Gentle_Intro_to_GPUs/README.md)
5. [Writing Your First Kernels](05_Writing_your_First_Kernels/README.md)
6. [CUDA APIs (cuBLAS, cuDNN, etc)](06_CUDA_APIs/README.md)
7. [Optimizing Matrix Multiplication](07_Faster_Matmul/README.md)
8. [Triton](08_Triton/README.md)
9. [PyTorch Extensions (CUDA)](08_PyTorch_Extensions/README.md)
10. [Final Project](09_Final_Project/README.md)
11. [Extras](10_Extras/README.md)

## Course Philosophy

Цели этого курса:

- Снизит барьер для поступления на работу в сфере высокопроизводительных вычислений
- Обеспечит основу для понимания таких проектов, как Karpathy's [llm.c](https://github.com/karpathy/llm.c)
- Объединит разрозненные ресурсы по программированию на CUDA в комплексный, организованный курс

## Overview

- Сосредоточен на оптимизации ядра графического процессора для повышения производительности
- Охватывает CUDA, PyTorch и Triton
- Уделяет особое внимание техническим деталям написания более быстрых ядер
- Адаптирован для графических процессоров NVIDIA
- Завершается простым проектом MLP MNIST в CUDA

## Prerequisites

- Программирование на Python (обязательно)
- Основы дифференцирования и векторного анализа для backprop (рекомендуется)
- Основы линейной алгебры (рекомендуется)

## Key Takeaways

- Оптимизация существующих реализаций
- Создание ядер CUDA для передовых исследований
- Понимание узких мест в производительности графического процессора, особенно пропускной способности памяти

## Hardware Requirements

- Любой графический процессор NVIDIA GTX, RTX или уровня центра обработки данных
- Возможности облачного графического процессора доступны для тех, у кого нет локального оборудования

## Use Cases for CUDA/GPU Programming

- Deep Learning (primary focus of this course)
- Graphics and Ray-tracing
- Fluid Simulation
- Video Editing
- Crypto Mining
- 3D modeling
- Anything that requires parallel processing with large arrays

## Resources

- Репозиторий на GitHub (этот репозиторий)
- Stack Overflow
- Форумы разработчиков NVIDIA
- Документация NVIDIA и PyTorch
- LLMS для навигации по пространству
- Cheatsheet [здесь](/11_Extras/assets/cheatsheet.md)

## Fun YouTube Videos:
- [How do GPUs works? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU)
- [But how do GPUs actually work?](https://www.youtube.com/watch?v=58jtf24uijw&ab_channel=Graphicode)
- [Getting Started With CUDA for Python Programmers](https://www.youtube.com/watch?v=nOxKexn3iBo&ab_channel=JeremyHoward)
- [Transformers Explained From The Atom Up](https://www.youtube.com/watch?v=7lJZHbg0EQ4&ab_channel=JacobRintamaki)
- [How CUDA Programming Works - Stephen Jones, CUDA Architect, NVIDIA](https://www.youtube.com/watch?v=QQceTDjA4f4&ab_channel=ChristopherHollinworth)
- [Parallel Computing with Nvidia CUDA - NeuralNine](https://www.youtube.com/watch?v=zSCdTOKrnII&ab_channel=NeuralNine)
- [CPU vs GPU vs TPU vs DPU vs QPU](https://www.youtube.com/watch?v=r5NQecwZs1A&ab_channel=Fireship)
- [Nvidia CUDA in 100 Seconds](https://www.youtube.com/watch?v=pPStdjuYzSI&ab_channel=Fireship)
- [How AI Discovered a Faster Matrix Multiplication Algorithm](https://www.youtube.com/watch?v=fDAPJ7rvcUw&t=1s&ab_channel=QuantaMagazine)
- [The fastest matrix multiplication algorithm](https://www.youtube.com/watch?v=sZxjuT1kUd0&ab_channel=Dr.TreforBazett)
- [From Scratch: Cache Tiled Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=ga2ML1uGr5o&ab_channel=CoffeeBeforeArch)
- [From Scratch: Matrix Multiplication in CUDA](https://www.youtube.com/watch?v=DpEgZe2bbU0&ab_channel=CoffeeBeforeArch)
- [Intro to GPU Programming](https://www.youtube.com/watch?v=G-EimI4q-TQ&ab_channel=TomNurkkala)
- [CUDA Programming](https://www.youtube.com/watch?v=xwbD6fL5qC8&ab_channel=TomNurkkala)
- [Intro to CUDA (part 1): High Level Concepts](https://www.youtube.com/watch?v=4APkMJdiudU&ab_channel=JoshHolloway)
- [Intro to GPU Hardware](https://www.youtube.com/watch?v=kUqkOAU84bA&ab_channel=TomNurkkala)
