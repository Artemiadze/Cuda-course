# CUDA API 
> Includes cuBLAS, cuDNN, cuBLASmp

- поначалу термин “API” может сбить с толку. все это означает, что у нас есть библиотека, в которой мы не можем видеть внутренние компоненты. существует документация по вызовам функций в API, но это предварительно скомпилированный двоичный файл, который не предоставляет исходный код. код сильно оптимизирован, но вы его не видите. (сохраните это здесь, поскольку это универсально применимо ко всем библиотекам/API, перечисленным ниже)

## Opaque Struct Types (CUDA API):
- вы не можете видеть или прикасаться к внутренним элементам типа, только к внешним, таким как имена, аргументы функций и т.д. На файл `.so` (общий объект) ссылаются как на непрозрачный двоичный файл, чтобы просто запускать скомпилированные функции с высокой пропускной способностью. Если вы поищете cuFFT, cuDNN или любое другое расширение CUDA, вы заметите, что оно поставляется в виде API, невозможность доступа к исходному коду assembly/C/C++ связана с использованием слова “непрозрачный”. типы структур - это всего лишь общий тип в C, который позволяет NVIDIA правильно создавать экосистему. cublasLtHandle_t - это пример непрозрачного типа, содержащего контекст для операции cublas Lt

Если вы пытаетесь просто выяснить, как получить максимально быстрый вывод для работы в вашем кластере, вам нужно будет разобраться в деталях. Чтобы ориентироваться в CUDA API, я бы рекомендовал использовать следующие приемы:
1. [perplexity.ai](http://perplexity.ai) (самая свежая информация и возможность получать данные в режиме реального времени)
2. поиск в Google (возможно, это хуже, чем недоумение, но можно воспользоваться классическим подходом к выяснению ситуации)
3. Чат для получения общих знаний, которые с меньшей вероятностью будут недоступны для обучения
4. поиск по ключевым словам в документах nvidia


## Error Checking (API Specific)

- cuBLAS для примера

```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- cuDNN пример

```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- Необходимость в проверке ошибок возникает следующим образом: у вас есть контекст для вызова CUDA API, который вы настраиваете, затем вы вызываете операцию, затем вы проверяете статус операции, передавая вызов API в поле "вызов" в макросе. Если он завершится успешно, ваш код продолжит выполнение, как и ожидалось. В случае сбоя вы получите описательное сообщение об ошибке, а не просто ошибку сегментации или некорректный результат, который не отображается автоматически.
- Очевидно, что для других API CUDA существует больше макросов для проверки ошибок, но это наиболее распространенные из них (необходимые для этого курса).
- Попробуйте прочитать это руководство здесь -> [Proper CUDA Error Checking](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/)


## Matrix Multiplication
- cuDNN неявно поддерживает matmul посредством специальных операций свертки и глубокого обучения, но не представлен в качестве одной из основных функций cuDNN
- Для умножения матриц вам лучше всего использовать программу глубокого изучения операций линейной алгебры в cuBLAS, поскольку она имеет более широкий охват и настроена на высокую пропускную способность matmul.
> Дополнительные примечания (представлены для того, чтобы показать, что не так уж сложно передать знания, скажем, о cuDNN, в cuFFT, используя способ настройки и вызова операции)

## Resources:
- [CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples)