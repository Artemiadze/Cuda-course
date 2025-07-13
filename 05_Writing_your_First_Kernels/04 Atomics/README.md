# Что такое Atomic Operations
    
Под “atomic” мы имеем в виду концепцию неделимости в физике, согласно которой объект не может быть разделен на части.

Атомарная операция гарантирует, что определенная операция с ячейкой памяти будет полностью выполнена одним потоком до того, как другой поток сможет получить доступ к той же ячейке памяти или изменить ее. Это предотвращает возникновение условий гонки.

Поскольку мы ограничиваем объем работы, выполняемой с одним фрагментом памяти в единицу времени на протяжении всей атомной операции, мы немного теряем в скорости. Аппаратная гарантия безопасности памяти достигается за счет снижения скорости.

### **Integer Atomic Operations**

- **`atomicAdd(int* address, int val)`**: Атомарно (Atomically) добавляет `val` в значение `address` и возвращает старое значение.
- **`atomicSub(int* address, int val)`**: Атомарно subtracts `val` from the value at `address` и возвращает старое значение.
- **`atomicExch(int* address, int val)`**: Атомарно exchanges the value at `address` with `val` и возвращает старое значение.
- **`atomicMax(int* address, int val)`**: Атомарно устанавливает значение в `address` максимум текущего значения и `val`.
- **`atomicMin(int* address, int val)`**: Атомарно устанавливает значение в `address` минимум текущего значения и `val`.
- **`atomicAnd(int* address, int val)`**: Атомарно выполняет побитовое И значение в `address` и `val`.
- **`atomicOr(int* address, int val)`**: Атомарно выполняет побитовое ИЛИ значение в `address` и `val`.
- **`atomicXor(int* address, int val)`**: AtomicАтомарноally выполняет побитовое исключение значения в `address` и `val`.
- **`atomicCAS(int* address, int compare, int val)`**: Атомарно сравнивает значение в `address` с `compare`, и если они равны, замените его на `val`. Возвращается исходное значение.

### **Floating-Point Atomic Operations**

- **`atomicAdd(float* address, float val)`**: Атомарно добавляет `val` к значению в `address` и возвращает старое значение. Доступно с CUDA 2.0.
- Примечание: Атомарные операции с плавающей запятой над переменными двойной точности поддерживаются начиная с CUDA Compute Capability 6.0 с использованием `atomicAdd(double* address, double val)`.

### From Scratch

Современные графические процессоры имеют специальные аппаратные инструкции для эффективного выполнения этих операций. На аппаратном уровне они используют такие методы, как сравнение и замена (CAS).

Вы можете рассматривать атомарность как очень быструю операцию взаимообмена на аппаратном уровне. Как будто каждая атомарная операция выполняет это:

1. lock(memory_location)
2. old_value = *memory_location
3. *memory_location = old_value + increment
4. unlock(memory_location)
5. return old_value

```cpp
__device__ int softwareAtomicAdd(int* address, int increment) {
    __shared__ int lock;
    int old;
    
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();
    
    while (atomicCAS(&lock, 0, 1) != 0);  // Acquire lock
    
    old = *address;
    *address = old + increment;
    
    __threadfence();  // Ensure the write is visible to other threads
    
    atomicExch(&lock, 0);  // Release lock
    
    return old;
}
```


- Mutual Exclusion ⇒ https://www.youtube.com/watch?v=MqnpIwN7dz0&t
- "Взаимный" (Mutual):
    - Подразумевает взаимные или разделяемые отношения между объектами (в данном случае, потоками или процессами).
    - Предполагает, что исключение применяется в равной степени ко всем вовлеченным сторонам.
- "Исключение" (Exclusion):
    - Означает действие по сокрытию чего-либо или предотвращению доступа.
    - В данном контексте это означает предотвращение одновременного доступа к ресурсу.


```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// Our mutex structure
struct Mutex {
    int *lock;
};

// Initialize the mutex
__host__ void initMutex(Mutex *m) {
    cudaMalloc((void**)&m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// Acquire the mutex
__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0, 1) != 0) {
        // Spin-wait
    }
}

// Release the mutex
__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

// Kernel function to demonstrate mutex usage
__global__ void mutexKernel(int *counter, Mutex *m) {
    lock(m);
    // Critical section
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main() {
    Mutex m;
    initMutex(&m);
    
    int *d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple threads
    mutexKernel<<<1, 1000>>>(d_counter, &m);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Counter value: %d\n", result);
    
    cudaFree(m.lock);
    cudaFree(d_counter);
    
    return 0;
}
```