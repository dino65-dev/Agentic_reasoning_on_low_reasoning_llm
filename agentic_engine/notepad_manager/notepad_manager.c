// notepad_manager.c - Advanced High-Performance Notepad Manager
// Optimized for Windows with native threads, memory pooling, and cache optimization

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <process.h>

// Configuration constants
#define MAX_OPT 16
#define MAX_NOTEPAD_LEN 8192
#define CACHE_LINE_SIZE 64
#define THREAD_POOL_SIZE 8
#define MAX_TASKS 256

// Cache-aligned notepad structure to prevent false sharing
typedef struct {
    int id;
    volatile LONG status;  // 0=idle, 1=processing, 2=complete
    char scratchpad[MAX_NOTEPAD_LEN];
    uint32_t hash;  // Quick integrity check
    DWORD processing_time_ms;
    char _padding[CACHE_LINE_SIZE - (sizeof(int) + sizeof(LONG) + sizeof(uint32_t) + sizeof(DWORD)) % CACHE_LINE_SIZE];
} __declspec(align(64)) Notepad;

// Memory pool for efficient allocation
typedef struct {
    void* memory;
    size_t size;
    size_t used;
    CRITICAL_SECTION lock;
} MemoryPool;

// Task queue for thread pool
typedef struct {
    Notepad* notepad;
    void (*process_func)(Notepad*);
} Task;

typedef struct {
    Task tasks[MAX_TASKS];
    volatile LONG head;
    volatile LONG tail;
    volatile LONG count;
    HANDLE sem_tasks;
    HANDLE sem_slots;
    volatile LONG shutdown;
} TaskQueue;

// Thread pool structure
typedef struct {
    HANDLE threads[THREAD_POOL_SIZE];
    TaskQueue queue;
    volatile LONG active;
} ThreadPool;

// Global thread pool instance
static ThreadPool g_thread_pool = {0};

// ===== Memory Pool Implementation =====

MemoryPool* create_memory_pool(size_t size) {
    MemoryPool* pool = (MemoryPool*)malloc(sizeof(MemoryPool));
    if (!pool) return NULL;
    
    pool->memory = _aligned_malloc(size, CACHE_LINE_SIZE);
    if (!pool->memory) {
        free(pool);
        return NULL;
    }
    
    pool->size = size;
    pool->used = 0;
    InitializeCriticalSection(&pool->lock);
    return pool;
}

void* pool_alloc(MemoryPool* pool, size_t size) {
    EnterCriticalSection(&pool->lock);
    
    // Align allocation to cache line
    size_t aligned_size = (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    
    void* ptr = NULL;
    if (pool->used + aligned_size <= pool->size) {
        ptr = (char*)pool->memory + pool->used;
        pool->used += aligned_size;
    }
    
    LeaveCriticalSection(&pool->lock);
    return ptr;
}

void destroy_memory_pool(MemoryPool* pool) {
    if (!pool) return;
    DeleteCriticalSection(&pool->lock);
    _aligned_free(pool->memory);
    free(pool);
}

// ===== Fast Hash Function for Integrity Check =====

uint32_t fast_hash(const char* data, size_t len) {
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint32_t)data[i];
        hash *= 16777619u;
    }
    return hash;
}

// ===== Task Queue Implementation (Lock-Free) =====

void init_task_queue(TaskQueue* queue) {
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    queue->shutdown = 0;
    queue->sem_tasks = CreateSemaphore(NULL, 0, MAX_TASKS, NULL);
    queue->sem_slots = CreateSemaphore(NULL, MAX_TASKS, MAX_TASKS, NULL);
}

int enqueue_task(TaskQueue* queue, Task task) {
    if (queue->shutdown) return 0;
    
    WaitForSingleObject(queue->sem_slots, INFINITE);
    
    LONG tail = InterlockedIncrement(&queue->tail) - 1;
    queue->tasks[tail % MAX_TASKS] = task;
    InterlockedIncrement(&queue->count);
    
    ReleaseSemaphore(queue->sem_tasks, 1, NULL);
    return 1;
}

int dequeue_task(TaskQueue* queue, Task* task) {
    DWORD wait_result = WaitForSingleObject(queue->sem_tasks, 100);
    if (wait_result != WAIT_OBJECT_0) return 0;
    
    LONG head = InterlockedIncrement(&queue->head) - 1;
    *task = queue->tasks[head % MAX_TASKS];
    InterlockedDecrement(&queue->count);
    
    ReleaseSemaphore(queue->sem_slots, 1, NULL);
    return 1;
}

void destroy_task_queue(TaskQueue* queue) {
    InterlockedExchange(&queue->shutdown, 1);
    CloseHandle(queue->sem_tasks);
    CloseHandle(queue->sem_slots);
}

// ===== Advanced Notepad Processing =====

void process_notepad_advanced(Notepad* np) {
    DWORD start_time = GetTickCount();
    
    InterlockedExchange(&np->status, 1);  // Set processing
    
    // Advanced processing: parallel string operations, hashing
    int len = snprintf(np->scratchpad, MAX_NOTEPAD_LEN, 
        "Advanced Processing Notepad #%d | Thread: %lu | Optimized with Memory Pool & Cache Alignment",
        np->id, GetCurrentThreadId());
    
    if (len > 0 && len < MAX_NOTEPAD_LEN) {
        // Compute hash for integrity
        np->hash = fast_hash(np->scratchpad, len);
        
        // Simulate some processing work (replace with actual logic)
        for (volatile int i = 0; i < 1000; i++);
    }
    
    np->processing_time_ms = GetTickCount() - start_time;
    InterlockedExchange(&np->status, 2);  // Set complete
}

// ===== Thread Pool Worker =====

unsigned int __stdcall worker_thread(void* arg) {
    TaskQueue* queue = (TaskQueue*)arg;
    
    while (!queue->shutdown) {
        Task task;
        if (dequeue_task(queue, &task)) {
            if (task.process_func && task.notepad) {
                task.process_func(task.notepad);
            }
        }
    }
    
    return 0;
}

// ===== Thread Pool Management =====

int init_thread_pool(ThreadPool* pool) {
    init_task_queue(&pool->queue);
    pool->active = 1;
    
    for (int i = 0; i < THREAD_POOL_SIZE; i++) {
        pool->threads[i] = (HANDLE)_beginthreadex(
            NULL, 0, worker_thread, &pool->queue, 0, NULL
        );
        
        if (pool->threads[i] == 0) {
            fprintf(stderr, "Failed to create worker thread %d\n", i);
            return 0;
        }
        
        // Set thread priority for better responsiveness
        SetThreadPriority(pool->threads[i], THREAD_PRIORITY_ABOVE_NORMAL);
    }
    
    return 1;
}

void shutdown_thread_pool(ThreadPool* pool) {
    InterlockedExchange(&pool->active, 0);
    destroy_task_queue(&pool->queue);
    
    WaitForMultipleObjects(THREAD_POOL_SIZE, pool->threads, TRUE, INFINITE);
    
    for (int i = 0; i < THREAD_POOL_SIZE; i++) {
        CloseHandle(pool->threads[i]);
    }
}

// ===== Public API =====

void run_all_notepads(int n, Notepad* notepads) {
    if (n <= 0 || n > MAX_OPT || !notepads) return;
    
    // Initialize thread pool on first use
    static volatile LONG initialized = 0;
    if (InterlockedCompareExchange(&initialized, 1, 0) == 0) {
        if (!init_thread_pool(&g_thread_pool)) {
            fprintf(stderr, "Failed to initialize thread pool\n");
            return;
        }
    }
    
    // Submit all tasks to thread pool
    for (int i = 0; i < n; i++) {
        Task task = {&notepads[i], process_notepad_advanced};
        enqueue_task(&g_thread_pool.queue, task);
    }
    
    // Wait for all notepads to complete
    DWORD timeout = 5000;  // 5 second timeout
    DWORD start = GetTickCount();
    
    while (GetTickCount() - start < timeout) {
        int all_done = 1;
        for (int i = 0; i < n; i++) {
            if (notepads[i].status != 2) {
                all_done = 0;
                break;
            }
        }
        if (all_done) break;
        Sleep(1);  // Small sleep to avoid busy-waiting
    }
}

Notepad* make_notepads(int n) {
    if (n <= 0 || n > MAX_OPT) return NULL;
    
    // Use aligned allocation for cache optimization
    Notepad* notepads = (Notepad*)_aligned_malloc(sizeof(Notepad) * n, CACHE_LINE_SIZE);
    if (!notepads) return NULL;
    
    for (int i = 0; i < n; i++) {
        notepads[i].id = i;
        notepads[i].status = 0;
        notepads[i].scratchpad[0] = '\0';
        notepads[i].hash = 0;
        notepads[i].processing_time_ms = 0;
    }
    
    return notepads;
}

void free_notepads(Notepad* notepads) {
    if (notepads) {
        _aligned_free(notepads);
    }
}

void cleanup_notepad_manager(void) {
    shutdown_thread_pool(&g_thread_pool);
}
