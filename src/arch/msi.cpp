#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MEMORY_SIZE 100
#define CORES 2

// Define MSI states
typedef enum {
    MODIFIED,
    SHARED,
    INVALID
} CacheState;

// Shared memory
CacheState memory[MEMORY_SIZE];

// Define core structure
typedef struct {
    int core_id;
    CacheState cache_state;
} Core;

// Function declarations
void* core_thread(void* arg);
void read_from_memory(Core* core, int address);
void write_to_memory(Core* core, int address);

// Mutex for synchronization
pthread_mutex_t memory_mutex = PTHREAD_MUTEX_INITIALIZER;

int main() {
    // Initialize memory with INVALID state
    for (int i = 0; i < MEMORY_SIZE; i++) {
        memory[i] = INVALID;
    }

    // Create an array of cores
    Core cores[CORES];

    // Create threads for each core
    pthread_t threads[CORES];

    // Initialize cores
    for (int i = 0; i < CORES; i++) {
        cores[i].core_id = i;
        cores[i].cache_state = INVALID;
        pthread_create(&threads[i], NULL, core_thread, &cores[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < CORES; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}

void* core_thread(void* arg) {
    Core* core = (Core*)arg;

    // Simulate read and write operations
    for (int i = 0; i < 5; i++) {
        int address = rand() % MEMORY_SIZE;

        // Simulate read or write with equal probability
        if (rand() % 2 == 0) {
            read_from_memory(core, address);
        } else {
            write_to_memory(core, address);
        }
    }

    pthread_exit(NULL);
}

void read_from_memory(Core* core, int address) {
    // Acquire lock before accessing shared memory
    pthread_mutex_lock(&memory_mutex);

    printf("Core %d is reading from memory address %d\n", core->core_id, address);

    // Simulate reading from memory
    core->cache_state = memory[address];

    printf("Core %d read data from memory: State = %d\n", core->core_id, core->cache_state);

    // Release lock after accessing shared memory
    pthread_mutex_unlock(&memory_mutex);
}

void write_to_memory(Core* core, int address) {
    // Acquire lock before accessing shared memory
    pthread_mutex_lock(&memory_mutex);

    printf("Core %d is writing to memory address %d\n", core->core_id, address);

    // Simulate writing to memory
    memory[address] = MODIFIED;
    core->cache_state = MODIFIED;

    printf("Core %d wrote data to memory: State = %d\n", core->core_id, core->cache_state);

    // Release lock after accessing shared memory
    pthread_mutex_unlock(&memory_mutex);
}
