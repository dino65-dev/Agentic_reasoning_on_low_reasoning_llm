// notepad_manager.h - Header file for advanced notepad manager
// Include this in your projects to use the notepad manager API

#ifndef NOTEPAD_MANAGER_H
#define NOTEPAD_MANAGER_H

#include <stdint.h>
#include <windows.h>

// Configuration
#define MAX_OPT 16
#define MAX_NOTEPAD_LEN 8192
#define CACHE_LINE_SIZE 64

// Notepad structure (aligned for cache optimization)
typedef struct {
    int id;
    volatile LONG status;  // 0=idle, 1=processing, 2=complete
    char scratchpad[MAX_NOTEPAD_LEN];
    uint32_t hash;  // FNV-1a hash for integrity
    DWORD processing_time_ms;
    char _padding[CACHE_LINE_SIZE - (sizeof(int) + sizeof(LONG) + sizeof(uint32_t) + sizeof(DWORD)) % CACHE_LINE_SIZE];
} __declspec(align(64)) Notepad;

// Public API functions

/**
 * Create an array of N notepads with cache-aligned allocation
 * @param n Number of notepads to create (1 to MAX_OPT)
 * @return Pointer to notepad array, or NULL on failure
 * @note Must be freed with free_notepads()
 */
Notepad* make_notepads(int n);

/**
 * Process all notepads in parallel using the thread pool
 * @param n Number of notepads in the array
 * @param notepads Pointer to notepad array
 * @note Initializes thread pool on first call
 * @note Blocks until all notepads complete or timeout (5 seconds)
 */
void run_all_notepads(int n, Notepad* notepads);

/**
 * Free notepad array (uses aligned deallocation)
 * @param notepads Pointer to notepad array
 */
void free_notepads(Notepad* notepads);

/**
 * Cleanup and shutdown the thread pool
 * @note Call this before program exit to properly cleanup resources
 * @note Safe to call multiple times
 */
void cleanup_notepad_manager(void);

// Status codes for Notepad.status field
#define NOTEPAD_IDLE       0
#define NOTEPAD_PROCESSING 1
#define NOTEPAD_COMPLETE   2

// Return codes
#define NOTEPAD_SUCCESS 0
#define NOTEPAD_ERROR   -1

#endif // NOTEPAD_MANAGER_H