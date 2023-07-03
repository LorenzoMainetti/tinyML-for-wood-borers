/*
 * circular_buffer.h
 *
 *  Created on: Apr 7, 2023
 *      Author: loren
 */

#ifndef INC_CIRCULAR_BUFFER_H_
#define INC_CIRCULAR_BUFFER_H_

#include "arm_math.h"
#include <stdlib.h>
#include <stdbool.h>

typedef struct circular_buf_t circular_buf_t;
typedef circular_buf_t* cbuf_handle_t;

/// Pass in a storage buffer and size
/// Returns a circular buffer handle
cbuf_handle_t circular_buf_init(float32_t* buffer, size_t size);

/// Free a circular buffer structure.
/// Does not free data buffer; owner is responsible for that
void circular_buf_free(cbuf_handle_t me);

/// Reset the circular buffer to empty, head == tail
void circular_buf_reset(cbuf_handle_t me);

/// Continues to add data if the buffer is full
/// Old data is overwritten
void circular_buf_put(cbuf_handle_t me, float32_t data);

/// Retrieve a value from the buffer
/// Returns true on success, false if the buffer is empty
bool circular_buf_get(cbuf_handle_t me, float32_t * data);

/// Returns true if the buffer is empty
bool circular_buf_empty(cbuf_handle_t me);

/// Returns true if the buffer is full
bool circular_buf_full(cbuf_handle_t me);

/// Returns the maximum capacity of the buffer
size_t circular_buf_capacity(cbuf_handle_t me);

/// Returns the current number of elements in the buffer
size_t circular_buf_size(cbuf_handle_t me);

#endif /* INC_CIRCULAR_BUFFER_H_ */
