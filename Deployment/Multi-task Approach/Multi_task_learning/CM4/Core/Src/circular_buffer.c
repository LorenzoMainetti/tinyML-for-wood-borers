/*
 * circular_buffer.c
 *
 *  Created on: Apr 7, 2023
 *      Author: loren
 */

#include "circular_buffer.h"


// The hidden definition of our circular buffer structure
struct circular_buf_t {
	float32_t * buffer;
	size_t head;
	size_t tail;
	size_t max; //of the buffer
};


cbuf_handle_t circular_buf_init(float32_t* buffer, size_t size)
{
	assert(buffer && size);

	cbuf_handle_t cbuf = malloc(sizeof(circular_buf_t));
	assert(cbuf);

	cbuf->buffer = buffer;
	cbuf->max = size;
	circular_buf_reset(cbuf);

	assert(circular_buf_empty(cbuf));

	return cbuf;
}


void circular_buf_reset(cbuf_handle_t me)
{
    assert(me);

    me->head = 0;
    me->tail = 0;
}


void circular_buf_free(cbuf_handle_t me)
{
	assert(me);
	free(me);
}


bool circular_buf_full(circular_buf_t* me)
{
	// We need to handle the wrap around case
    size_t head = me->head + 1;
    if(head == me->max)
   {
	head = 0;
   }

	return head == me->tail;
}


bool circular_buf_empty(circular_buf_t* me)
{
	// We define empty as head == tail
    return (me->head == me->tail);
}


size_t circular_buf_capacity(cbuf_handle_t me)
{
	assert(me);

	return me->max;
}


size_t circular_buf_size(cbuf_handle_t me)
{
	assert(me);

	size_t size = me->max;

	if(!circular_buf_full(me))
	{
		if(me->head >= me->tail)
		{
			size = (me->head - me->tail);
		}
		else
		{
			size = (me->max + me->head - me->tail);
		}
	}

	return size;
}


static void advance_pointer(cbuf_handle_t me)
{
	assert(me);

	if(circular_buf_full(me))
   	{
		if (++(me->tail) == me->max)
		{
			me->tail = 0;
		}
	}

	if (++(me->head) == me->max)
	{
		me->head = 0;
	}
}


static void retreat_pointer(cbuf_handle_t me)
{
	assert(me);

	if (++(me->tail) == me->max)
	{
		me->tail = 0;
	}
}


void circular_buf_put(cbuf_handle_t me, float32_t data)
{
	assert(me && me->buffer);

    me->buffer[me->head] = data;

    advance_pointer(me);
}


bool circular_buf_get(cbuf_handle_t me, float32_t * data)
{
    assert(me && me->buffer);

    bool r = false;

    if(!circular_buf_empty(me))
    {
        *data = me->buffer[me->tail];
        retreat_pointer(me);

        r = true;
    }

    return r;
}


