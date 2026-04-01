/*
 *  timing.h
 *
 *
 *
 */

#ifndef TIMING_H
#define TIMING_H

#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Subtract the `struct timeval' value 'then' from 'now',
   returning the difference as a float representing seconds
   elapsed.
*/
float elapsedTime(struct timeval now, struct timeval then);

double currentTime();

#ifdef __cplusplus
}
#endif

#endif
