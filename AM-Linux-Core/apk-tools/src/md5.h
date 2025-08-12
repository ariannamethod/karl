#ifndef MD5_H
#define MD5_H

#include <sys/types.h>

typedef unsigned char md5sum_t[16];
typedef u_int32_t md5_uint32;

struct md5_ctx
{
	md5_uint32 A;
	md5_uint32 B;
	md5_uint32 C;
	md5_uint32 D;

	md5_uint32 total[2];
	md5_uint32 buflen;
	char buffer[128];
};

/* Initialize structure containing state of computation.
   (RFC 1321, 3.3: Step 3)  */
void md5_init(struct md5_ctx *ctx);

/* Starting with the result of former calls of this function (or the
   initialization function update the context for the next LEN bytes
   starting at BUFFER.
   It is NOT required that LEN is a multiple of 64.  */
void md5_process(struct md5_ctx *ctx, const void *buffer, size_t len);

/* Process the remaining bytes in the buffer and put result from CTX
   in first 16 bytes following RESBUF.  The result is always in little
   endian byte order, so that a byte-wise output yields to the wanted
   ASCII representation of the message digest.

   IMPORTANT: On some systems it is required that RESBUF is correctly
   aligned for a 32 bits value.  */
void md5_finish(struct md5_ctx *ctx, md5sum_t resbuf);

#endif

