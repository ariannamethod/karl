#ifndef APK_PRINT_H
#define APK_PRINT_H

#include "apk_blob.h"

#define apk_error(args...)	do { apk_log("ERROR: ", args); } while (0)
#define apk_warning(args...)	do { if (apk_verbosity > 0) { apk_log("WARNING: ", args); } } while (0)
#define apk_message(args...)	do { if (apk_verbosity > 0) { apk_log(NULL, args); } } while (0)

void apk_log(const char *prefix, const char *format, ...);
const char *apk_error_str(int error);

int apk_print_indented(struct apk_indent *i, apk_blob_t blob);
void apk_print_indented_words(struct apk_indent *i, const char *text);

#endif
