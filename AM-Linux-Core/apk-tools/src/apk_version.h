#ifndef APK_VERSION_H
#define APK_VERSION_H

#include "apk_blob.h"

#define APK_VERSION_EQUAL		1
#define APK_VERSION_LESS		2
#define APK_VERSION_GREATER		4

const char *apk_version_op_string(int result_mask);
int apk_version_result_mask(const char *str);
int apk_version_validate(apk_blob_t ver);
int apk_version_compare_blob(apk_blob_t a, apk_blob_t b);
int apk_version_compare(const char *str1, const char *str2);

#endif
