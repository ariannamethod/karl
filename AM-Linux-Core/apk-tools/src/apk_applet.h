#ifndef APK_APPLET_H
#define APK_APPLET_H

#include <getopt.h>
#include "apk_defines.h"
#include "apk_database.h"

struct apk_option {
	int val;
	const char *name;
	const char *help;
	int has_arg;
	const char *arg_name;
};

struct apk_applet {
	const char *name;
	const char *arguments;
	const char *help;

	unsigned int open_flags, forced_flags;
	int context_size;
	int num_options;
	struct apk_option *options;

	int (*parse)(void *ctx, struct apk_db_options *dbopts,
		     int optch, int optindex, const char *optarg);
	int (*main)(void *ctx, struct apk_database *db, int argc, char **argv);
};

extern struct apk_applet *__start_apkapplets, *__stop_apkapplets;

#define APK_DEFINE_APPLET(x) \
	static struct apk_applet *__applet_##x \
	__attribute__((__section__("apkapplets"))) __attribute((used)) = &x;

#endif
