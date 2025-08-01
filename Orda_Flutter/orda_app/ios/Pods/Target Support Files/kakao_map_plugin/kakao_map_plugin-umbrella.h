#ifdef __OBJC__
#import <UIKit/UIKit.h>
#else
#ifndef FOUNDATION_EXPORT
#if defined(__cplusplus)
#define FOUNDATION_EXPORT extern "C"
#else
#define FOUNDATION_EXPORT extern
#endif
#endif
#endif

#import "KakaoMapPlugin.h"

FOUNDATION_EXPORT double kakao_map_pluginVersionNumber;
FOUNDATION_EXPORT const unsigned char kakao_map_pluginVersionString[];

