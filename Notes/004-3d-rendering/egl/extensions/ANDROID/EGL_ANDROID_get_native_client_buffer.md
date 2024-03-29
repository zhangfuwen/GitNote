# ANDROID_get_native_client_buffer

Name

    ANDROID_get_native_client_buffer

Name Strings

    EGL_ANDROID_get_native_client_buffer

Contributors

    Craig Donner

Contact

    Craig Donner, Google Inc. (cdonner 'at' google.com)

Status

    Complete

Version

    Version 3, October 11, 2017

Number

    EGL Extension #123

Dependencies

    Requires EGL 1.2.

    EGL_ANDROID_image_native_buffer and EGL_KHR_image_base are required.

    This extension is written against the wording of the EGL 1.2
    Specification as modified by EGL_KHR_image_base and
    EGL_ANDROID_image_native_buffer.

Overview

    This extension allows creating an EGLClientBuffer from an Android
    AHardwareBuffer object which can be later used to create an EGLImage.

New Types

    struct AHardwareBuffer

New Procedures and Functions

    EGLClientBuffer eglGetNativeClientBufferANDROID(const struct AHardwareBuffer *buffer)

New Tokens

    None

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add the following to section 2.5.1 "EGLImage Specification" (as modified by
    the EGL_KHR_image_base and EGL_ANDROID_image_native_buffer specifications),
    below the description of eglCreateImageKHR:

   "The command

        EGLClientBuffer eglGetNativeClientBufferANDROID(
                                const struct AHardwareBuffer *buffer)

    may be used to create an EGLClientBuffer from an AHardwareBuffer object.
    EGL implementations must guarantee that the lifetime of the returned
    EGLClientBuffer is at least as long as the EGLImage(s) it is bound to,
    following the lifetime semantics described below in section 2.5.2; the
    EGLClientBuffer must be destroyed no earlier than when all of its associated
    EGLImages are destroyed by eglDestroyImageKHR.

    Errors

        If eglGetNativeClientBufferANDROID fails, NULL will be returned, no
        memory will be allocated, and the following error will be generated:

       * If the value of buffer is NULL, the error EGL_BAD_PARAMETER is
         generated.

Issues

    1. Should this extension define what particular AHardwareBuffer formats EGL
    implementations are required to support?

    RESOLVED: No.

    The set of valid formats is implementation-specific and may depend on
    additional EGL extensions. The particular valid combinations for a given
    Android version and implementation should be documented by that version.

Revision History

#3 (Jesse Hall, October 11, 2017)
    - Assigned extension number, fixed minor issues for publication

#2 (Craig Donner, February 17, 2017)
    - Fix typographical errors.

#1 (Craig Donner, January 27, 2017)
    - Initial draft.
