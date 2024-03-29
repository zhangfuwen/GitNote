# ANDROID_image_native_buffer

Name

    ANDROID_image_native_buffer

Name Strings

    EGL_ANDROID_image_native_buffer

Contributors

    Mathias Agopian
    Jamie Gennis
    Jesse Hall

Contact

    Jesse Hall, Google Inc. (jessehall 'at' google.com)

Status

    Complete

Version

    Version 1, November 28, 2012

Number

    EGL Extension #49

Dependencies

    EGL 1.2 is required.

    EGL_KHR_image_base is required.

    This extension is written against the wording of the EGL 1.2
    Specification.

Overview

    This extension enables using an Android window buffer (struct
    ANativeWindowBuffer) as an EGLImage source.

New Types

    None.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <target> parameter of eglCreateImageKHR:

    EGL_NATIVE_BUFFER_ANDROID              0x3140

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add to section 2.5.1 "EGLImage Specification" (as defined by the
    EGL_KHR_image_base specification), in the description of
    eglCreateImageKHR:

   "Values accepted for <target> are listed in Table aaa, below.

      +----------------------------+-----------------------------------------+
      |  <target>                  |  Notes                                  |
      +----------------------------+-----------------------------------------+
      |  EGL_NATIVE_BUFFER_ANDROID |  Used for ANativeWindowBuffer objects   |
      +----------------------------+-----------------------------------------+
       Table aaa.  Legal values for eglCreateImageKHR <target> parameter

    ...

    If <target> is EGL_NATIVE_BUFFER_ANDROID, <dpy> must be a valid display,
    <ctx> must be EGL_NO_CONTEXT, <buffer> must be a pointer to a valid
    ANativeWindowBuffer object (cast into the type EGLClientBuffer), and
    attributes other than EGL_IMAGE_PRESERVED_KHR are ignored."

    Add to the list of error conditions for eglCreateImageKHR:

      "* If <target> is EGL_NATIVE_BUFFER_ANDROID and <buffer> is not a
         pointer to a valid ANativeWindowBuffer, the error EGL_BAD_PARAMETER
         is generated.

       * If <target> is EGL_NATIVE_BUFFER_ANDROID and <ctx> is not
         EGL_NO_CONTEXT, the error EGL_BAD_CONTEXT is generated.

       * If <target> is EGL_NATIVE_BUFFER_ANDROID and <buffer> was created
         with properties (format, usage, dimensions, etc.) not supported by
         the EGL implementation, the error EGL_BAD_PARAMETER is generated."

Issues

    1. Should this extension define what combinations of ANativeWindowBuffer
    properties implementations are required to support?

    RESOLVED: No.

    The requirements have evolved over time and will continue to change with
    future Android releases. The minimum requirements for a given Android
    version should be documented by that version.

Revision History

#1 (Jesse Hall, November 28, 2012)
    - Initial draft.
