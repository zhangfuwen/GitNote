# TIZEN_image_native_buffer

Name

    TIZEN_image_native_buffer

Name Strings

    EGL_TIZEN_image_native_buffer

Contributors

    Dongyeon Kim
    Minsu Han
    Inpyo Kang
    Zeeshan Anwar

Contact

    Dongyeon Kim, Samsung Electronics (dy5.kim 'at' samsung.com)
    Zeeshan Anwar, Samsung Electronics (z.anwar 'at' samsung.com)

Status

    This extension is obsolete and has been replaced by 
	EGL_TIZEN_image_native_buffer

Version

    Version 2, July 23, 2014

Number

    EGL Extension #76

Dependencies

    EGL 1.2 is required.

    EGL_KHR_image_base is required.

    This extension is written against the wording of the EGL 1.2
    Specification.

Overview

    This extension enables using a Tizen native buffer (struct
    native_buffer) as an EGLImage source.

New Types

    None.

New Procedures and Functions    

    None.

New Tokens

    Accepted by the <target> parameter of eglCreateImageKHR:

    EGL_NATIVE_BUFFER_TIZEN             0x32A0

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add to section 2.5.1 "EGLImage Specification" (as defined by the
    EGL_KHR_image_base specification), in the description of
    eglCreateImageKHR:

   "Values accepted for <target> are listed in Table aaa, below.

      +-------------------------+--------------------------------------+
      | <target>                | Notes                                |
      +-------------------------+--------------------------------------+
      | EGL_NATIVE_BUFFER_TIZEN | Used for Tizen native_buffer objects |
      +-------------------------+--------------------------------------+
       Table aaa.  Legal values for eglCreateImageKHR <target> parameter

    ...

    If <target> is EGL_NATIVE_BUFFER_TIZEN, <dpy> must be a valid display,
    <ctx> must be EGL_NO_CONTEXT, <buffer> must be a pointer to a valid
    native_buffer object (cast into the type EGLClientBuffer), and
    attributes other than EGL_IMAGE_PRESERVED_KHR are ignored."

    Add to the list of error conditions for eglCreateImageKHR:

      "* If <target> is EGL_NATIVE_BUFFER_TIZEN and <buffer> is not a
         pointer to a valid native_buffer, the error EGL_BAD_PARAMETER
         is generated.

       * If <target> is EGL_NATIVE_BUFFER_TIZEN and <ctx> is not
         EGL_NO_CONTEXT, the error EGL_BAD_CONTEXT is generated.

       * If <target> is EGL_NATIVE_BUFFER_TIZEN and <buffer> was created
         with properties (format, usage, dimensions, etc.) not supported by
         the EGL implementation, the error EGL_BAD_PARAMETER is generated."

Issues

    1. Should this extension define what combinations of native_buffer
    properties implementations are required to support?

    RESOLVED: No.

    The requirements have evolved over time and will continue to change with
    future Tizen releases. The minimum requirements for a given Tizen
    version should be documented by that version.


Revision History

#2 (Zeeshan Anwar, July 01, 2014)
    - Assigned the value to EGL_NATIVE_BUFFER_TIZEN
#1 (Dongyeon Kim, June 05, 2013)
    - Initial draft.
