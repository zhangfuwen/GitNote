# ANDROID_create_native_client_buffer

Name

    ANDROID_create_native_client_buffer

Name Strings

    EGL_ANDROID_create_native_client_buffer

Contributors

    Craig Donner

Contact

    Craig Donner, Google Inc. (cdonner 'at' google.com)

Status

    Draft

Version

    Version 1, January 19, 2016

Number

    EGL Extension #99

Dependencies

    Requires EGL 1.2.

    EGL_ANDROID_image_native_buffer and EGL_KHR_image_base are required.

    This extension is written against the wording of the EGL 1.2
    Specification as modified by EGL_KHR_image_base and
    EGL_ANDROID_image_native_buffer.

Overview

    This extension allows creating an EGLClientBuffer backed by an Android
    window buffer (struct ANativeWindowBuffer) which can be later used to
    create an EGLImage.

New Types

    None.

New Procedures and Functions

    EGLClientBuffer eglCreateNativeClientBufferANDROID(
                        const EGLint *attrib_list)

New Tokens

    EGL_NATIVE_BUFFER_USAGE_ANDROID                  0x3143
    EGL_NATIVE_BUFFER_USAGE_PROTECTED_BIT_ANDROID    0x00000001
    EGL_NATIVE_BUFFER_USAGE_RENDERBUFFER_BIT_ANDROID 0x00000002
    EGL_NATIVE_BUFFER_USAGE_TEXTURE_BIT_ANDROID      0x00000004

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add the following to section 2.5.1 "EGLImage Specification" (as modified by
    the EGL_KHR_image_base and EGL_ANDROID_image_native_buffer specifications),
    below the description of eglCreateImageKHR:

   "The command

        EGLClientBuffer eglCreateNativeClientBufferANDROID(
                                const EGLint *attrib_list)

    may be used to create an EGLClientBuffer backed by an ANativeWindowBuffer
    struct. EGL implementations must guarantee that the lifetime of the
    returned EGLClientBuffer is at least as long as the EGLImage(s) it is bound
    to, following the lifetime semantics described below in section 2.5.2; the
    EGLClientBuffer must be destroyed no earlier than when all of its associated
    EGLImages are destroyed by eglDestroyImageKHR. <attrib_list> is a list of
    attribute-value pairs which is used to specify the dimensions, format, and
    usage of the underlying buffer structure. If <attrib_list> is non-NULL, the
    last attribute specified in the list must be EGL_NONE.

    Attribute names accepted in <attrib_list> are shown in Table aaa,
    together with the <target> for which each attribute name is valid, and
    the default value used for each attribute if it is not included in
    <attrib_list>.

      +---------------------------------+----------------------+---------------+
      | Attribute                       | Description          | Default Value |
      |                                 |                      |               |
      +---------------------------------+----------------------+---------------+
      | EGL_NONE                        | Marks the end of the | N/A           |
      |                                 | attribute-value list |               |
      | EGL_WIDTH                       | The width of the     | 0             |
      |                                 | buffer data          |               |
      | EGL_HEIGHT                      | The height of the    | 0             |
      |                                 | buffer data          |               |
      | EGL_RED_SIZE                    | The bits of Red in   | 0             |
      |                                 | the color buffer     |               |
      | EGL_GREEN_SIZE                  | The bits of Green in | 0             |
      |                                 | the color buffer     |               |
      | EGL_BLUE_SIZE                   | The bits of Blue in  | 0             |
      |                                 | the color buffer     |               |
      | EGL_ALPHA_SIZE                  | The bits of Alpha in | 0             |
      |                                 | the color buffer     |               |
      |                                 | buffer data          |               |
      | EGL_NATIVE_BUFFER_USAGE_ANDROID | The usage bits of    | 0             |
      |                                 | the buffer data      |               |
      +---------------------------------+----------------------+---------------+
       Table aaa.  Legal attributes for eglCreateNativeClientBufferANDROID
       <attrib_list> parameter.

    The maximum width and height may depend on the amount of available memory,
    which may also depend on the format and usage flags. The values of
    EGL_RED_SIZE, EGL_GREEN_SIZE, and EGL_BLUE_SIZE must be non-zero and
    correspond to a valid pixel format for the implementation. If EGL_ALPHA_SIZE
    is non-zero then the combination of all four sizes must correspond to a
    valid pixel format for the implementation. The
    EGL_NATIVE_BUFFER_USAGE_ANDROID flag may include any of the following bits:

        EGL_NATIVE_BUFFER_USAGE_PROTECTED_BIT_ANDROID: Indicates that the
        created buffer must have a hardware-protected path to external display
        sink. If a hardware-protected path is not available, then either don't
        composite only this buffer (preferred) to the external sink, or (less
        desirable) do not route the entire composition to the external sink.

        EGL_NATIVE_BUFFER_USAGE_RENDERBUFFER_BIT_ANDROID: The buffer will be
        used to create a renderbuffer. This flag must not be set if
        EGL_NATIVE_BUFFER_USAGE_TEXTURE_BIT_ANDROID is set.

        EGL_NATIVE_BUFFER_USAGE_TEXTURE_BIT_ANDROID: The buffer will be used to
        create a texture. This flag must not be set if
        EGL_NATIVE_BUFFER_USAGE_RENDERBUFFER_BIT_ANDROID is set.

    Errors

        If eglCreateNativeClientBufferANDROID fails, NULL will be returned, no
        memory will be allocated, and one of the following errors will be
        generated:

       * If the value of EGL_WIDTH or EGL_HEIGHT is not positive, the error
         EGL_BAD_PARAMETER is generated.

       * If the combination of the values of EGL_RED_SIZE, EGL_GREEN_SIZE,
         EGL_BLUE_SIZE, and EGL_ALPHA_SIZE is not a valid pixel format for the
         EGL implementation, the error EGL_BAD_PARAMETER is generated.

       * If the value of EGL_NATIVE_BUFFER_ANDROID is not a valid combination
         of gralloc usage flags for the EGL implementation, or is incompatible
         with the value of EGL_FORMAT, the error EGL_BAD_PARAMETER is
         Generated.

       * If both the EGL_NATIVE_BUFFER_USAGE_RENDERBUFFER_BIT_ANDROID and
         EGL_NATIVE_BUFFER_USAGE_TEXTURE_BIT_ANDROID are set in the value of
         EGL_NATIVE_BUFFER_USAGE_ANDROID, the error EGL_BAD_PARAMETER is
         Generated."

Issues

    1. Should this extension define what combinations of formats and usage flags
    EGL implementations are required to support?

    RESOLVED: Partially.

    The set of valid color combinations is implementation-specific and may
    depend on additional EGL extensions, but generally RGB565 and RGBA888 should
    be supported. The particular valid combinations for a given Android version
    and implementation should be documented by that version.

    2. Should there be an eglDestroyNativeClientBufferANDROID to destroy the
    client buffers created by this extension?

    RESOLVED: No.

    A destroy function would add several complications:

        a) ANativeWindowBuffer is a reference counted object, may be used
           outside of EGL.
        b) The same buffer may back multiple EGLImages, though this usage may
           result in undefined behavior.
        c) The interactions between the lifetimes of EGLImages and their
           EGLClientBuffers would become needlessly complex.

    Because ANativeWindowBuffer is a reference counted object, implementations
    of this extension should ensure the buffer has a lifetime at least as long
    as a generated EGLImage (via EGL_ANDROID_image_native_buffer). The simplest
    method is to increment the reference count of the buffer in
    eglCreateImagKHR, and then decrement it in eglDestroyImageKHR. This should
    ensure proper lifetime semantics.

Revision History

#2 (Craig Donner, April 15, 2016)
    - Set color formats and usage bits explicitly using additional attributes,
    and add value for new token EGL_NATIVE_BUFFER_USAGE_ANDROID.

#1 (Craig Donner, January 19, 2016)
    - Initial draft.
