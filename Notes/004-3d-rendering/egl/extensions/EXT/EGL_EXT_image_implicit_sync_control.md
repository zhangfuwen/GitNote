# EXT_image_implicit_sync_control

Name

    EXT_image_implicit_sync_control

Name Strings

    EGL_EXT_image_implicit_sync_control

Contributors

    Daniel Stone, Collabora Ltd.

Contacts

    Daniel Stone (daniels 'at' collabora 'dot' com)

Status

    Complete

Version

    Version 2, March 16, 2020

Number

    EGL Extension #120

Dependencies

    EGL 1.2 is required.

    EGL_KHR_image_base and EGL_EXT_image_dma_buf_import are required.

    The EGL implementation must be running on a Linux kernel supporting implicit
    synchronization, as the usage is defined in the
    EGL_ARM_implicit_external_sync extension, but does not require that extension.

    This extension is written against the wording of the EGL 1.2 Specification.

Overview

    This extension allows a client to selectively use implicit or explicit
    synchronization mechanisms when addressing externally-imported EGLImages.
    A new token is added to EGLImage creation which allows the client to select
    whether a platform's implicit synchronization will be in use for a buffer
    imported into EGLImage.

    Heterogeneous systems (supporting multiple APIs, mixed legacy/updated
    clients, etc) already supporting implicit synchronization, may not be able
    to change to explict synchronization in a single switch. This extension
    allows synchronization to be controlled on a per-buffer basis, so explicit
    synchronization can be enabled for a complete pipeline which supports it,
    or implicit synchronization used otherwise.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute in the <attrib_list> parameter of
    eglCreateImageKHR:

        EGL_IMPORT_SYNC_TYPE_EXT           0x3470

    Accepted as the value for the EGL_IMPORT_SYNC_TYPE_EXT attribute:

        EGL_IMPORT_IMPLICIT_SYNC_EXT       0x3471
        EGL_IMPORT_EXPLICIT_SYNC_EXT       0x3472

New Types

    None.

Additions to Chapter 2 of the EGL 1.2 Specification (EGL Operation)

    Add to section 2.5.1 "EGLImage Specification" (as defined by the
    EGL_KHR_image_base specification), in the description of
    eglCreateImageKHR:

    Add the following to Table bbb (Legal attributes for eglCreateImageKHR
    <attr_list> parameter), Section 2.5.1 (EGLImage Specification)

      +-----------------------------+-------------------------+---------------------------+---------------+
      | Attribute                   | Description             | Valid <target>s           | Default Value |
      +-----------------------------+-------------------------+---------------------------+---------------+
      | EGL_IMPORT_SYNC_TYPE_EXT    | The type of             | EGL_LINUX_DMA_BUF_EXT     | EGL_IMPORT_   |
      |                             | synchronization to      |                           | IMPLICT_SYNC_ |
      |                             | apply to previously     |                           | EXT           |
      |                             | submitted rendering on  |                           |               |
      |                             | the platform buffer     |                           |               |
      +-----------------------------+-------------------------+---------------------------+---------------+
      Table bbb. Legal attributes for eglCreateImageKHR <attrib_list> parameter

    ...


    Add to section 2.5.1 "EGLImage Specification" (as defined by the
    EGL_KHR_image_base specification), in the description of
    eglCreateImageKHR:

    The behaviour of the imported buffer with regard to commands previously
    submitted (including via other APIs and from other clients) is controlled
    by the EGL_IMPORT_SYNC_TYPE_EXT attribute. If the default value of
    implicit synchronization is used, the platform may synchronize any access
    to the imported buffer, against accesses previously made (including by
    other clients or APIs) to that same buffer. If explicit synchronization
    is specified, the platform will not synchronize access to that buffer
    against other accesses; the client must use another synchronization
    mechanism if it wishes to order its accesses with respect to others.

    Add to the list of error conditions for eglCreateImageKHR:

       * If <attrib_list> contains the EGL_IMPORT_SYNC_TYPE_EXT name, but the
         value is not one of EGL_IMPORT_IMPLICIT_SYNC_EXT or
         EGL_IMPORT_EXPLICIT_SYNC_EXT, EGL_BAD_ATTRIBUTE is generated.


Revision History

#1 (Daniel Stone, May 15, 2017)
   - Initial revision.

#2 (Eric Engestrom, March 16, 2020)
   - Change "bad attribute value" error from EGL_BAD_PARAMETER to
     EGL_BAD_ATTRIBUTE to follow the EGL convention.
