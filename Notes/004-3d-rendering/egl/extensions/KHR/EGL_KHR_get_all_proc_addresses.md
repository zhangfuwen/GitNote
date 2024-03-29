# KHR_get_all_proc_addresses

Names

    KHR_get_all_proc_addresses
    KHR_client_get_all_proc_addresses

Name Strings

    EGL_KHR_get_all_proc_addresses
    EGL_KHR_client_get_all_proc_addresses

Contributors

    Jon Leech
    Marcus Lorentzon
    Robert Palmer
    Acorn Pooley
    Greg Prisament
    Chad Versace

Contacts

    James Jones, NVIDIA  (jajones 'at' nvidia.com)

Notice

    Copyright (c) 2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete. Approved by the EGL Working Group on April 3, 2013.
    Ratified by the Khronos Board of Promoters on July 19, 2013.

Version

    Version 3 - July 31, 2013

Number

    EGL Extension #61

Extension Types

    EGL_KHR_get_all_proc_addresses is an EGL display extension
    EGL_KHR_client_get_all_proc_addresses is an EGL client extension

Dependencies

    EGL 1.2 is required.

    This extension is written based on the wording of the EGL 1.4
    specification.

    Interacts with EGL_EXT_client_extensions.

Overview

    eglGetProcAddress is currently defined to not support the querying
    of non-extension EGL or client API functions.  Non-extension
    functions are expected to be exposed as library symbols that can
    be resolved statically at link time, or dynamically at run time
    using OS-specific runtime linking mechanisms.

    With the addition of OpenGL and OpenGL ES 3 support to EGL, the
    definition of a non-extension function becomes less clear.  It is
    common for one OpenGL library to implement many versions of
    OpenGL.  The suggested library name for OpenGL ES 3 is the same as
    that of OpenGL ES 2.  If OpenGL ES 3 applications linked
    statically to OpenGL ES 3 functions are run on a system with only
    OpenGL ES 2 support, they may fail to load.  Similar problems
    would be encountered by an application linking statically to
    various OpenGL functions.

    To avoid requiring applications to fall back to OS-specific
    dynamic linking mechanisms, this extension drops the requirement
    that eglGetProcAddress return only non-extension functions.  If
    the extension string is present, applications can query all EGL
    and client API functions using eglGetProcAddress.

    To allow users to query this extension before initializing a display, and
    to also allow vendors to ship this extension without
    EGL_EXT_client_extensions, two names are assigned to this extension: one
    a display extension and the other a client extension.  Identical
    functionality is exposed by each name, but users query each name using
    different methods.  Users query EGL_KHR_get_all_proc_addresses in the
    usual way; that is, by calling eglQueryString(dpy, EGL_EXTENSIONS) on an
    initialized display.  To query EGL_KHR_client_get_all_proc_addresses,
    users must use a different method which is described below in the section
    concerning EGL_EXT_client_extensions.

New Types

    None

New functions

    None

New Tokens

    None

Rename section "3.10 Obtaining Extension Function Pointers" to "3.10
Obtaining Function Pointers", and replace its content with the
following:

   "The client API and EGL extensions and versions which are available to a
    client may vary at runtime, depending on factors such as the rendering
    path being used (hardware or software), resources available to the
    implementation, or updated device drivers. Therefore, the address of
    functions may be queried at runtime. The function

        void (*eglGetProcAddress(const char *procname))(void);

    returns the address of the function named by <procName>. <procName> must
    be a NULL-terminated string. The pointer returned should be cast to a
    function pointer matching the function's definition in the corresponding
    API or extension specification. A return value of NULL indicates that
    the specified function does not exist for the implementation.

    A non-NULL return value for eglGetProcAddress does not guarantee that a
    function is actually supported at runtime. The client must also make a
    corresponding query, such as glGetString(GL_EXTENSIONS) for OpenGL and
    OpenGL ES extensions; vgGetString(VG_EXTENSIONS) for OpenVG extensions;
    eglQueryString(dpy, EGL_EXTENSIONS) for EGL extensions; or query the
    corresponding API's version for non-extension functions, to determine if
    a function is supported by a particular client API context or display.

    Client API function pointers returned by eglGetProcAddress are
    independent of the display and the currently bound client API context,
    and may be used by any client API context which supports the function.

    eglGetProcAddress may be queried for all EGL and client API extension
    and non-extension functions supported by the implementation (whether
    those functions are supported by the current client API context or not).

    For functions that are queryable with eglGetProcAddress, implementations
    may also choose to export those functions statically from the object
    libraries implementing them. However, portable clients cannot rely on
    this behavior."

Interactions with EGL_EXT_client_extensions

    The EGL specification describes the behavior of eglGetProcAddress as
    independent of any display.  Therefore, this extension's functionality
    falls under the classification 'client extension' rather than 'display
    extension'. Accordingly, users may wish to query this extension before
    initializing a display.

    If the EGL_EXT_client_extensions is supported, then users can query this
    extension by checking for the name EGL_KHR_client_get_all_proc_addresses
    in the extension string of EGL_NO_DISPLAY.

    The EGL implementation must expose the name
    EGL_KHR_client_get_all_proc_addresses if and only if it exposes
    EGL_KHR_get_all_proc_addresses and supports EGL_EXT_client_extensions.
    This requirement eliminates the problematic situation where, if an EGL
    implementation exposed only one name, then an EGL client would fail to
    detect the extension if it queried only the other name.

    Despite having two names assigned to this extension, the restrictions
    described in EGL_EXT_client_extensions still apply. As
    EGL_KHR_client_get_all_proc_addresses is defined as a client extension,
    its name can appear only in the extension string of EGL_NO_DISPLAY and
    not in the extension string of any valid display. The converse applies
    to EGL_KHR_get_all_proc_addresses, as it is defined as a display
    extension.

Issues

    1.  What should this spec be called?

        PROPOSED: KHR_get_all_proc_addresses

    2.  Should this extension be classified as a client extension, as defined
        by EGL_EXT_client_extensions?

        DISCUSSION: Yes and no.

        Yes, because this extension exposes functionality that is solely
        a property of the EGL library itself, independent of any display.
        Such functionality falls under the classification of 'client
        extension'.

        No, because classifying it as a client extension would create
        a dependency on EGL_EXT_client_extensions, and there exists no
        precedent for a KHR extension that depends on an EXT extension.

        RESOLUTION: Expose this extension under two names, one a client
        extension and the other a display extension.

Revision History

    #3 (July 31, 2013) Chad Versace
        - Assign additional name, EGL_KHR_client_get_all_proc_addresses.
        - Add section "Extension Types", section "Interactions with
          EGL_EXT_client_extensions", and issue #2.

    #2  (March 6, 2013) Jon Leech
        - Bring into sync with latest EGL 1.4 spec update and simplify
          language describing which functions may be queried. Minor
          formatting changes for greater consistency with other KHR
          extension specs.

    #1  (February 4, 2013) James Jones
        - Initial draft
