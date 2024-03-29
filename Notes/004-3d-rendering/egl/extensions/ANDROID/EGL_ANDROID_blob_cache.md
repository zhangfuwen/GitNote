# ANDROID_blob_cache

Name

    ANDROID_blob_cache

Name Strings

    EGL_ANDROID_blob_cache

Contributors

    Jamie Gennis

Contact

    Jamie Gennis, Google Inc. (jgennis 'at' google.com)

Status

    Complete

Version

    Version 3, December 13, 2012

Number

    EGL Extension #48

Dependencies

    Requires EGL 1.0

    This extension is written against the wording of the EGL 1.4 Specification

Overview

    Shader compilation and optimization has been a troublesome aspect of OpenGL
    programming for a long time.  It can consume seconds of CPU cycles during
    application start-up.  Additionally, state-based re-compiles done
    internally by the drivers add an unpredictable element to application
    performance tuning, often leading to occasional pauses in otherwise smooth
    animations.

    This extension provides a mechanism through which client API
    implementations may cache shader binaries after they are compiled.  It may
    then retrieve those cached shaders during subsequent executions of the same
    program.  The management of the cache is handled by the application (or
    middleware), allowing it to be tuned to a particular platform or
    environment.

    While the focus of this extension is on providing a persistent cache for
    shader binaries, it may also be useful for caching other data.  This is
    perfectly acceptable, but the guarantees provided (or lack thereof) were
    designed around the shader use case.

    Note that although this extension is written as if the application
    implements the caching functionality, on the Android OS it is implemented
    as part of the Android EGL module.  This extension is not exposed to
    applications on Android, but will be used automatically in every
    application that uses EGL if it is supported by the underlying
    device-specific EGL implementation.

New Types

    /*
     * EGLsizeiANDROID is a signed integer type for representing the size of a
     * memory buffer.
     */
    #include <khrplatform.h>
    typedef khronos_ssize_t EGLsizeiANDROID;

    /*
     * EGLSetBlobFunc is a pointer to an application-provided function that a
     * client API implementation may use to insert a key/value pair into the
     * cache.
     */
    typedef void (*EGLSetBlobFuncANDROID) (const void* key,
        EGLsizeiANDROID keySize, const void* value, EGLsizeiANDROID valueSize)

    /*
     * EGLGetBlobFunc is a pointer to an application-provided function that a
     * client API implementation may use to retrieve a cached value from the
     * cache.
     */
    typedef EGLsizeiANDROID (*EGLGetBlobFuncANDROID) (const void* key,
        EGLsizeiANDROID keySize, void* value, EGLsizeiANDROID valueSize)

New Procedures and Functions

    void eglSetBlobCacheFuncsANDROID(EGLDisplay dpy,
                                     EGLSetBlobFuncANDROID set,
                                     EGLGetBlobFuncANDROID get);

New Tokens

    None.

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Add a new subsection after Section 3.8, page 50
    (Synchronization Primitives)

    "3.9 Persistent Caching

    In order to facilitate persistent caching of internal client API state that
    is slow to compute or collect, the application may specify callback
    function pointers through which the client APIs can request data be cached
    and retrieved.  The command

        void eglSetBlobCacheFuncsANDROID(EGLDisplay dpy,
            EGLSetBlobFuncANDROID set, EGLGetBlobFuncANDROID get);

    sets the callback function pointers that client APIs associated with
    display <dpy> can use to interact with caching functionality provided by
    the application.  <set> points to a function that inserts a new value into
    the cache and associates it with the given key.  <get> points to a function
    that retrieves from the cache the value associated with a given key.  The
    semantics of these callback functions are described in Section 3.9.1 (Cache
    Operations).

    Cache functions may only be specified once during the lifetime of an
    EGLDisplay.  The <set> and <get> functions may be called at any time and
    from any thread from the time at which eglSetBlobCacheFuncsANDROID is
    called until the time that the last resource associated with <dpy> is
    deleted and <dpy> itself is terminated.  Concurrent calls to these
    functions from different threads is also allowed.

    If eglSetBlobCacheFuncsANDROID generates an error then all client APIs must
    behave as though eglSetBlobCacheFuncsANDROID was not called for the display
    <dpy>.  If <set> or <get> is NULL then an EGL_BAD_PARAMETER error is
    generated.  If a successful eglSetBlobCacheFuncsANDROID call was already
    made for <dpy> and the display has not since been terminated then an
    EGL_BAD_PARAMETER error is generated.

    3.9.1 Cache Operations

    To insert a new binary value into the cache and associate it with a given
    key, a client API implementation can call the application-provided callback
    function

        void (*set) (const void* key, EGLsizeiANDROID keySize,
            const void* value, EGLsizeiANDROID valueSize)

    <key> and <value> are pointers to the beginning of the key and value,
    respectively, that are to be inserted.  <keySize> and <valueSize> specify
    the size in bytes of the data pointed to by <key> and <value>,
    respectively.

    No guarantees are made as to whether a given key/value pair is present in
    the cache after the set call.  If a different value has been associated
    with the given key in the past then it is undefined which value, if any, is
    associated with the key after the set call.  Note that while there are no
    guarantees, the cache implementation should attempt to cache the most
    recently set value for a given key.

    To retrieve the binary value associated with a given key from the cache, a
    client API implementation can call the application-provided callback
    function

        EGLsizeiANDROID (*get) (const void* key, EGLsizeiANDROID keySize,
            void* value, EGLsizeiANDROID valueSize)

    <key> is a pointer to the beginning of the key.  <keySize> specifies the
    size in bytes of the binary key pointed to by <key>.  If the cache contains
    a value associated with the given key then the size of that binary value in
    bytes is returned.  Otherwise 0 is returned.

    If the cache contains a value for the given key and its size in bytes is
    less than or equal to <valueSize> then the value is written to the memory
    pointed to by <value>.  Otherwise nothing is written to the memory pointed
    to by <value>.

Issues

    1. How should errors be handled in the callback functions?

    RESOLVED: No guarantees are made about the presence of values in the cache,
    so there should not be a need to return error information to the client API
    implementation.  The cache implementation can simply drop a value if it
    encounters an error during the 'set' callback.  Similarly, it can simply
    return 0 if it encouters an error in a 'get' callback.

    2. When a client API driver gets updated, that may need to invalidate
    previously cached entries.  How can the driver handle this situation?

    RESPONSE: There are a number of ways the driver can handle this situation.
    The recommended way is to include the driver version in all cache keys.
    That way each driver version will use a set of cache keys that are unique
    to that version, and conflicts should never occur.  Updating the driver
    could then leave a number of values in the cache that will never be
    requested again.  If needed, the cache implementation can handle those
    values in some way, but the driver does not need to take any special
    action.

    3. How much data can be stored in the cache?

    RESPONSE: This is entirely dependent upon the cache implementation.
    Presumably it will be tuned to store enough data to be useful, but not
    enough to become problematic. :)

Revision History

#3 (Jon Leech, December 13, 2012)
    - Fix typo in New Functions section & assign extension #.

#2 (Jamie Gennis, April 25, 2011)
    - Swapped the order of the size and pointer arguments to the get and set
      functions.

#1 (Jamie Gennis, April 22, 2011)
    - Initial draft.
