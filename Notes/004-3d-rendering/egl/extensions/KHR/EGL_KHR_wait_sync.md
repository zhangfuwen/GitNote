# KHR_wait_sync

Name

    KHR_wait_sync

Name Strings

    EGL_KHR_wait_sync

Contributors

    Jon Leech
    Tom Cooksey
    Alon Or-bach

Contacts

    Jon Leech (jon 'at' alumni.caltech.edu)

Notice

    Copyright (c) 2012-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the Khronos Board of Promoters on October 26, 2012.

Version

    Version 7, March 12, 2014

Number

    EGL Extension #43

Dependencies

    EGL 1.1 is required.

    EGL_KHR_fence_sync is required.

    EGL_KHR_reusable_sync is affected by this extension.

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    This extension adds the ability to wait for signaling of sync objects
    in the server for a client API context, rather than in the application
    thread bound to that context. This form of wait does not necessarily
    block the application thread which issued the wait (unlike
    eglClientWaitSyncKHR), so the application may continue to issue commands
    to the client API context or perform other work in parallel, leading to
    increased performance. The best performance is likely to be achieved by
    implementations which can perform this new wait operation in GPU
    hardware, although this is not required.

New Types

    None

New Procedures and Functions

    EGLint eglWaitSyncKHR(
                  EGLDisplay dpy,
                  EGLSyncKHR sync,
                  EGLint flags)

New Tokens

    None

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Add to section 3.8.1 "Sync Objects" (as already modified
    by EGL_KHR_fence_sync and/or EGL_KHR_reusable_sync):

    Following the description of eglClientWaitSyncKHR, add:

   "The command

        EGLint eglWaitSyncKHR(EGLDisplay dpy,
                              EGLSyncKHR sync,
                              EGLint flags)

    is similar to eglClientWaitSyncKHR, but instead of blocking and not
    returning to the application until <sync> is signaled, eglWaitSyncKHR
    returns immediately. On success, EGL_TRUE is returned, and the server
    for the client API context [fn1] will block until <sync> is signaled
    [fn2].
       [fn1 - the server may choose to wait either in the CPU executing
        server-side code, or in the GPU hardware if it supports this
        operation.]
       [fn2 - eglWaitSyncKHR allows applications to continue to queue
        commands from the application in anticipation of the sync being
        signaled, potentially increasing parallelism between application,
        client API server code, and the GPU. The server only blocks
        execution of commands for the specific context on which
        eglWaitSyncKHR was issued; other contexts implemented by the same
        server are not affected.]

    <sync> has the same meaning as for eglClientWaitSyncKHR.

    <flags> must be 0.

    Errors
    ------
    eglWaitSyncKHR returns EGL_FALSE on failure and generates an error as
    specified below, but does not cause the server for the client API
    context to block.

      * If the current context for the currently bound client API does not
        support the client API extension indicating it can perform server
        waits, an EGL_BAD_MATCH error is generated.
      * If no context is current for the currently bound client API (i.e.,
        eglGetCurrentContext returns EGL_NO_CONTEXT), an EGL_BAD_MATCH error
        is generated.
      * If <dpy> does not match the EGLDisplay passed to eglCreateSyncKHR
        when <sync> was created, the behavior is undefined.
      * If <sync> is not a valid sync object for <dpy>, an EGL_BAD_PARAMETER
        error is generated.
      * If <flags> is not 0, an EGL_BAD_PARAMETER error is generated.

    Each client API which supports eglWaitSyncKHR indicates this support in
    the form of a client API extension. If the GL_OES_EGL_sync extension is
    supported by any version of OpenGL ES, a server wait may be performed
    when the currently bound API is OpenGL ES. If the VG_KHR_EGL_sync
    extension is supported by OpenVG, a server wait may be performed when
    the currently bound API is OpenVG."

    Add new subsubsection following eglWaitSyncKHR:

   "Multiple Waiters
    ----------------

    It is possible for the application thread calling a client API to be
    blocked on a sync object in a eglClientWaitSyncKHR command, the server
    for that client API context to be blocked as the result of a previous
    eglWaitSyncKHR command, and for additional eglWaitSyncKHR commands to be
    queued in the server, all for a single sync object. When the sync object
    is signaled in this situation, the client will be unblocked, the server
    will be unblocked, and all such queued eglWaitSyncKHR commands will
    continue immediately when they are reached.

    Sync objects may be waited on or signaled from multiple contexts of
    different client API types in multiple threads simultaneously, although
    some client APIs may not support eglWaitSyncKHR. This support is
    determined by client API-specific extensions."

    Additions to the EGL_KHR_reusable_sync extension, modifying the description
    of eglSignalSyncKHR to include both client and server syncs:

   "... If as a result of calling eglSignalSyncKHR the status of <sync>
    transitions from unsignaled to signaled, any eglClientWaitSyncKHR
        * or eglWaitSyncKHR *
    commands blocking on <sync> will unblock. ..."

    Additions to the EGL_KHR_reusable_sync extension, modifying the description
    of eglDestroySyncKHR to include both client and server syncs:

   "... If any eglClientWaitSyncKHR
        * or eglWaitSyncKHR *
    commands are blocking on <sync> when eglDestroySyncKHR is called, they
    will be woken up, as if <sync> were signaled."


    Additions to the EGL_KHR_fence_sync extension, modifying the description
    of eglCreateSyncKHR to include both client and server syncs:

   "... causing any eglClientWaitSyncKHR 
        * or eglWaitSyncKHR *
    commands (see below) blocking on <sync> to unblock ..."

    Additions to the EGL_KHR_fence_sync extension, modifying the description
    of eglDestroySyncKHR to include both client and server syncs:

   "... If any eglClientWaitSyncKHR
        * or eglWaitSyncKHR *
    commands are blocking on <sync> when eglDestroySyncKHR is called, <sync>
    is flagged for deletion and will be deleted when it is no longer
    associated with any fence command and is no longer blocking any
    eglClientWaitSyncKHR or eglWaitSyncKHR commands."


Issues

    1. Explain the key choices made in this extension.

    RESPONSE: This extension has been written to behave as similarly as
    possible to the glWaitSync functionality available in desktop OpenGL.
    Server waits are functionality which was only available in GL syncs
    until now.

    In the interest of maintaining similarity with OpenGL sync objects, this
    extension attempts to copy the glWaitSync functionality of OpenGL
    wherever possible (both functionally and stylistically), only making
    changes where needed to operate inside EGL (rather than a client API
    context) and match EGL naming conventions.

    2. Must all EGL client APIs support server waits?

    PROPOSED: Only if the client APIs also support fence syncs, which also
    interacts with the server for that client API. The same client API
    extensions defining fence sync support (GL_OES_EGL_sync and
    VG_KHR_EGL_sync) are used here to indicate server wait ability for those
    client APIs.

    Reusable syncs in EGL_KHR_reusable_sync do not have this dependency,
    because it is (at least in principle) possible for eglClientWaitSyncKHR
    to be performed entirely within the EGL implementation. However,
    eglWaitSyncKHR requires cooperation of the client API, whether fence
    syncs or reusable syncs are being waited upon.

    It would be possible to completely decouple fence syncs and the ability
    to do server waits, but this would require new client API extensions.

    3. What is the relationship between EGL sync objects and OpenGL / OpenGL
    ES sync objects?

    RESPONSE: There is no direct relationship. GL and ES servers may not
    even implement sync objects as defined by some versions of those APIs.
    However, EGL sync objects are intended to be functionally equivalent to
    GL sync objects, and the language describing them is drawn from the GL
    specifications. Implementations supporting both forms of sync object
    will probably use the same implementation internally.

    4. Should eglWaitSyncKHR take a timeout as a parameter as its equivalent
    in OpenGL / OpenGL ES and eglWaitClientKHR does?

    PROPOSED: No. A timeout is of limited use to a well-behaved application.
    If a timeout was added, the result of it expiring is likely to be a
    corrupted output. Adding a timeout would likely necessitate a way to
    query if the wait completed because the condition was signaled or
    because of a timeout. There doesn't seem to be an obvious, elegant API
    mechanism to do that. If a timeout is needed in the future, it can be
    added via an additional extension with a new entry-point.

    5. What happens if an application issues a server-side wait on a fence
    which never gets signaled?

    RESPONSE: Further rendering in the context which issued the server-side
    wait will not progress. Any additional behavior is undefined and is
    likely to be different depending on a particular implementation. Could
    be handled in the same way as an extremely long-running GLSL shader.

    6. Does this extension affect the destruction behavior?

    RESOLVED: No. The behavior of eglDestroySyncKHR is determined by the type
    of sync object, and is not affected by this extension.

Revision History

#7   (Alon Or-bach, March 12, 2014)
   - Clarified that eglDestroySyncKHR behavior is set in 
     EGL_KHR_fence_sync / EGL_KHR_reusable_sync and is not modified by this
     extension (bug 11458).
#6   (Jon Leech, January 24, 2013)
   - eglWaitSyncKHR causes a server wait in OpenGL ES when GL_OES_EGL_sync
     is supported, not a client wait as formerly specified (Bug 9493).
#5   (Jon Leech, October 31, 2012)
   - Change return type of eglWaitSyncKHR in spec body to EGLint to match
     New Functions section, and rearrange description of return type and
     errors section for clarity.
#4   (Tom Cooksey, August 16, 2012)
   - Removed timeout parameter and text relating to it. Add issue 4
     discussing timeout parameter. Add issue 5 explaining the behavior of
     waiting on a never-signaled fence. Minor corrections to use US English.
#3   (Jon Leech, June 26, 2012)
   - Fix typos (bug 9137).
#2   (Jon Leech, June 20, 2012)
   - Clarifications and language cleanup (Bug 9137). Some paragraph
     reflowing. Note that eglWaitSyncKHR only blocks the server for the
     specific context on which the wait was issued. Add issue 3 discussing
     relationship to GL/ES sync objects.
#1   (Jon Leech, June 6, 2012)
   - Initial draft branched from GL 4.x API spec language.
