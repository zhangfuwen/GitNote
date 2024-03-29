# ARM_implicit_external_sync

Name

    ARM_implicit_external_sync

Name Strings

    EGL_ARM_implicit_external_sync

Contributors

    David Garbett
    Ray Smith

Contacts

    David Garbett, ARM Ltd. (david 'dot' garbett 'at' arm 'dot' com)

Status

    Draft

Version

    Version 1, September 8, 2014

Number

    EGL Extension #103

Dependencies

    Requires EGL 1.1.

    This extension is written against the wording of the EGL 1.2 Specification.

    EGL_KHR_fence_sync is required.

Overview

    This extension extends the "fence sync objects" defined in
    EGL_KHR_fence_sync. It allows the condition that triggers the associated
    fence command in the client API command stream to be explicitly specified on
    fence object creation. It introduces a new condition that can be used to
    ensure ordering between operations on buffers that may be accessed
    externally to the client API, when those operations use an implicit
    synchronization mechanism. Such a fence object will be signaled when all
    prior commands affecting such buffers are guaranteed to be executed before
    such external commands.

    Applications have limited control over when a native buffer is read or
    written by the GPU when imported as an EGLImageKHR or via
    eglCreatePixmapSurface, which is controlled by the EGL and client API
    implementations.  While eglWaitClient or a client call such as glFinish
    could be called, this forces all rendering to complete, which can result in
    CPU/GPU serialization. Note this isn't an issue for window surfaces, where
    eglSwapBuffers ensures the rendering occurs in the correct order for the
    platform.

    Some platforms have an implicit synchronization mechanism associated with
    native resources, such as buffers. This means that accesses to the buffer
    have an implicit ordering imposed on them, without involvement from the
    application. However, this requires that an application that has imported
    an affected buffer into EGL has a mechanism to flush any drawing operations
    in flight such that they are waiting on the synchronization mechanism.
    Otherwise the application cannot guarantee that subsequent operations (such
    as displaying a rendered buffer) will occur after the commands performed by
    the client API (such as rendering the buffer).

    The mechanism to wait for the synchronization mechanism should not require
    the application to wait for all rendering to complete, so that it can
    continue preparing further commands asynchronously to the queued commands.
    This extension provides this functionality using the new condition type for
    fence sync objects, so the application only waits for the external
    synchronization.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as a value of the EGL_SYNC_CONDITION_KHR attribute passed in the
    <attrib_list> list to eglCreateSyncKHR when <type> is EGL_FENCE_SYNC_KHR,
    and can populate <*value> when eglGetSyncAttribKHR is called with
    <attribute> set to EGL_SYNC_CONDITION_KHR:

    EGL_SYNC_PRIOR_COMMANDS_IMPLICIT_EXTERNAL_ARM  0x328A

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add the following after the fifth paragraph of Section 3.8.1 (Sync Objects),
    added by KHR_fence_sync:

    "Typically client APIs are considered to execute commands in a linear queue,
    where a prior command is executed and completes before a later command is
    started. By default fence sync objects adhere to this model - a fence is
    signaled once prior commands have completed. However on some platforms a
    command in a client API may transition through multiple states before it
    completes, which may impact other components of the system. Therefore the
    condition that all prior commands must meet before the fence is triggered is
    configurable."

    Replace the sixth paragraph of Section 3.8.1 (Sync Objects), added by
    KHR_fence_sync:

    "If, <type> is EGL_SYNC_FENCE_KHR, a fence sync object is created. In this
    case <attrib_list> can be NULL or empty, or can specify the
    EGL_SYNC_CONDITION_KHR attribute. Attributes of the fence sync object have
    the following default values:"

    Replace the eighth paragraph of Section 3.8.1 (Sync Objects), added by
    KHR_fence_sync:

    "When the condition of the sync object is satisfied by the fence command,
    the sync is signaled by the associated client API context, causing any
    eglClientWaitSyncKHR commands (see below) blocking on <sync> to unblock. The
    condition is specified by the EGL_SYNC_CONDITION_KHR attribute passed to
    eglCreateSyncKHR.

    If the condition is specified as EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR, the
    fence sync object is satisfied by completion of the fence command
    corresponding to the sync object, and all preceding commands in the
    associated client API context's command stream. The sync object will not be
    signaled until all effects from these commands on the client API's internal
    and framebuffer state are fully realized. No other state is affected by
    execution of the fence command.

    If the condition is specified as
    EGL_SYNC_PRIOR_COMMANDS_IMPLICIT_EXTERNAL_ARM, the fence sync object is
    satisfied by the completion of the fence command corresponding to the sync
    object, and the <submission> of all preceding commands in the associated
    client API context's command stream. <Submission> defines the point in time
    when a command has been queued on any implicit synchronization mechanisms
    present on the platform which apply to any of the resources used by the
    command. This enforces an ordering, as defined by the synchronization
    mechanism, between the command and any other operations that also respect
    the synchronization mechanism(s)."

    Replace the second entry in the list of eglCreateSyncKHR errors in Section
    3.8.1 (Sync Objects), added by KHR_fence_sync:

    " * If <type> is EGL_SYNC_FENCE_KHR and <attrib_list> contains an attribute
        other than EGL_SYNC_CONDITION_KHR, EGL_NO_SYNC_KHR is returned and an
        EGL_BAD_ATTRIBUTE error is generated.
      * If <type> is EGL_SYNC_FENCE_KHR and the value specified for
        EGL_SYNC_CONDITION_KHR is not EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR or
        EGL_SYNC_PRIOR_COMMANDS_SUBMITTED_ARM, EGL_NO_SYNC_KHR is returned and
        an EGL_BAD_ATTRIBUTE error is generated."

Issues

    1. Could glFlush guarantee commands are submitted, making this extension
    unnecessary?

    RESOLVED: The Open GL ES 3.1 specification defines glFlush() as causing "all
    previously issued GL commands to complete in finite time". There is no
    requirement for the execution of commands to reach any specific point before
    it returns - a valid implementation of glFlush() could spawn a new thread
    that sleeps for a minute before submitting the pending commands.  While an
    implementation could decide to ensure all commands are submitted within
    glFlush(), it could not be assumed to be the case across all
    implementations.

    In addition, there may be scenarios when submitting commands within
    glFlush() is harmful. Waiting for command submission may have a performance
    impact on some implementations that perform processing of commands
    asynchronously. In addition such a change may restrict what is possible in
    the future. For example if user events were introduced into OpenGL ES they
    have the potential of introducing deadlocks if submission in glFlush() is
    guaranteed.

    2. Should a new entry point be defined that flushes commands synchronously,
    instead of the new fence type as defined by this extension?

    RESOLVED: While a synchronous "flush and submit" entrypoint would meet the
    requirements for this extension, there may be a small benefit in enabling
    the application to continue processing between flushing and waiting for
    submission. In addition, the semantics of the existing EGL_KHR_fence_sync
    extension closely match what is required for this extension, so defining
    the new functionality in terms of fences may enable simpler implementations.

    3. Should OpenGL ES 3 glFenceSync be extended in preference to
    eglCreateSyncKHR?

    RESOLVED: Some platforms are yet to move to a OpenGL ES 3 implementation, or
    may be unwilling to expose OpenGL ES 3 entrypoints to applications. As
    EGL_KHR_fence_sync is older than OpenGL ES 3, and is comparatively a small
    change, it has a better chance of adoption in a platform.

    In addition this extension is based on the idea that there are
    platform-specific ways to interact with the client API command stream. As
    this is platform-specific, and does not fit with the existing model
    typically used by client APIs (such as Open GL ES) it is better placed in
    EGL.

    Finally extending EGL has the advantage that the extension applies to all
    client APIs.

    4. Should a new <type> parameter be defined, instead of extending the
    EGL_FENCE_SYNC_KHR fence sync objects defined by EGL_KHR_fence_sync?

    RESOLVED: Whether the new functionality is defined as an extension to the
    existing fence sync objects, or whether they are defined as a new type of
    sync object, we must acknowledge that the model of a client API processing
    commands serially (with prior commands completing before later commands are
    executed) is too simplistic for some platforms.

    Extending the existing fence sync objects allows us to use the existing
    concept of conditions that trigger the fences. It also allows the maximum
    amount of reuse of existing functionality, potentially simplifying the
    implementation and the use of the extension by applications.

Revision History
#1   (David Garbett, September 8, 2014)
   - Initial draft.
