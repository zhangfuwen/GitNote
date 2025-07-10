# EXT_sync_reuse

Name

    EXT_sync_reuse

Name Strings

    EGL_EXT_sync_reuse

Contributors

    Daniel Kartch
    Jeff Vigil
    Ray Smith

Contacts

    Daniel Kartch, NVIDIA Corporation (dkartch 'at' nvidia.com)

Status

    Complete

Version

    Version 4, May 16, 2018

Number

    EGL Extension #128

Extension type

    EGL display extension

Dependencies

    Requires EGL 1.5 or EGL 1.4 with EGL_KHR_fence_sync

    Interacts with EGL_KHR_reusable_sync 
    Interacts with EGL_ANDROID_native_fence_sync
    Interacts with EGL_NV_cuda_event

    This extension is written against the wording of the EGL 1.5
    Specification.

Overview

    The original EGLSync extensions separated sync objects into two
    types: fence sync objects signaled by one time events in an
    API command pipeline; and reusable sync objects signaled by commands
    which can be issued again and again. However, this conflates
    reusability of the event triggering a sync object with the EGLSync
    object itself.

    Although the event associated with a fence sync object will only
    occur once, there is no reason that it can't be replaced with a new
    event. Doing so would avoid unnecessary allocation and free
    operations in an application that repeatedly waits for events. With
    the current interfaces, such applications must constantly create and
    destroy new EGLSync objects.

    This extension allows all sync objects to be reusable. When a sync
    object is in the signaled state, it can be reset back to an
    unsignaled state, regenerating or reevaluating the events that
    trigger them. For fence sync objects, this means generating a new
    fence in the current API. For OpenCL event sync objects, this means
    waiting for a new OpenCL event handle. This mechanism also allows
    sync objects to be created in the signaled state with no associated
    fence/event, and have one applied later. Thus all EGLSyncs required
    by an application can be allocated up front, before any rendering
    operations have begun.

New Types

    None

New Tokens

    None

New Procedures and Functions

    EGLBoolean eglUnsignalSyncEXT(
                    EGLDisplay dpy,
                    EGLSync sync,
                    const EGLAttrib *attrib_list);

Replace text of subsections of 3.8.1 through 3.8.1.2 of EGL 1.5
Specification. Existing tables are preserved.

    3.8.1 Sync Objects

    In addition to the aforementioned synchronization functions, which
    provide an efficient means of serializing client and native API
    operations within a thread, <sync objects> are provided to enable
    synchronization of client API operations between threads and/or
    between API contexts. Sync objects may be tested or waited upon by
    application threads.

    Sync objects have a status with two possible states: <signaled> and
    <unsignaled>, and may initially be in either state. EGL may be asked
    to wait for a sync object to become signaled, or a sync object