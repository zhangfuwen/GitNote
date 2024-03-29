# KHR_cl_event2

Name

    KHR_cl_event2

Name Strings

    EGL_KHR_cl_event2

Contributors

    Jon Leech, Khronos
    Alon Or-bach, Samsung Electronics
    Tom Cooksey, ARM

Contact

    Jon Leech (jon 'at' alumni.caltech.edu)

IP Status

    No known claims.

Notice

    Copyright (c) 2010-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the EGL Working Group on December 4, 2013.

Version

    Version 5, December 4, 2013

Number

    EGL Extension #65

Dependencies

    EGL 1.4 and the EGL_KHR_fence_sync extension are required.

    This extension is written against the language added to EGL 1.2 by
    the EGL_KHR_fence_sync extension.

    An OpenCL implementation supporting sharing OpenCL event objects
    with EGL is required.

    Khronos recommends obsoleting and replacing implementations of
    EGL_KHR_cl_event with this extension as soon as possible.

Overview

    This extension allows creating an EGL sync object linked to an OpenCL
    event object, potentially improving efficiency of sharing images between
    the two APIs. The companion cl_khr_egl_event extension provides the
    complementary functionality of creating an OpenCL event object from an
    EGL sync object.

    This extension is functionally identical to EGL_KHR_cl_event, but is
    intended to replace that extension. It exists only to address an
    implementation issue on 64-bit platforms where passing OpenCL event
    handles in an EGLint attribute list value is impossible, because the
    implementations use a 32-bit type for EGLint.

    This extension also incorporates some required functionality from the
    EGL_KHR_fence_sync extension, similarly modified for 64-bit platforms.

New Types

    /*
     * EGLAttribKHR is a integer type used to pass arrays of attribute
     * name/value pairs which may include pointer and handle attribute
     * values.
     */
    #include <khrplatform.h>
    typedef intptr_t EGLAttribKHR;

    Event handles of type cl_event, defined in the OpenCL header files, may
    be included in the attribute list passed to eglCreateSync64KHR.

New Procedures and Functions

    EGLSyncKHR eglCreateSync64KHR(
                        EGLDisplay dpy,
                        EGLenum type,
                        const EGLAttribKHR *attrib_list);

New Tokens

    Accepted as attribute names in the <attrib_list> argument
    of eglCreateSync64KHR:

        EGL_CL_EVENT_HANDLE_KHR         0x309C

    Returned in <values> for eglGetSyncAttribKHR <attribute>
    EGL_SYNC_TYPE_KHR:

        EGL_SYNC_CL_EVENT_KHR           0x30FE

    Returned in <values> for eglGetSyncAttribKHR <attribute>
    EGL_SYNC_CONDITION_KHR:

        EGL_SYNC_CL_EVENT_COMPLETE_KHR  0x30FF

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Modify the language in section 3.8.1 (Sync Objects) starting at the
    sixth paragraph, describing commands to create sync objects:

    "The commands

        EGLSyncKHR eglCreateSync64KHR(
                        EGLDisplay dpy,
                        EGLenum type,
                        const EGLAttribKHR *attrib_list);

    and

        EGLSyncKHR eglCreateSyncKHR(
                        EGLDisplay dpy,
                        EGLenum type,
                        const EGLint *attrib_list);

    create a sync object ...

    ... When a fence sync object is created, eglCreateSyncKHR and
    eglCreateSync64KHR also insert a fence command into... "

    Add following the eigth paragraph (the paragraph beginning "<Fence sync
    objects> are created..."):

   "A <CL event sync object> reflects the status of a corresponding OpenCL
    event object to which the sync object is linked. This provides another
    method of coordinating sharing of images between EGL and OpenCL (see
    Chapter 9 of the OpenCL 1.0 Specification and the cl_khr_egl_sharing
    extension). Waiting on such a sync object is equivalent to waiting for
    completion of the linked CL event object.

    CL event sync objects may only be created using the command
    eglCreateSync64KHR, because they require an attribute which may not be
    representable in the attrib_list argument of eglCreateSyncKHR."

    Add following the description of fence sync objects (prior to the
    "Errors" section for eglCreateSyncKHR):

   "If <type> is EGL_SYNC_CL_EVENT_KHR, a CL event sync object is
    created. In this case <attrib_list> must contain the attribute
    EGL_CL_EVENT_HANDLE_KHR, set to a valid OpenCL event. Note that
    EGL_CL_EVENT_HANDLE_KHR is not a queriable property of a sync
    object. Attributes of the CL event sync objects are set as follows:

        Attribute Name          Initial Attribute Value(s)
        -------------           --------------------------
        EGL_SYNC_TYPE_KHR       EGL_SYNC_CL_EVENT_KHR
        EGL_SYNC_STATUS_KHR     Depends on status of <event>
        EGL_SYNC_CONDITION_KHR  EGL_SYNC_CL_EVENT_COMPLETE_KHR

    The status of such a sync object depends on <event>. When the status
    of <event> is CL_QUEUED, CL_SUBMITTED, or CL_RUNNING, the status of
    the linked sync object will be EGL_UNSIGNALED_KHR. When the status
    of <event> changes to CL_COMPLETE, the status of the linked sync
    object will become EGL_SIGNALED_KHR.

    Creating a linked sync object places a reference on the linked
    OpenCL event object. When the sync object is deleted, the reference
    will be removed from the event object.

    However, implementations are not required to validate the OpenCL
    event, and passing an invalid event handle in <attrib_list> may
    result in undefined behavior up to and including program
    termination."

    The command eglCreateSync64KHR must be used to create a CL event sync
    object[fn1].

    [fn1] If the implementation also supports the older EGL_KHR_cl_event
          extension, then eglCreateSyncKHR may also be used to create a CL
          event sync object. However, this use is not recommended because it
          is not portable to platforms where OpenCL event handles are larger
          than 32 bits."

    Modify the ninth and tenth paragraphs, starting "When the condition":

    "When the condition of the sync object is satisfied, the sync is
    signaled by the associated client API context, causing any
    eglClientWaitSyncKHR commands (see below) blocking on <sync> to unblock.

    The only condition supported for fence sync objects is
    EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR, which is satisfied by completion
    of the fence command corresponding to the sync object, and all preceding
    commands in the associated client API context's command stream. The sync
    object will not be signaled until all effects from these commands on the
    client API's internal and framebuffer state are fully realized. No other
    state is affected by execution of the fence command.

    Each client API which supports fence commands indicates this support
    in the form of a client API extension. If the GL_OES_EGL_sync
    extension is supported by OpenGL ES (either version 1.x or 2.0), a
    fence sync object may be created when the currently bound API is
    OpenGL ES. If the VG_KHR_EGL_sync extension is supported by OpenVG,
    a fence sync object may be created when the currently bound API is
    OpenVG.

    The only condition supported for CL event sync objects is
    EGL_SYNC_CL_EVENT_COMPLETE_KHR, which is satisfied when the status of
    the OpenCL event associated with the sync object changes to CL_COMPLETE."

    Add to the "Errors" section for eglCreateSyncKHR and eglCreateSync64KHR:

   "* If <type> is EGL_SYNC_CL_EVENT_KHR then

    ** If eglCreateSyncKHR was called, then EGL_NO_SYNC_KHR is returned and
       an EGL_BAD_ATTRIBUTE error is generated.

    ** If eglCreateSync64KHR was called and EGL_CL_EVENT_HANDLE_KHR is not
       specified in <attrib_list>, or its attribute value is not a valid
       OpenCL event handle returned by a call to clEnqueueReleaseGLObjects
       or clEnqueueReleaseEGLObjects, then EGL_NO_SYNC_KHR is returned and
       an EGL_BAD_ATTRIBUTE error is generated."

    Replace the EGL_SYNC_CONDITION_KHR row of table 3.cc with:

   "Attribute              Description                Supported Sync Objects
    -----------------      -----------------------    ----------------------
    EGL_SYNC_CONDITION_KHR Signaling condition        EGL_SYNC_FENCE_KHR or
                                                      EGL_SYNC_CL_EVENT_KHR

    Table 3.cc  Attributes Accepted by eglGetSyncAttribKHR Command"


    Replace the second paragraph describing eglDestroySync with:

   "If any eglClientWaitSyncKHR commands are blocking on <sync> when
    eglDestroySyncKHR is called, <sync> is flagged for deletion and will
    be deleted when the associated fence command or CL event object has
    completed, and <sync> is no longer blocking any eglClientWaitSyncKHR
    command. Otherwise, the sync object is destroyed immediately."

Sample Code

    None

Conformance Tests

    None yet

Issues

    Note that some issues from the EGL_KHR_cl_event and EGL_KHR_fence_sync
    extensions also apply to this extension, which incorporates
    functionality from both of those extensions while making it usable on a
    64-bit architecture. Issues specific to this extension are below.

    1) Why does this extension exist?

    The existence of this extension is an unfortunate necessity. Khronos did
    not define EGLint as a 64-bit type in the version of <khrplatform.h> we
    provided, assuming that vendors on those platforms would do so. By the
    time we discovered that not all vendors had done this, it was too late
    to fix, because ABI considerations made it impossible for those vendors
    to change to a 64-bit EGLint type. Our only option was to define new
    extensions and commands using a new attribute type, EGLAttribKHR, which
    is explicitly large enough to hold a pointer or handle.

    2) What is the relationship of this extension to EGL_KHR_cl_event?

    RESOLVED: The only functional difference is that the new
    eglCreateSync64KHR command must be used to create CL event sync objects.
    This is necessary because some 64-bit platforms define EGLint as a
    32-bit type, making it impossible to pass an arbitrary OpenCL event
    handle in the EGLint *attrib_list passed to eglCreateSyncKHR.

    3) How are pointer- and handle-sized attributes represented?

    RESOLVED: Using the new type EGLAttribKHR, which is explicitly defined
    as an integer type large enough to hold a pointer.

    EGLAttribKHR is defined as an alias of the ISO C intptr_t type, rather
    than using one of the explicitly-sized types from khrplatform.h.
    Requiring this means that khrplatform.h must make sure to include the
    appropriate header file (probably <stdint.h>) and that a C compiler
    supporting intptr_t must be used. In the past we were concerned about
    older C/C++ compilers, but this seems an acceptable choice in 2013.

    We could choose to use intptr_t as the base type of attribute lists,
    instead of the EGLAttribKHR alias. As Ian Romanick has pointed out
    passionately in ARB discussions, modern C compilers are required to
    support a well-defined set of scalar types. There is no requirement to
    use API-specific scalar types when explicitly defining a C API.

    However, there is some value in semantically tagging parameters with EGL
    types. Also, using 'intptr_t *attrib_list' would be cosmetically
    objectionable due to mixing EGL* and C native scalar types in EGL APIs.

    We probably want to wait until there's an EGL API compatibility break -
    a hypothetical "EGL 2.0" - before moving to native ISO C types in our
    interfaces.

    4) Why is the new fence sync creation function defined here, instead of
    in a separate EGL_KHR_fence_sync2 extension?

    RESOLVED: eglCreateSync64KHR is defined here because this is the only
    functionality requiring it, and we expect this extension to be a stopgap
    for 64-bit platforms until the time that EGL 1.5 is defined. The EGL 1.5
    core will presumably include only the EGLAttribKHR-based version of this
    command.

    If there are any new extensions using handle or pointer attributes in
    the meantime, they should copy the EGLAttribKHR and eglCreateSync64KHR
    language here as required. There is no harm in defining the same type or
    command in multiple extensions, so long as the definitions are
    compatible.

    5) Why is the new command called eglCreateSync64KHR?

    UNRESOLVED: For consistency with OpenGL, which has '64'-suffixed
    commands for representing 64-bit integers and arbitrary offsets into GPU
    memory. If we ever support EGL on 128-bit platforms this would be a
    silly naming convention, but that time is probably many decades away and
    by then EGL 1.5 should be defined and widely supported. The name
    eglCreateSync2KHR was originally suggested.

    6) Why is there no command for querying EGLAttribKHR attributes from
    sync objects?

    RESOLVED: Because the only sync attribute which requires the extra bits
    in an EGLAttribKHR type is EGL_CL_EVENT_HANDLE_KHR, which is not
    queryable. Sync attributes which are queryable will all fit into the
    EGLint returned by eglGetSyncAttribKHR.

    NOTE: It's unfortunate that this name is used, since it uses the
    "AttribKHR" name for command returning EGLints. In EGL 1.5 we should use
    a different name for the query.

    7) Does this extension replace EGL_KHR_fence_sync and EGL_KHR_cl_event?

    RESOLVED: It does not replace EGL_KHR_fence_sync, but extends it to
    support creation of a new type of sync object, the CL event sync object.

    RESOLVED: It is intended to replace EGL_KHR_cl_event; this extension
    must be used for OpenCL interop on 64-bit platforms, and we hope all
    vendors will implement it even on 32-bit platforms, for maximum code
    portability.

Revision History

    Version 5, 20130/12/04 (Jon Leech) - minor cleanup for public release.

    Version 4, 20130/10/16 (Jon Leech) - add Dependencies and Overview text
    noting that this extension obsoletes and should replace
    EGL_KHR_cl_event.

    Version 3, 20130/10/15 (Jon Leech) - change type of EGLAttribKHR from
    uintptr to intptr (Bug 11027).

    Version 2, 20130/10/12 (Jon Leech) - merge EGL_KHR_fence_sync2 with this
    extension, change the naming scheme, define EGLAttribKHR as uintptr_t,
    and add a new issues list.

    Version 1, 2010/10/02 (Tom Cooksey) - initial version based on
    EGL_KHR_cl_event and adding 64-bit EGLAttrKHR variants.
