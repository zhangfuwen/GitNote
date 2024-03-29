# KHR_cl_event

Name

    KHR_cl_event

Name Strings

    EGL_KHR_cl_event

Contributors

    Jon Leech, Khronos
    Alon Or-bach, Samsung Electronics

Contact

    Jon Leech (jon 'at' alumni.caltech.edu)

IP Status

    No known claims.

Notice

    Copyright (c) 2010-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    This extension is obsolete and has been replaced by EGL_KHR_cl_event2.
    Khronos recommends implementers who support this extension also
    implement cl_event2, and begin transitioning developers to using that
    extension. See issue 17 for the reason.

    Complete. Approved by the EGL Working Group on 2013/05/15.
    Approved by the Khronos Board of Promoters on 2013/07/19.

Version

    Version 10, December 4, 2013

Number

    EGL Extension #60

Dependencies

    EGL 1.4 and the EGL_KHR_fence_sync extension are required.

    This extension is written against the language added to EGL 1.2 by
    the EGL_KHR_fence_sync extension.

    An OpenCL implementation supporting sharing OpenCL event objects
    with EGL is required.

Overview

    This extension allows creating an EGL fence sync object linked to an
    OpenCL event object, potentially improving efficiency of sharing
    images between the two APIs. The companion cl_khr_egl_event
    extension provides the complementary functionality of creating an
    OpenCL event object from an EGL fence sync object.

New Types

    None. However, event handles of type cl_event, defined in the OpenCL
    header files, may be included in the attribute list passed to
    eglCreateSyncKHR.

New Procedures and Functions

    None

New Tokens

    Accepted as attribute names in the <attrib_list> argument
    of eglCreateSyncKHR:

        EGL_CL_EVENT_HANDLE_KHR         0x309C

    Returned in <values> for eglGetSyncAttribKHR <attribute>
    EGL_SYNC_TYPE_KHR:

        EGL_SYNC_CL_EVENT_KHR           0x30FE

    Returned in <values> for eglGetSyncAttribKHR <attribute>
    EGL_SYNC_CONDITION_KHR:

        EGL_SYNC_CL_EVENT_COMPLETE_KHR  0x30FF

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add following the description of fence sync objects in section 3.8.1
    (e.g. following the paragraph beginning "<Fence sync objects> are
    created..."

   "A <CL event sync object> reflects the status of a corresponding
    OpenCL event object to which the sync object is linked. This
    provides another method of coordinating sharing of images between
    EGL and OpenCL (see Chapter 9 of the OpenCL 1.0 Specification and
    the cl_khr_egl_sharing extension). Waiting on such a sync object is
    equivalent to waiting for completion of the linked CL event object."

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

    Add to the "Errors" section for eglCreateSyncKHR:

   "* If <type> is EGL_SYNC_CL_EVENT_KHR then

    ** If EGL_CL_EVENT_HANDLE_KHR is not specified in <attrib_list>
       or is not a valid OpenCL event handle returned by a call to
       clEnqueueReleaseGLObjects or clEnqueueReleaseEGLObjects, then
       EGL_NO_SYNC_KHR is returned and an EGL_BAD_ATTRIBUTE error is
       generated.

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

    1) Does this extension need to introduce eglWaitSync
       functionality?

    RESOLVED: The EGL_KHR_wait_sync extension introduces this, to allow
    server-side synchronization, without blocking the client from issuing
    commands. Whilst this is not a required dependency, GPU-to-GPU
    synchronization is the most likely use of this extension.

    2) What should the command to create a sync object linked to an
       OpenCL event look like?

    RESOLVED: We reuse the general attribute list mechanism rather than
    having a constructor specific to CL events. This was intended in the
    sync object design from the start.

    3) How will the OpenCL header dependencies interact with
       specifying the API for this extension?

    DISCUSSION: To use this extension, OpenCL event handles of type cl_event
    are specified in the attribute lists passed to eglCreateSyncKHR. Because
    no formal parameters are of type cl_event, the EGL headers do not need
    to define this type. Applications must #include the appropriate OpenCL
    header files as well as <EGL/eglext.h> when using this extension.

    This issue resolution is consistent with the equivalent issue for
    GL_ARB_cl_event.

    4) Should all possible statuses of the CL event be reflected through to the
       state of the sync object?

    DISCUSSION: CL event objects have four execution statuses:
    CL_QUEUED, CL_SUBMITTED, CL_RUNNING, and CL_COMPLETE. GL sync
    objects have only two statuses: UNSIGNALED and SIGNALED. The
    cl_khr_gl_event extension maps UNSIGNALED into CL_SUBMITTED, and
    SIGNALED into CL_COMPLETE.

    RESOLVED: Invert the cl_khr_egl_event mapping. CL_QUEUED,
    CL_SUBMITTED, and CL_RUNNING all map into UNSIGNALED.
    CL_COMPLETE maps into SIGNALED.

    This issue resolution is consistent with the equivalent issue for
    GL_ARB_cl_event.

    5) Are there any restrictions on the use of a sync object linked to a CL
       event object?

    RESOLVED: No restrictions.

    6) How are sync object lifetimes defined?

    RESOLVED: A sync object linked to a CL event object places a single
    reference on the event. Deleting the sync object removes that reference.

    eglDestroySync has a dependency on the completion of the linked event
    object, and will not delete the sync objectwhile the event object has not
    yet completed. This is equivalent to behavior of deleting a fence sync
    object, where deletion of the object will be deferred until the underlying
    fence command has completed.

    This issue resolution is consistent with the equivalent issue for
    GL_ARB_cl_event.

    7) Should all OpenCL events be supported?

    RESOLVED: No. Only events returned by clEnqueueReleaseGLObjects, or
    clEnqueueReleaseEGLObjects since those are the only known use cases for
    this extension.

    8) Why has this extension been obsoleted and replaced by
    EGL_KHR_cl_event2?

    RESOLVED: Starting with the December 4, 2013 release of EGL 1.4, EGLint
    is defined to be the same size as the native platform "int" type. Handle
    and pointer attribute values *cannot* be represented in attribute lists
    on platforms where sizeof(handle/pointer) > sizeof(int). Existing
    extensions which assume this functionality are being replaced with new
    extensions specifying new entry points to work around this issue. See
    the latest EGL 1.4 Specification for more details.

Revision History

    Version 10, 2013/12/04 (Jon Leech) - add issue 8 explaining that OpenCL
    event handles cannot be safely passed in attribute lists on 64-bit
    platforms, and suggest using EGL_KHR_cl_event2 instead.

    Version 9, 2013/08/12 (Jon Leech) - remove unused cl_event type from the
    extension and from <EGL/eglext.h> (Bug 10661).

    Version 8, 2013/07/19 (Jon Leech) - assign extension number and
    missing enum value, and clean up a few typos for publication.

    Version 7, 2013/07/08 (Jon Leech) - assign enums (Bug 10490).

    Version 6, 2013/06/11 (Alon Or-bach) - typo correction

    Version 5, 2013/05/30 (Alon Or-bach) - wording cleanup

    Version 4, 2013/05/15 (Alon Or-bach) - updated issue resolutions as agreed,
    consistent with GL_ARB_cl_event, including using typedef for cl_event

    Version 3, 2013/04/25 (Alon Or-bach) - remove use of CL context,
    accept events from clEnqueueAcquireEGLObjects and minor cleanup

    Version 2, 2012/06/26 (Jon Leech) - update link to complementary CL
    extension.

    Version 1, 2010/05/18 (Jon Leech) - initial version based on
    equivalent GL_ARB_cl_event extension.
