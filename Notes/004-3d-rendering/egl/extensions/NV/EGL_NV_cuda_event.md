# NV_cuda_event

Name

    NV_cuda_event

Name Strings

    EGL_NV_cuda_event

Contributors

    Debalina Bhattacharjee
    Michael Chock
    James Jones
    Daniel Kartch

Contact

    Michael Chock (mchock 'at' nvidia.com)

Status

    Complete

Version

    Version 2, June 28, 2018

Number

    EGL Extension #75

Extension Type

    EGL display extension

Dependencies

    This extension is written against the language of EGL 1.5 and the
    EGL_EXT_sync_reuse extension.

    Either EGL_KHR_fence_sync and the EGLAttrib type or EGL 1.5 are
    required.

    This extension interacts with, but does not require,
    EGL_EXT_sync_reuse.

    This extension interacts with EGL_NV_device_cuda.

Overview

    This extension allows creating an EGL sync object linked to a CUDA
    event object, potentially improving efficiency of sharing images and
    compute results between the two APIs.

IP Status

    No known claims.

New Types

    A pointer to type cudaEvent_t, defined in the CUDA header files, may
    be included in the attribute list passed to eglCreateSync.

New Procedures and Functions

    None.

New Tokens

    Accepted as attribute names in the <attrib_list> argument
    of eglCreateSync:

        EGL_CUDA_EVENT_HANDLE_NV        0x323B

    Returned in <values> for eglGetSyncAttrib <attribute>
    EGL_SYNC_TYPE:

        EGL_SYNC_CUDA_EVENT_NV          0x323C

    Returned in <values> for eglGetSyncAttrib <attribute>
    EGL_SYNC_CONDITION:

        EGL_SYNC_CUDA_EVENT_COMPLETE_NV 0x323D

Add to section 3.8.1 (Sync Objects) of the EGL 1.5 specification, after
the sixth paragraph:

    Likewise, a <CUDA event sync object> reflects the status of a
    corresponding CUDA object. Waiting on this type of sync object is
    equivalent to waiting for completion of the corresponding linked
    CUDA event object.

Add a new section following section 3.8.1.2 (Creating and Signaling
OpenCL Event Sync Objects):

    Section 3.8.1.X Creating and Signaling CUDA Event Sync Objects

    If <type> is EGL_SYNC_CUDA_EVENT_NV, a CUDA event sync object is
    created. The <attrib_list> may contain the attribute
    EGL_CUDA_EVENT_HANDLE_NV, set to a pointer to a cudaEvent_t object.
    If it does not contain this attribute, the sync object will start in
    the signaled state, and an event attribute must be provided the
    first time eglUnsignalSyncEXT is called. Otherwise, a call to
    eglUnsignalSyncEXT may replace this event attribute or leave it
    unspecified, causing the previous object to be reused.

    A cudaEvent_t object provided to eglCreateSync or eglUnsignalSyncEXT
    must be properly initialized and recorded by the CUDA API (using
    cudaEventCreate and cudaEventRecord), and the CUDA device
    used to create the event must correspond to <dpy>[fn1]. Note that
    EGL_CUDA_EVENT_HANDLE_NV is not a queryable property of a sync
    object.

    [fn1] If EGL_NV_device_cuda is supported, it is sufficient that the
          CUDA device used to create the CUDA event matches the
          EGL_CUDA_DEVICE_NV attribute of <dpy>'s underlying EGL
          device.

    Attributes of the CUDA event sync object are set as follows:

        Attribute Name          Initial Attribute Value(s)
        -------------           --------------------------
        EGL_SYNC_TYPE           EGL_SYNC_CUDA_EVENT_NV
        EGL_SYNC_STATUS         Depends on status of <event>
        EGL_SYNC_CONDITION      EGL_SYNC_CUDA_EVENT_COMPLETE_NV

    If an <event> is linked to the sync object, the status of this type
    of sync object depends on the state of <event> evaluated at the time
    of the most recent call to eglCreateSync or eglUnsignalSyncEXT. If
    all device work preceding the most recent call to cudaEventRecord on
    the event has not yet completed, the status of the linked sync
    object will be EGL_UNSIGNALED. If all such work has completed, the
    status of the linked sync object will be EGL_SIGNALED. Calling
    cudaEventRecord to modify an event has no effect on the sync object
    while its status is EGL_UNSIGNALED, but will have an effect if the
    event is reevaluated at a subsequent eglUnsignalSyncEXT call.

    The only condition supported for CUDA event sync objects is
    EGL_SYNC_CUDA_EVENT_COMPLETE_NV. It is satisfied when all device
    work prior to the most recent call to cudaEventRecord at sync
    unsignaling time has completed.

If EGL_EXT_sync_reuse is not present, then change the second sentence of
3.8.1.X above to "The <attrib_list> must contain ..." and omit the
remaining sentences in the paragraph. Omit all references to
eglUnsignalSyncEXT.

In 3.8.1 (Sync Objects), if EGL_EXT_sync_reuse is present, then add the
following to the error list for eglUnsignalSyncEXT. Otherwise add it to
the error list for eglCreateSync:

    If <type> is EGL_SYNC_CUDA_EVENT_NV and a EGL_CUDA_EVENT_HANDLE_NV
    is not linked to the sync object, then an EGL_BAD_ATTRIBUTE error is
    generated. If its attribute value is not a valid CUDA event pointer
    or has not been initialized as described above, then
    EGL_BAD_ATTRIBUTE may be generated, but the results are undefined
    and may include program termination.

Modify the third paragraph of section 3.8.1.4 (Querying Sync Object
Attributes):

    If any eglClientWaitSync or eglWaitSync commands are blocking on
    <sync> when eglDestroySync is called, <sync> is flagged for deletion
    and will be deleted when the associated fence command, OpenCL event
    object, or CUDA event object has completed, and <sync> is no longer
    blocking any such egl*WaitSync command. Otherwise, the sync object
    is destroyed immediately.

Replace the EGL_SYNC_CONDITION row of table 3.9 with:

    Attribute           Description              Supported Sync Objects
    ------------------  -----------------------  ----------------------
    EGL_SYNC_CONDITION  Signaling condition      EGL_SYNC_FENCE,
                                                 EGL_SYNC_CL_EVENT, or
                                                 EGL_SYNC_CUDA_EVENT_NV

    Table 3.9  Attributes Accepted by eglGetSyncAttrib


Interactions with EGL versions prior to 1.5

    This extension may be used with earlier versions of EGL, provided
    that the EGL_KHR_fence_sync extension is supported. In this case,
    replace all references to sync functions and tokens with
    corresponding KHR-suffixed versions (e.g., replace eglCreateSync
    with eglCreateSyncKHR).

    Additionally, this extension may be used with the 64-bit types and
    functions added to EGL_KHR_fence_sync introduced by
    EGL_KHR_cl_event2 (EGLAttribKHR and eglCreateSync64KHR). Support
    for OpenCL events is not required.

Issues

    None

Revision History

    Version 2, 2018/06/28 (Daniel Kartch)
        - Rewritten to clearly define interactions with
          EGL_EXT_sync_reuse, without requiring it
        - Fixed incorrect CUDA function name
        - Fixed table spacing

    Version 1, 2014/06/20 (Michael Chock)
        - initial version.
