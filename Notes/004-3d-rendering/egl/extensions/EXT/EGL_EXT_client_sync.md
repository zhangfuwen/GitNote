# EXT_client_sync

Name

    EXT_client_sync

Name Strings

    EGL_EXT_client_sync

Contributors

    Daniel Kartch

Contacts

    Daniel Kartch, NVIDIA Corporation (dkartch 'at' nvidia.com)

Status

    Complete

Version

    Version 2, April 20, 2018

Number

    EGL Extension #129

Extension type

    EGL display extension

Dependencies

    Requires EGL_EXT_sync_reuse

Overview

    The EGL_KHR_reusable_sync extension defines an EGL_SYNC_REUSABLE_KHR
    EGLSync type which is signaled and unsignaled by client events. The
    EGL_EXT_sync_reuse extension allows all EGLSyncs to become reusable.
    The signaling behavior associated with EGL_SYNC_REUSABLE_KHR is
    still desirable, but the name becomes misleading if all EGLSyncs can
    be reused. This extension defines an EGLSync type with equivalent
    behavior, separating the signaling mechanism from the reusability.

New Procedures and Functions

    EGLBoolean eglClientSignalSyncEXT(
                        EGLDisplay dpy,
                        EGLSync sync,
                        const EGLAttrib *attrib_list);

New Types

    None

New Tokens

    Accepted by the <type> parameter of eglCreateSync, and returned
    in <value> when eglGetSyncAttrib is called with <attribute>
    EGL_SYNC_TYPE:

    EGL_SYNC_CLIENT_EXT                   0x3364

    Returned in <value> when eglGetSyncAttrib is called with attribute
    EGL_SYNC_CONDITION:

    EGL_SYNC_CLIENT_SIGNAL_EXT            0x3365

Add to the list of sync object decriptions in 3.8.1 Sync Objects

    A <client sync object> reflects the readiness of some client-side
    state. Sync objects of this type are not visible to API contexts and
    may not be used with eglWaitSync. They may be waited for with
    eglClientWaitSync or polled with eglGetSyncAttrib as other sync
    types.

Add to the end of 3.8.1 Sync Objects

    The command

        EGLBoolean eglClientSignalSyncEXT(EGLDisplay dpy, EGLSync sync,
            const EGLAttrib *attrib_list);

    may be called to switch sync objects which support it to the
    signaled state. Currently only sync objects with type
    EGL_SYNC_CLIENT_EXT provide this support. The attribute list may be
    used to provide additional information to the signaling operation,
    as defined for the sync type.

    Errors

        eglClientSignalSyncEXT returns EGL_FALSE on failure, and has no
        effect on <sync>.
        If <dpy> is not the name of a valid, initialized EGLDisplay, an
        EGL_BAD_DISPLAY error is generated.
        If <sync> is not a valid sync object associated with <dpy>, an
        EGL_BAD_PARAMETER error is generated.
        If <attrib_list> contains an attribute name not defined for the
        type of <sync>, an EGL_BAD_ATTRIBUTE error is generated.
        If <sync>'s type does not support this direct signaling, an
        EGL_BAD_ACCESS error is generated.

Insert new subsection in 3.8.1 Sync Objects

    3.8.1.x Creating and Signaling Client Sync Objects

    If type is EGL_SYNC_CLIENT_EXT, a client sync object is created. The
    EGL_SYNC_STATUS attribute may be specified as either EGL_UNSIGNALED
    or EGL_SIGNALED, and will default to EGL_UNSIGNALED. No other
    attributes may be specified for a client sync object. The value of
    EGL_SYNC_CONDITION will be set to EGL_SYNC_CLIENT_SIGNAL_EXT.

    A client sync object in the unsignaled state will switch to the
    signaled state when eglClientSignalSyncEXT is called. No attributes
    are supported for signaling a sync object of this type. Signaling a
    client sync object which is already in the signaled state will have
    no effect.

    A client sync object which is in the signaled state may be switched
    back to the unsignaled state with eglUnsignalSyncEXT. No attributes
    are supported for unsignaling a sync object of this type.

Add to the error list for eglWaitSync in 3.8.1.3 Waiting for Sync
Objects

    If <sync> is of type EGL_SYNC_CLIENT_EXT, an EGL_BAD_ACCESS error is
    generated.

Issues

    None

Revision History

    #2 (April 20, 2018) Daniel Kartch
       - Renamed to EXT
       - Fixed missing attrib_list in New Functions section
       - Eliminated condition as an allowed attribute at creation. This
         is inconsistent with other sync extensions, and there is no
         need to make it configurable at this time. Future extensions
         can make the condition configurable if desired.

    #1 (Feburary 22, 2018) Daniel Kartch
       - Initial draft as XXX
