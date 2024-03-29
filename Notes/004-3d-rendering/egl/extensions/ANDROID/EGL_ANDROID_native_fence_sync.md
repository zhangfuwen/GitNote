# ANDROID_native_fence_sync

Name

    ANDROID_native_fence_sync

Name Strings

    EGL_ANDROID_native_fence_sync

Contributors

    Jamie Gennis

Contact

    Jamie Gennis, Google Inc. (jgennis 'at' google.com)

Status

    Complete

Version

    Version 3, September 4, 2012

Number

    EGL Extension #50

Dependencies

    Requires EGL 1.1

    This extension is written against the wording of the EGL 1.2 Specification

    EGL_KHR_fence_sync is required.

Overview

    This extension enables the creation of EGL fence sync objects that are
    associated with a native synchronization fence object that is referenced
    using a file descriptor.  These EGL fence sync objects have nearly
    identical semantics to those defined by the KHR_fence_sync extension,
    except that they have an additional attribute storing the file descriptor
    referring to the native fence object.

    This extension assumes the existence of a native fence synchronization
    object that behaves similarly to an EGL fence sync object.  These native
    objects must have a signal status like that of an EGLSyncKHR object that
    indicates whether the fence has ever been signaled.  Once signaled the
    native object's signal status may not change again.

New Types

    None.

New Procedures and Functions

    EGLint eglDupNativeFenceFDANDROID(
                        EGLDisplay dpy,
                        EGLSyncKHR);

New Tokens

    Accepted by the <type> parameter of eglCreateSyncKHR, and returned
    in <value> when eglGetSyncAttribKHR is called with <attribute>
    EGL_SYNC_TYPE_KHR:

    EGL_SYNC_NATIVE_FENCE_ANDROID          0x3144

    Accepted by the <attrib_list> parameter of eglCreateSyncKHR:

    EGL_SYNC_NATIVE_FENCE_FD_ANDROID       0x3145

    Accepted by the <attrib_list> parameter of eglCreateSyncKHR, and returned
    by eglDupNativeFenceFDANDROID in the event of an error:

    EGL_NO_NATIVE_FENCE_FD_ANDROID         -1

    Returned in <value> when eglGetSyncAttribKHR is called with <attribute>
    EGL_SYNC_CONDITION_KHR:

    EGL_SYNC_NATIVE_FENCE_SIGNALED_ANDROID 0x3146

Changes to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add the following after the sixth paragraph of Section 3.8.1 (Sync
    Objects), added by KHR_fence_sync

    "If <type> is EGL_SYNC_NATIVE_FENCE_ANDROID, an EGL native fence sync
    object is created. In this case the EGL_SYNC_NATIVE_FENCE_FD_ANDROID
    attribute may optionally be specified. If this attribute is specified, it
    must be set to either a file descriptor that refers to a native fence
    object or to the value EGL_NO_NATIVE_FENCE_FD_ANDROID.

    The default values for the EGL native fence sync object attributes are as
    follows:

      Attribute Name                     Initial Attribute Value(s)
      ---------------                    --------------------------
      EGL_SYNC_TYPE_KHR                  EGL_SYNC_NATIVE_FENCE_ANDROID
      EGL_SYNC_STATUS_KHR                EGL_UNSIGNALED_KHR
      EGL_SYNC_CONDITION_KHR             EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR
      EGL_SYNC_NATIVE_FENCE_FD_ANDROID   EGL_NO_NATIVE_FENCE_FD_ANDROID

    If the EGL_SYNC_NATIVE_FENCE_FD_ANDROID attribute is not
    EGL_NO_NATIVE_FENCE_FD_ANDROID then the EGL_SYNC_CONDITION_KHR attribute is
    set to EGL_SYNC_NATIVE_FENCE_SIGNALED_ANDROID and the EGL_SYNC_STATUS_KHR
    attribute is set to reflect the signal status of the native fence object.
    Additionally, the EGL implementation assumes ownership of the file
    descriptor, so the caller must not use it after calling eglCreateSyncKHR."

    Modify Section 3.8.1 (Sync Objects), added by KHR_fence_sync, starting at
    the seventh paragraph

    "When a fence sync object is created or when an EGL native fence sync
    object is created with the EGL_SYNC_NATIVE_FENCE_FD_ANDROID attribute set
    to EGL_NO_NATIVE_FENCE_FD_ANDROID, eglCreateSyncKHR also inserts a fence
    command into the command stream of the bound client API's current context
    (i.e., the context returned by eglGetCurrentContext), and associates it
    with the newly created sync object.

    After associating the fence command with an EGL native fence sync object,
    the next Flush() operation performed by the current client API causes a
    new native fence object to be created, and the
    EGL_SYNC_NATIVE_FENCE_ANDROID attribute of the EGL native fence object is
    set to a file descriptor that refers to the new native fence object. This
    new native fence object is signaled when the EGL native fence sync object
    is signaled.

    When the condition of the sync object is satisfied by the fence command,
    the sync is signaled by the associated client API context, causing any
    eglClientWaitSyncKHR commands (see below) blocking on <sync> to unblock.
    If the sync object is an EGL native fence sync object then the native
    fence object is also signaled when the condition is satisfied. The
    EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR condition is satisfied by completion
    of the fence command corresponding to the sync object and all preceding
    commands in the associated client API context's command stream. The sync
    object will not be signaled until all effects from these commands on the
    client API's internal and framebuffer state are fully realized. No other
    state is affected by execution of the fence command.

    The EGL_SYNC_NATIVE_FENCE_SIGNALED_ANDROID condition is satisfied by the
    signaling of the native fence object referred to by the
    EGL_SYNC_NATIVE_FENCE_FD_ANDROID attribute. When this happens any
    eglClientWaitSyncKHR commands blocking on <sync> unblock."

    Modify the list of eglCreateSyncKHR errors in Section 3.8.1 (Sync Objects),
    added by KHR_fence_sync

    "Errors
    ------

      * If <dpy> is not the name of a valid, initialized EGLDisplay,
        EGL_NO_SYNC_KHR is returned and an EGL_BAD_DISPLAY error is
        generated.
      * If <type> is EGL_SYNC_FENCE_KHR and <attrib_list> is neither NULL nor
        empty (containing only EGL_NONE), EGL_NO_SYNC_KHR is returned and an
        EGL_BAD_ATTRIBUTE error is generated.
      * If <type> is EGL_SYNC_NATIVE_FENCE_ANDROID and <attrib_list> contains
        an attribute other than EGL_SYNC_NATIVE_FENCE_FD_ANDROID,
        EGL_NO_SYNC_KHR is returned and an EGL_BAD_ATTRIBUTE error is
        generated.
      * If <type> is not a supported type of sync object,
        EGL_NO_SYNC_KHR is returned and an EGL_BAD_ATTRIBUTE error is
        generated.
      * If <type> is EGL_SYNC_FENCE_KHR or EGL_SYNC_NATIVE_FENCE_ANDROID and
        no context is current for the bound API (i.e., eglGetCurrentContext
        returns EGL_NO_CONTEXT), EGL_NO_SYNC_KHR is returned and an
        EGL_BAD_MATCH error is generated.
      * If <type> is EGL_SYNC_FENCE_KHR or EGL_SYNC_NATIVE_FENCE_ANDROID and
        <dpy> does not match the EGLDisplay of the currently bound context for
        the currently bound client API (the EGLDisplay returned by
        eglGetCurrentDisplay()) then EGL_NO_SYNC_KHR is returned and an
        EGL_BAD_MATCH error is generated.
      * If <type> is EGL_SYNC_FENCE_KHR or EGL_SYNC_NATIVE_FENCE_ANDROID and
        the currently bound client API does not support the client API
        extension indicating it can place fence commands, then EGL_NO_SYNC_KHR
        is returned and an EGL_BAD_MATCH error is generated."

    Modify table 3.cc in Section 3.8.1 (Sync Objects), added by KHR_fence_sync

    "
    Attribute                          Description                Supported Sync Objects
    -----------------                  -----------------------    ----------------------
    EGL_SYNC_TYPE_KHR                  Type of the sync object    All
    EGL_SYNC_STATUS_KHR                Status of the sync object  All
    EGL_SYNC_CONDITION_KHR             Signaling condition        EGL_SYNC_FENCE_KHR and
                                                                  EGL_SYNC_NATIVE_FENCE_ANDROID only
    "

    Modify the second paragraph description of eglDestroySyncKHR in Section
    3.8.1 (Sync Objects), added by KHR_fence_sync

    "If no errors are generated, EGL_TRUE is returned, and <sync> will no
    longer be the handle of a valid sync object.  Additionally, if <sync> is an
    EGL native fence sync object and the EGL_SYNC_NATIVE_FENCE_FD_ANDROID
    attribute is not EGL_NO_NATIVE_FENCE_FD_ANDROID then that file descriptor
    is closed."

    Add the following after the last paragraph of Section 3.8.1 (Sync
    Objects), added by KHR_fence_sync

    The command

        EGLint eglDupNativeFenceFDANDROID(
                            EGLDisplay dpy,
                            EGLSyncKHR sync);

    duplicates the file descriptor stored in the
    EGL_SYNC_NATIVE_FENCE_FD_ANDROID attribute of an EGL native fence sync
    object and returns the new file descriptor.

    Errors
    ------

      * If <sync> is not a valid sync object for <dpy>,
        EGL_NO_NATIVE_FENCE_FD_ANDROID is returned and an EGL_BAD_PARAMETER
        error is generated.
      * If the EGL_SYNC_NATIVE_FENCE_FD_ANDROID attribute of <sync> is
        EGL_NO_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID is
        returned and an EGL_BAD_PARAMETER error is generated.
      * If <dpy> does not match the display passed to eglCreateSyncKHR
        when <sync> was created, the behaviour is undefined."

Issues

    1. Should EGLSyncKHR objects that wrap native fence objects use the
    EGL_SYNC_FENCE_KHR type?

    RESOLVED: A new sync object type will be added.

    We don't want to require all EGL fence sync objects to wrap native fence
    objects, so we need some way to tell the EGL implementation at sync object
    creation whether the sync object should support querying the native fence
    FD attribute. We could do this with either a new sync object type or with a
    boolean attribute. It might be nice to pick up future signaling conditions
    that might be added for fence sync objects, but there may be things that
    get added that don't make sense in the context of native fence objects.

    2. Who is responsible for dup'ing the native fence file descriptors?

    RESOLVED: Whenever a file descriptor is passed into or returned from an
    EGL call in this extension, ownership of that file descriptor is
    transfered. The recipient of the file descriptor must close it when it is
    no longer needed, and the provider of the file descriptor must dup it
    before providing it if they require continued use of the native fence.

    3. Can the EGL_SYNC_NATIVE_FENCE_FD_ANDROID attribute be queried?

    RESOLVED: No.

    Returning the file descriptor owned by the EGL implementation would
    violate the file descriptor ownership rule described in issue #2. Having
    eglGetSyncAttribKHR return a different (dup'd) file descriptor each time
    it's called seems wrong, so a new function was added to explicitly dup the
    file descriptor.

    That said, the attribute is useful both as a way to pass an existing file
    descriptor to eglCreateSyncKHR and as a way to describe the subsequent
    behavior of EGL native fence sync objects, so it is left as an attribute
    for which the value cannot be queried.

Revision History

#3 (Jamie Gennis, September 4, 2012)
    - Reworded the extension to refer to "native fence" objects rather than
    "Android fence" objects.
    - Added a paragraph to the overview that describes assumptions about the
    native fence sync objects.

#2 (Jamie Gennis, July 23, 2012)
    - Changed the file descriptor ownership transferring behavior.
    - Added the eglDupAndroidFenceFDANDROID function.
    - Removed EGL_SYNC_NATIVE_FENCE_FD_ANDROID from the table of gettable
    attributes.
    - Added language specifying that a native Android fence is created at the
    flush following the creation of an EGL Android fence sync object that is
    not passed an existing native fence.

#1 (Jamie Gennis, May 29, 2012)
    - Initial draft.
